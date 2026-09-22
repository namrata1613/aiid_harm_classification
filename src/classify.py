"""Automate the taxonomy classification: an LLM reads title + description and
picks one label from the target taxonomy's own label set.

Providers: "anthropic" (ANTHROPIC_API_KEY), "openai" (OPENAI_API_KEY), or
"mock" (no API — assigns a deterministic pseudo-label, for wiring tests only).

No GPU. Results are cached to data/pred_cache.json so re-runs are free and a
rate-limit stop is recoverable.

Output: data/predictions.parquet (incident_id, gold, pred)
"""

import argparse
import json
import random
import time
from typing import Optional
import pandas as pd
from tqdm import tqdm

from .common import load_config, data_path

SYS = ("You classify AI incidents into a fixed taxonomy. Read the incident and "
       "reply with EXACTLY ONE label copied verbatim from the allowed list. "
       "No explanation, no punctuation — just the label.")


def _prompt(title, desc, labels, examples=None):
    label_block = "\n".join(f"- {l}" for l in labels)
    demo = ""
    if examples:
        parts = ["Here are worked examples:"]
        for ex in examples:
            parts.append(f"Incident title: {ex['title']}\n"
                         f"Incident description: {ex['description']}\n"
                         f"Label: {ex['label']}")
        demo = "\n\n".join(parts) + "\n\n"
    return (f"Allowed labels:\n{label_block}\n\n"
            f"{demo}"
            f"Now classify this incident.\n"
            f"Incident title: {title}\n"
            f"Incident description: {desc}\n\n"
            f"Return exactly one label from the list above.")


def _select_fewshot(df, k, seed):
    """Pick k labeled incidents as demonstrations, preferring distinct labels.
    Returns (examples list, set of their incident_ids to exclude from scoring)."""
    if not k:
        return [], set()
    shuffled = df.sample(frac=1.0, random_state=seed)
    picked, seen = [], set()
    for _, r in shuffled.iterrows():           # first pass: one per distinct label
        lab = str(r["target"])
        if lab not in seen:
            seen.add(lab)
            picked.append(r)
        if len(picked) >= k:
            break
    examples = [{"title": str(r["title"]),
                 "description": str(r["description"])[:400],
                 "label": str(r["target"])} for r in picked]
    return examples, {str(r["incident_id"]) for r in picked}


def _coerce(reply: str, labels: list[str]) -> str:
    r = reply.strip()
    for l in labels:                       # exact, then case-insensitive contains
        if r == l:
            return l
    low = r.lower()
    for l in labels:
        if l.lower() in low or low in l.lower():
            return l
    return "UNMAPPED"


class Backend:
    def __init__(self, cfg):
        c = cfg["classify"]
        self.provider = c["provider"]
        self.model = c["model"]
        self.temperature = c.get("temperature", 0)
        self.ollama_host = c.get("ollama_host", "http://localhost:11434")
        self._client = None

    def complete(self, title, desc, labels, examples=None) -> str:
        prompt = _prompt(title, desc, labels, examples)

        if self.provider == "mock":
            return random.choice(labels)

        if self.provider == "ollama":
            import requests
            r = requests.post(
                f"{self.ollama_host}/api/chat",
                json={"model": self.model, "stream": False,
                      "options": {"temperature": self.temperature},
                      "messages": [{"role": "system", "content": SYS},
                                   {"role": "user", "content": prompt}]},
                timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"]

        if self.provider == "anthropic":
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic()
            m = self._client.messages.create(
                model=self.model, max_tokens=40, temperature=self.temperature,
                system=SYS,
                messages=[{"role": "user", "content": prompt}])
            return m.content[0].text

        if self.provider in ("openai", "grok"):
            if self._client is None:
                import openai, os
                if self.provider == "grok":
                    # xAI API is OpenAI-compatible; needs XAI_API_KEY.
                    self._client = openai.OpenAI(
                        base_url="https://api.x.ai/v1",
                        api_key=os.environ.get("XAI_API_KEY"))
                else:
                    self._client = openai.OpenAI()   # uses OPENAI_API_KEY
            r = self._client.chat.completions.create(
                model=self.model, max_tokens=40, temperature=self.temperature,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt}])
            return r.choices[0].message.content

        raise ValueError(f"unknown provider {self.provider!r}")


def main(sample: Optional[int]) -> pd.DataFrame:
    cfg = load_config()
    c = cfg["classify"]
    random.seed(cfg["run"]["seed"])

    df = pd.read_parquet(data_path("incidents.parquet"))
    df = df[df["target"].notna()].copy()          # only labeled incidents (ground truth)
    labels = sorted(df["target"].astype(str).unique().tolist())

    # few-shot demonstrations, held out from scoring so there's no leakage
    k = c.get("few_shot_k", 0)
    examples, held_out = _select_fewshot(df, k, cfg["run"]["seed"])
    if held_out:
        df = df[~df["incident_id"].astype(str).isin(held_out)]

    n = sample or c["sample"]
    if n and n < len(df):
        df = df.sample(n=n, random_state=cfg["run"]["seed"]).reset_index(drop=True)

    cache_file = data_path("pred_cache.json")
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    backend = Backend(cfg)
    interval = 1.0 / max(c["requests_per_second"], 0.1)
    # settings baked into the cache key so changing model/provider/few-shot
    # invalidates stale predictions automatically
    sig = f"{c['provider']}|{c['model']}|fs{k}"

    rows, last = [], 0.0
    print(f"[classify] provider={c['provider']} model={c['model']} "
          f"labels={len(labels)} few_shot={k} incidents={len(df)}")
    for _, it in tqdm(df.iterrows(), total=len(df), desc="classify"):
        iid = str(it["incident_id"])
        key = f"{sig}|{iid}"
        if key in cache:
            pred = cache[key]
        else:
            for attempt in range(c["max_retries"]):
                try:
                    dt = time.time() - last
                    if dt < interval:
                        time.sleep(interval - dt)
                    last = time.time()
                    raw = backend.complete(str(it["title"]), str(it["description"]),
                                           labels, examples)
                    pred = _coerce(raw, labels)
                    break
                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    print(f"[classify] retry {attempt+1}/{c['max_retries']}: {e} ({wait}s)")
                    time.sleep(wait)
            else:
                pred = "ERROR"
            cache[key] = pred
            cache_file.write_text(json.dumps(cache))
        rows.append((iid, str(it["target"]), pred))

    out = pd.DataFrame(rows, columns=["incident_id", "gold", "pred"])
    dest = data_path("predictions.parquet")
    out.to_parquet(dest, index=False)
    print(f"[classify] wrote {len(out)} predictions -> {dest}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    main(sample=ap.parse_args().sample)