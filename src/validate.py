"""Measure how well the automation reproduces AIID's human taxonomy labels.

This is the project's "automation vs expert classification" test, using AIID's
existing human labels as ground truth. Reports:
  - overall agreement (accuracy)
  - Cohen's kappa (chance-corrected agreement — the honest headline number)
  - per-class precision/recall/F1
  - a confusion matrix figure

Output: data/validation_report.json + data/confusion_matrix.png
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             classification_report, confusion_matrix)

from .common import load_config, data_path
from .taxonomy import domain_of


def _metrics(gold, pred):
    return (round(float(accuracy_score(gold, pred)), 4),
            round(float(cohen_kappa_score(gold, pred)), 4))


def main() -> dict:
    cfg = load_config()
    df = pd.read_parquet(data_path("predictions.parquet"))

    # exclude rows the model failed to map, but report how many there were
    unmapped = int((df["pred"].isin(["UNMAPPED", "ERROR"])).sum())
    ok = df[~df["pred"].isin(["UNMAPPED", "ERROR"])]

    gold, pred = ok["gold"], ok["pred"]
    acc, kappa = _metrics(gold, pred)
    report = classification_report(gold, pred, output_dict=True, zero_division=0)

    # --- domain-level (collapse 22 subdomains -> 7 MIT domains) --------------
    g_dom = gold.map(domain_of)
    p_dom = pred.map(domain_of)
    unknown = sorted(set(gold[g_dom.isna()]).union(pred[p_dom.isna()]))
    dom_mask = g_dom.notna() & p_dom.notna()
    dom_acc, dom_kappa = _metrics(g_dom[dom_mask], p_dom[dom_mask])
    if unknown:
        print(f"[validate] {len(unknown)} label(s) not in the domain map "
              f"(excluded from domain metrics): {unknown}")

    results = {
        "target_taxonomy": cfg["fields"]["target_taxonomy"],
        "provider": cfg["classify"]["provider"],
        "model": cfg["classify"]["model"],
        "n_evaluated": int(len(ok)),
        "n_unmapped_or_error": unmapped,
        "subdomain_level": {"n_classes": int(gold.nunique()),
                            "accuracy": acc, "cohen_kappa": kappa},
        "domain_level": {"n_classes": int(g_dom[dom_mask].nunique()),
                         "n_evaluated": int(dom_mask.sum()),
                         "accuracy": dom_acc, "cohen_kappa": dom_kappa,
                         "labels_unmapped_to_domain": unknown},
        # keep flat keys too for backward compatibility
        "accuracy": acc,
        "cohen_kappa": kappa,
        "headline": (f"The {cfg['classify']['model']} classifier reproduces AIID's "
                     f"human labels with {acc:.0%} agreement (kappa={kappa:.2f}) at "
                     f"the {gold.nunique()}-way subdomain level, rising to "
                     f"{dom_acc:.0%} (kappa={dom_kappa:.2f}) at the 7-domain level, "
                     f"over {len(ok)} incidents."),
    }
    dest = data_path("validation_report.json")
    dest.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # confusion matrix (top classes by support, to stay readable)
    labels = gold.value_counts().head(15).index.tolist()
    cm = confusion_matrix(gold, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted (LLM)"); ax.set_ylabel("gold (AIID human)")
    ax.set_title("Agreement on top-15 classes")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(data_path("confusion_matrix.png"), dpi=150)

    # domain-level confusion matrix (7 classes -> readable)
    dl = pd.DataFrame({"g": g_dom[dom_mask], "p": p_dom[dom_mask]})
    dlabels = sorted(dl["g"].unique())
    cmd = confusion_matrix(dl["g"], dl["p"], labels=dlabels)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im2 = ax2.imshow(cmd, cmap="Greens")
    ax2.set_xticks(range(len(dlabels))); ax2.set_yticks(range(len(dlabels)))
    ax2.set_xticklabels(dlabels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(dlabels, fontsize=8)
    ax2.set_xlabel("predicted domain (LLM)"); ax2.set_ylabel("gold domain (AIID human)")
    ax2.set_title("Agreement at the 7-domain level")
    fig2.colorbar(im2, fraction=0.046)
    fig2.tight_layout()
    fig2.savefig(data_path("confusion_matrix_domains.png"), dpi=150)

    print(f"[validate] {results['headline']}")
    print(f"[validate] reports + confusion matrices -> {dest.parent}")
    return results


if __name__ == "__main__":
    main()