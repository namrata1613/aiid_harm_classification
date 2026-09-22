# AIID harm classification

Can a small, free, locally-run language model reproduce the human harm
classifications in the [AI Incident Database](https://incidentdatabase.ai/)? This
repo runs that experiment end to end and measures the agreement.

It classifies each incident into the [MIT AI Risk Repository](https://airisk.mit.edu/risks)
taxonomy (7 domains, ~24 subdomains) using Llama 3.1 8B through
[Ollama](https://ollama.com/), then scores the model's labels against AIID's
human-assigned ones.

The full write-up of the findings is in [`writeup.md`](writeup.md).

## Result

Over 1,495 labeled incidents, zero-shot, with no malformed outputs:

| Granularity | Accuracy | Cohen's kappa |
|---|---|---|
| 24-way subdomain | 47.6% | 0.41 |
| 7-domain (collapsed) | 62.1% | 0.51 |

The model reliably gets the broad harm domain right and is shakier on the exact
subdomain. Adding 3 few-shot examples did not help (subdomain kappa fell to 0.36),
likely from label bias on a many-class task. Agreement is strong on concrete
categories (fraud, cyberattacks, discrimination) and weak on diffuse ones
(misinformation, privacy, socioeconomic).

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com/) installed and running.

```bash
pip install -r requirements.txt
ollama pull llama3.1
```

## Run

```bash
python run.py --sample 200      # quick pass
python run.py --sample 2000     # all labeled incidents (~1,500)
```

`run.py` chains three steps, each of which also runs standalone:

- `python -m src.ingest`    download + parse the AIID Excel export into a clean table
- `python -m src.classify`  ask the model for one label per incident (cached, resumable)
- `python -m src.validate`  score against human labels, at subdomain and domain level

Outputs land in `data/`: `validation_report.json`, `confusion_matrix.png`,
`confusion_matrix_domains.png`, and the intermediate tables.

## Configuration

Everything is in `config.yaml`. The knobs you'll touch most:

- `classify.provider` — `ollama` (default, local, free), or `grok` / `openai` /
  `anthropic` if you set the matching API key in your environment
- `classify.model` — e.g. `llama3.1`
- `classify.few_shot_k` — `0` for zero-shot, or `N` to prepend N held-out examples
- `classify.requests_per_second` — raise it for local Ollama (no rate limit)
- `classify.sample` — how many labeled incidents to classify

Changing the model, provider, or few-shot setting automatically invalidates the
prediction cache, so you never mix results from different configurations.

## How it works

`ingest.py` auto-detects the header row in AIID's human-formatted Excel export and
pulls out `date`, `year`, `title`, `description`, and the target taxonomy column
(`Risk Subdomain`, filled for ~89% of incidents). `classify.py` sends title +
description plus the allowed label list to the model and caches every answer to
`data/pred_cache.json`, so a run can be stopped and resumed. `validate.py` computes
accuracy and Cohen's kappa at the subdomain level, then collapses the 24 subdomains
to the 7 MIT domains (`taxonomy.py`) and reports that level too.

`profile_aiid.py` is a standalone helper that prints the raw schema of the Excel
export, useful if a future export changes shape.

## Data

AIID publishes weekly snapshots at
[incidentdatabase.ai/research/snapshots](https://incidentdatabase.ai/research/snapshots).
The default config points at one dated export; update the URL for a fresher one.
Incident data is © the Responsible AI Collaborative under AIID's terms of use.

## Limitations

- Agreement is measured against AIID's human labels, which have their own noise.
  It is consistency with a taxonomy, not ground truth about the world.
- AIID over-represents English-language, media-covered incidents, so it is a
  sample of reported harms, not all harms.
- This is one 8B model at temperature 0. A larger model would likely do better.

## License

Code under the MIT License (see `LICENSE`). Incident data and the MIT taxonomy
carry their own terms; see the links above.
