"""Ingest the AIID Excel 'Incidents' sheet into a clean incidents table.

The export is human-formatted: the real column names are on the 2nd sheet row
(hence header_row=1), the first column's header is a section banner but its cells
hold the incident id, and section-banner cells are interleaved. We therefore
select the columns we need by tolerant name-matching rather than by position.

Output: data/incidents.parquet + data/incidents.csv with columns
  incident_id, date, year, title, description, target  (target = the taxonomy label)
"""

import argparse
import sys

import pandas as pd
from typing import Optional
from .common import load_config, data_path, norm


def _read_excel(cfg: dict) -> pd.DataFrame:
    from .common import _ROOT
    fpath = _ROOT / cfg["data"]["excel_file"]     # resolve relative to repo root
    if not fpath.exists():
        url = cfg["data"]["excel_url"]
        print(f"[ingest] {fpath} not found; downloading {url}")
        import urllib.request
        fpath.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, fpath)

    # The export has several banner/title rows before the real field-name row.
    # Auto-detect the header: the first of the first ~8 rows that contains both
    # 'title' and 'description'. Robust to layout shifts between weekly exports.
    raw = pd.read_excel(fpath, sheet_name=cfg["data"]["sheet"], header=None)
    header_idx = None
    for i in range(min(8, len(raw))):
        vals = {norm(v) for v in raw.iloc[i].tolist()}
        if {"title", "description"} <= vals:
            header_idx = i
            break
    if header_idx is None:
        preview = "\n".join(f"  row {i}: {raw.iloc[i].tolist()[:8]}"
                            for i in range(min(6, len(raw))))
        raise SystemExit("[ingest] couldn't locate the header row (no row with "
                         f"'title' and 'description'). First rows:\n{preview}")

    print(f"[ingest] header row detected at sheet row {header_idx + 1}")
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = raw.iloc[header_idx].tolist()
    return df.reset_index(drop=True)


def _find_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    target = norm(wanted)
    for c in df.columns:
        if norm(c) == target:
            return c
    return None


def main() -> pd.DataFrame:
    cfg = load_config()
    f = cfg["fields"]
    df = _read_excel(cfg)

    # first column carries the incident id (its field-name cell may be blank);
    # rename positionally to avoid clashing with any NaN-labelled banner columns
    cols = list(df.columns)
    cols[0] = "incident_id"
    df.columns = cols

    wanted = {
        "date": f["date"], "year": f["year"], "title": f["title"],
        "description": f["description"], "target": f["target_taxonomy"],
    }
    resolved, missing = {"incident_id": "incident_id"}, []
    for key, colname in wanted.items():
        hit = _find_col(df, colname)
        if hit is None:
            missing.append(colname)
        else:
            resolved[key] = hit
    if missing:
        sys.exit(f"[ingest] could not find columns {missing}.\n"
                 f"Available columns:\n  " + "\n  ".join(map(str, df.columns)))

    out = df[[resolved[k] for k in ["incident_id", "date", "year", "title",
                                    "description", "target"]]].copy()
    out.columns = ["incident_id", "date", "year", "title", "description", "target"]

    # drop the banner/leftover header rows and empty incidents
    out = out[out["incident_id"].notna()]
    out = out[~out["incident_id"].astype(str).str.contains("INCIDENT IDENTITY", na=False)]
    out = out.dropna(subset=["title", "description"], how="all")

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    labeled = out["target"].notna().sum()
    dest = data_path("incidents.parquet")
    out.to_parquet(dest, index=False)
    out.to_csv(data_path("incidents.csv"), index=False)
    print(f"[ingest] {len(out)} incidents -> {dest}")
    print(f"[ingest] target taxonomy '{cfg['fields']['target_taxonomy']}': "
          f"{labeled} labeled ({labeled/max(len(out),1):.0%}), "
          f"{out['target'].nunique(dropna=True)} distinct classes")
    return out


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()