import os, json
import numpy as np
import pandas as pd

META_COLS = ["run_time", "dataset", "method", "backbone",
             "init_type", "norm", "seed", "fraction", "metric"]


def metric_row(run_time, h_params, data_info, metric, metrics=None, class_names=None):
    """One flat row: fixed metadata + flattened metrics. Same shape everywhere."""
    row = {
        "run_time":  run_time,
        "dataset":   data_info["dataset"],
        "method":    h_params["model"].__name__,
        "backbone":  h_params["backbone"].__name__,
        "init_type": h_params["init_type"],
        "norm":      data_info.get("bio_transf"),
        "seed":      h_params["seed"],
        "fraction":  h_params["fraction"],
        "metric":    metric,
    }
    for k, v in (metrics or {}).items():
        if k == "f1_class":
            names = class_names or [str(i) for i in range(len(v))]
            row.update({f"f1_{n}": float(x) for n, x in zip(names, v)})
        elif isinstance(v, (list, tuple, np.ndarray)):
            row[k] = json.dumps(np.asarray(v).tolist())
        elif isinstance(v, (np.floating, np.integer)):
            row[k] = float(v)
        else:
            row[k] = v
    return row


def save_rows(rows, out_csv):
    """Append rows, aligning to the existing header; rewrite if the schema grew."""
    df = pd.DataFrame(rows)
    ordered = [c for c in META_COLS if c in df.columns] + \
              [c for c in df.columns if c not in META_COLS]
    df = df[ordered]

    if not os.path.exists(out_csv):
        df.to_csv(out_csv, index=False)
        return out_csv

    existing = pd.read_csv(out_csv, nrows=0).columns.tolist()
    if set(df.columns) - set(existing):              # new columns -> rewrite whole file
        merged = pd.concat([pd.read_csv(out_csv), df], ignore_index=True)
        merged.to_csv(out_csv, index=False)
    else:
        df.reindex(columns=existing).to_csv(out_csv, mode="a", header=False, index=False)
    return out_csv


def scalar_summary(rows):
    """Flat {metric_key: value} for wandb summary (scalars only)."""
    out = {}
    for r in rows:
        pre = r.get("metric", "")
        for k, v in r.items():
            if k in META_COLS:
                continue
            if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                out[f"{pre}/f{r.get('fraction')}/{k}"] = float(v)
    return out