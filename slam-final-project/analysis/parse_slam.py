"""Parse SLAM 2018 data files into parquet frames."""
import sys
import pandas as pd

def parse(path, key_path=None):
    labels = {}
    if key_path:
        with open(key_path) as f:
            for line in f:
                tid, lab = line.split()
                labels[tid] = int(lab)
    rows = []
    meta = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                if "user:" in line:
                    meta = {}
                    for part in line[1:].split():
                        if ":" in part:
                            k, v = part.split(":", 1)
                            meta[k] = v
                continue
            cols = line.split()
            tid = cols[0]
            token, pos, morph, dep = cols[1], cols[2], cols[3], cols[4]
            if key_path:
                lab = labels.get(tid)
            else:
                lab = int(cols[-1])
            rows.append((
                tid, meta.get("user"), meta.get("countries", "").split("|")[0],
                float(meta.get("days", 0)),
                meta.get("client"), meta.get("session"), meta.get("format"),
                float(meta["time"]) if meta.get("time", "null") not in ("null", None) else -1.0,
                token.lower(), pos, dep, lab,
            ))
    df = pd.DataFrame(rows, columns=[
        "tid", "user", "country", "days", "client", "session", "format",
        "time", "token", "pos", "dep", "label",
    ])
    return df

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    key = sys.argv[3] if len(sys.argv) > 3 else None
    df = parse(src, key)
    df.to_parquet(out)
    print(out, len(df), "tokens,", df.user.nunique(), "users, err rate",
          round(df.label.mean(), 4))
