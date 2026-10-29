#!/usr/bin/env python3
"""Course-staff check: which reported numbers can be traced to an output in this repository?

The write-up (REPORT.md), the deck generator (tools/make_deck.js) and the figure script
(tools/make_figures.py) report results from three places: the PyMC notebooks in
analysis/ (committed with outputs, and as HTML renders), the MAP scripts in analysis/
(committed without any output or log), and a Colab pipeline that is not in the
repository. This script reads every committed notebook output, in the two .ipynb files
and in the four HTML renders, and

  1. prints the headline numbers the notebooks did produce, with file and cell;
  2. checks each headline claim of the write-up and the deck against those outputs;
  3. searches all committed code for NUTS, LOO/WAIC, an HMM and a conformal library;
  4. prints the code lines behind five problems the staff README describes: the missing
     `morph` column, dev labels in the model-selection grid, dev labels in the sequence
     features, the stacker and threshold fit on one calibration half, and the hand-typed
     figure arrays.

It cannot say whether the Colab numbers are right, only that nothing here produced them.

Standard library only:  python staff/check_provenance.py
"""
import ast
import json
import pathlib
import re
from html.parser import HTMLParser

REPO = pathlib.Path(__file__).resolve().parent.parent
A = REPO / "analysis"


class NotebookHTML(HTMLParser):
    """Collect code and output text per cell from an nbconvert (JupyterLab) HTML render."""

    def __init__(self):
        super().__init__()
        self.cells, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        if tag != "div":
            return
        cls = (dict(attrs).get("class") or "").split()
        if "jp-Cell" in cls:
            self.cells.append({"code": "", "out": ""})
        kind = "out" if "jp-OutputArea" in cls else "code" if "jp-InputArea" in cls else None
        self.stack.append(kind)

    def handle_endtag(self, tag):
        if tag == "div" and self.stack:
            self.stack.pop()

    def handle_data(self, data):
        kind = next((k for k in reversed(self.stack) if k), None)
        if kind and self.cells:
            self.cells[-1][kind] += data


def load_sources():
    """{name: [ {code, out}, ... ]} for every committed notebook and render."""
    src = {}
    for f in sorted(A.glob("*.ipynb")):
        nb = json.loads(f.read_text(encoding="utf-8"))
        cells = []
        for c in nb["cells"]:
            out = ""
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    out += "".join(o["text"])
                elif "data" in o and "text/plain" in o["data"]:
                    out += "".join(o["data"]["text/plain"])
            cells.append({"code": "".join(c["source"]) if c["cell_type"] == "code" else "", "out": out})
        src[f.name] = cells
    for f in sorted(A.glob("*.html")):
        p = NotebookHTML()
        p.feed(f.read_text(encoding="utf-8"))
        src[f.name] = p.cells
    return src


def find(src, pattern, field="out"):
    rx = re.compile(pattern)
    for name, cells in src.items():
        for i, c in enumerate(cells):
            for m in rx.finditer(c[field]):
                yield name, i, m


def auroc_outputs(src):
    vals = []
    for name, i, m in find(src, r"(?i)auroc[\s:=]+(0\.\d{3,6})"):
        vals.append((float(m.group(1)), name, i))
    for name, i, m in find(src, r"(?m)^\s*\d+\s+\S+\s+\d\s+(0\.8\d{5})\s"):  # ranked grid table rows
        vals.append((float(m.group(1)), name, i))
    return vals


def main():
    src = load_sources()
    print("Committed notebook sources:", ", ".join(f"{k} ({len(v)} cells)" for k, v in src.items()))

    print("\n1. Headline numbers in the committed outputs (cell = 0-based cell index):")
    pats = [
        ("training tokens (fr_en)", r"Train tokens: \((\d+)"),
        ("dev tokens (fr_en)", r"Development tokens: \((\d+)"),
        ("training error rate", r"Training error rate: ([\d.]+%)"),
        ("L2 logistic, dev AUROC", r"L2 AUROC: ([\d.]+)"),
        ("hierarchical ADVI, dev AUROC", r"Bayesian AUROC: ([\d.]+)"),
        ("final refit, dev AUROC", r"Final AUROC:\s+([\d.]+)"),
        ("final refit, dev ECE", r"Final ECE:\s+([\d.]+)"),
        ("final refit, dev Brier", r"Final Brier:\s+([\d.]+)"),
        ("final refit, ADVI runtime (min)", r"Final runtime minutes: ([\d.]+)"),
        ("forward CV fold AUROC", r"Fold (\d): AUROC=([\d.]+)"),
        ("forward CV mean AUROC", r"Bayesian CV auroc: ([\d.]+ ± [\d.]+)"),
        ("extension grid, configurations fit", r"Successful: (\d+)"),
        ("extension grid, winner", r"Best configuration: (\S+)"),
    ]
    for label, pat in pats:
        for name, i, m in find(src, pat):
            print(f"   {label:36s} {' '.join(m.groups()):>16s}   {name}, cell {i}")
    grid = [v for v, n, _ in auroc_outputs(src) if n == "slam_bayesian_extension_grid.html"]
    if grid:
        print(f"   {'extension grid, dev AUROC range':36s} {min(grid):.6f} to {max(grid):.6f} over {len(grid)} printed values")

    print("\n2. Headline claims of the write-up and deck against those outputs:")
    texts = {p: (REPO / p).read_text(encoding="utf-8") for p in
             ("REPORT.md" if (REPO / "REPORT.md").exists() else "README.md", "tools/make_deck.js",
              "tools/make_figures.py", "docs/model_ladder_comparison.md", "docs/transfer_gap_deep_dive.md")}
    aurocs = auroc_outputs(src)
    out_text = "\n".join(c["out"] for cells in src.values() for c in cells)
    eces = sorted({float(m.group(1)) for _, _, m in find(src, r"(?i)\bece[\s:=]+(0\.\d+)")})

    def auroc_match(values, note=""):
        hits = [(v, n, i) for v, n, i in aurocs if any(abs(v - x) < 0.0005 for x in values)]
        if not hits:
            return "no matching AUROC in any committed output" + note
        v, n, i = hits[0]
        return f"matches output {v:.6f} in {n}, cell {i}" + note

    claims = [
        ("native AUROC 0.755 / 0.777 / 0.777", ["0.755 / 0.777", '"0.755"', "0.7765"], lambda: auroc_match([0.7554, 0.7765, 0.7769])),
        ("ELPD +2,158 ± 77 vs pooled (PSIS-LOO)", ["2,158"],
         lambda: "no LOO, WAIC or ELPD computation in any committed code or output"),
        ("ECE 0.020 -> 0.014", ["0.020 → 0.014", "ECE 0.014"],
         lambda: f"ECE values in committed outputs: {eces}"),
        ("HMM hybrid NLL +0.00055 [+0.00018, +0.00092]", ["0.00055"],
         lambda: "no HMM code anywhere; value found in outputs: " + str("0.00055" in out_text)),
        ("transfer AUROC 0.732, CI [0.727, 0.737]", ["0.727, 0.737"], lambda: auroc_match([0.732, 0.727, 0.737])),
        ("forward gate, IRT +0.036 AUROC", ["0.036"],
         lambda: "code is analysis/forward_gate.py; it writes no file and no output is committed"),
        ("ladder M6 0.780 / 0.800 / 0.806", ["0.780 / 0.800 / 0.806", "0.8001"],
         lambda: auroc_match([0.78, 0.8001, 0.8061], "; code is analysis/ladder_track.py, no output committed")),
        ("conformal ~50% / ~83% auto-handled", ["~50%", "50.5%"],
         lambda: "code is analysis/ladder_track.py and stack_conformal.py; no output committed"),
        ("epistemic rises 4.6x under transfer", ["4.6×"],
         lambda: "no output; staff/check_demo_uncertainty.py tests it on demo_data.json"),
        ("deck: 'tuned non-Bayesian logistic baseline 0.814'", ["non-Bayesian logistic baseline: AUROC 0.814"],
         lambda: auroc_match([0.814], " (the notebook labels it 'Final hierarchical Bayesian logistic')")),
    ]
    for label, needles, check in claims:
        where = [p for p, t in texts.items() if any(n in t for n in needles)]
        print(f"   {label:48s} stated in: {', '.join(where) or '-'}\n   {'':48s} -> {check()}")
    print("   Every AUROC in the outputs is on fr_en; no committed output evaluates es_en or en_es:",
          not any(re.search(r"es_en|en_es", c["out"]) for cells in src.values() for c in cells))

    print("\n3. Keyword search over all committed code (scripts, notebook code cells, rendered code, deck generators):")
    code = {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8")
            for p in list(A.glob("*.py")) + list((REPO / "tools").glob("*"))}
    for name, cells in src.items():
        code[name] = "\n".join(c["code"] for c in cells)
    searches = [
        ("NUTS or MCMC sampling", r"pm\.sample\(|NUTS\(|nuts_sampler|numpyro\.infer\.MCMC"),
        ("the word NUTS anywhere in code", r"NUTS"),
        ("variational fit (pm.fit)", r"pm\.fit\("),
        ("LOO / WAIC / ELPD", r"az\.loo|az\.waic|az\.compare|elpd|psis"),
        ("an HMM", r"hmmlearn|GaussianHMM|CategoricalHMM|DiscreteMarkovChain|viterbi|forward_backward"),
        ("a conformal library", r"mapie|crepes|puncc|torchcp"),
    ]
    for label, pat in searches:
        hits = sorted({n for n, t in code.items() if re.search(pat, t, flags=re.I if label != "the word NUTS anywhere in code" else 0)})
        hits = [n + " (slide text)" if n.startswith("tools/make_") and n.endswith(".js") else n for n in hits]
        print(f"   {label:32s} {', '.join(hits) if hits else 'none'}")
    for n, t in code.items():
        for line in t.splitlines():
            if "NUTS" in line:
                print(f"     {n}: {line.strip()}")

    print("\n4. Code lines behind the staff README's points:")
    tree = ast.parse((A / "parse_slam.py").read_text(encoding="utf-8"))
    cols = next(
        [e.value for e in kw.value.elts]
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        for kw in node.keywords if kw.arg == "columns" and isinstance(kw.value, ast.List)
    )
    print(f"   parse_slam.py writes columns {cols}; 'morph' among them: {'morph' in cols}")
    for f in sorted(A.glob("*.py")):
        for k, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"""\[["']morph["']\]""", line):
                print(f"     reads ['morph']: analysis/{f.name}:{k}")
                break

    def show(path, pattern):
        for k, line in enumerate((REPO / path).read_text(encoding="utf-8").splitlines(), 1):
            if re.search(pattern, line):
                print(f"     {path}:{k}: {line.strip()}")

    print("   the extension grid scores its 64 configurations on the official dev labels:")
    for name, i, m in find(src, r".*(DEV_LABEL_PATH = |\"auroc\": roc_auc_score\(y_dev).*", field="code"):
        if name == "slam_bayesian_extension_grid.html":
            print(f"     cell {i}: {m.group(0).strip()}")
    print("   and the final notebook describes its configuration as:")
    for name, i, m in find(src, r".*(selected by the 64-model screening grid|training-only screening).*", field="code"):
        if name == "slam_bayesian_final_refit_commented.ipynb":
            print(f"     {name}, cell {i}: {m.group(0).strip()}")
    nb = json.loads((A / "slam_bayesian_final_refit_commented.ipynb").read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            for line in "".join(c["source"]).splitlines():
                if re.search(r"No hyperparameter is changed|came from training-only", line):
                    print(f"     slam_bayesian_final_refit_commented.ipynb, markdown cell {i}: {line.strip()}")
    print("   sequence features: dev rows carry their labels from the .key file, train and dev are")
    print("   concatenated, and the state is read, then updated with each token's own label, token by token:")
    show("analysis/parse_slam.py", r"lab = labels.get")
    show("analysis/ladder_track.py", r"pd.concat\(\[tr2, dv2\]|out\[k, 0\] =|y = labs\[k\]")
    print("   stacker and threshold on the same calibration half:")
    show("analysis/ladder_track.py", r"\.fit\(Z\[cal\]|pc, yc = p_stack\[cal\]|\(yc\[acc\]\.sum\(\) \+ 1\)")
    print("   figure values typed into tools/make_figures.py:")
    ftree = ast.parse((REPO / "tools/make_figures.py").read_text(encoding="utf-8"))
    for node in ast.walk(ftree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("M", "vals"):
            nums = [n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
            print(f"     line {node.lineno}: {node.targets[0].id} = {len(nums)} literal numbers, e.g. {nums[:4]}")


if __name__ == "__main__":
    main()
