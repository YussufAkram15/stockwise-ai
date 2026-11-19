"""Applies the agreed team fixes in ONE step.

Run from the project root (the folder that contains src/):
    python apply_team_fixes.py

SAFE: it first checks that every place it wants to change exists. If even one is missing,
it changes NOTHING and tells you which one. Running it twice is harmless.
Use git to see what changed:   git diff

What it fixes
  src/config.py               VAL_END (it swallowed the test slice), adds TEST_START,
                              adds the new columns qc_sales_suspect and sales_capped
  src/train_models.py         MAE / RMSE / WAPE next to RMSLE, best model chosen by WAPE,
                              MLflow in the shared store + experiment "stockwise-ai",
                              suspected data errors left out of TRAINING
  src/feature_engineering.py  lags / rolling means / series mean built from sales_capped
  .gitignore, requirements.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path.cwd()
problems: list[str] = []
new_texts: dict[Path, str] = {}
notes: list[str] = []


def read(rel: str) -> str | None:
    p = ROOT / rel
    if not p.exists():
        problems.append(f"{rel}: file not found (run this from the project root)")
        return None
    return p.read_text(encoding="utf-8")


def replace_once(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        problems.append(f"{where}: expected 1 place, found {n}  ->  {old.strip()[:70]!r}")
        return text
    return text.replace(old, new)


def regex_once(text: str, pattern: str, repl: str, where: str) -> str:
    new, n = re.subn(pattern, repl, text, flags=re.M)
    if n != 1:
        problems.append(f"{where}: expected 1 place, found {n}  ->  {pattern[:70]!r}")
        return text
    return new


# ----------------------------------------------------------------------------- config.py
def patch_config() -> None:
    rel = "src/config.py"
    text = read(rel)
    if text is None:
        return
    for name, want in [("TRAIN_END", "2017-06-30"), ("VAL_START", "2017-07-01"), ("TEST_END", "2017-08-15")]:
        m = re.search(rf'^{name}\s*=\s*(.+)$', text, flags=re.M)
        got = m.group(1).strip() if m else "MISSING"
        if want not in got:
            notes.append(f"CHECK config.py: {name} is {got}, the agreed value is {want}")
    if re.search(r'^VAL_END\s*=', text, flags=re.M):
        text = regex_once(text, r'^VAL_END\s*=.*$',
                          'VAL_END = "2017-07-30"   # validation ends 30 Jul; 31 Jul - 15 Aug is the TEST slice',
                          rel + " VAL_END")
    else:
        text += '\nVAL_END = "2017-07-30"   # validation ends 30 Jul; 31 Jul - 15 Aug is the TEST slice\n'
    if not re.search(r'^TEST_START\s*=', text, flags=re.M):
        text += 'TEST_START = "2017-07-31"\n'
    if "# >>> team-fixes" not in text:
        text += '''
# >>> team-fixes: suspected data errors (details in reports/cleaning_report.md)
# A sale is a SUSPECTED DATA ERROR when it is above Q3 + OUTLIER_SUSPECT_MULT x IQR of its own series
# AND no holiday, event, earthquake or promotion explains it. Real peaks are never flagged.
OUTLIER_SUSPECT_MULT = 10.0
# False = only suspected data errors are capped in sales_capped (real peaks untouched).
# True  = every outlier is capped there (experiment: stops real peaks echoing into the lags).
CAP_ALL_OUTLIERS_IN_HISTORY = False
if "qc_sales_suspect" not in QC_FLAGS:
    QC_FLAGS.append("qc_sales_suspect")
if "sales_capped" not in TRAIN_CLEAN_COLUMNS:
    TRAIN_CLEAN_COLUMNS.insert(TRAIN_CLEAN_COLUMNS.index("sales") + 1, "sales_capped")
if "qc_sales_suspect" not in TRAIN_CLEAN_COLUMNS:
    TRAIN_CLEAN_COLUMNS.append("qc_sales_suspect")
TEST_CLEAN_COLUMNS = [c for c in TRAIN_CLEAN_COLUMNS if c not in ("sales", "sales_capped") and c not in QC_FLAGS]
COLUMN_DOC["sales_capped"] = ("Same as sales, except suspected data errors are capped at the normal outlier limit. "
                              "Use it to build lags / rolling means. `sales` itself is never changed.")
COLUMN_DOC["qc_sales_suspect"] = ("QC FLAG: extremely high sale (> Q3 + 10 x IQR of its series) that no holiday, event, "
                                  "earthquake or promotion explains. Left out of TRAINING. Never a model feature.")
# <<< team-fixes
'''
    new_texts[ROOT / rel] = text


# ----------------------------------------------------------------------------- train_models.py
def patch_train_models() -> None:
    rel = "src/train_models.py"
    text = read(rel)
    if text is None:
        return
    if "def wape(" in text:
        notes.append("train_models.py: already patched, skipped")
        return
    text = replace_once(text, 'TARGET = "sales"\n', '''TARGET = "sales"
# One MLflow store and one experiment name for the whole team (same as the Polynomial run)
MLFLOW_URI = f"sqlite:///{(config.PROJECT_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "stockwise-ai"
''', rel + " TARGET")
    text = replace_once(text, 'def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:', '''def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """100 * sum|actual - predicted| / sum(actual)  (in %). The MAIN team metric."""
    total = float(np.sum(actual))
    return float("nan") if total <= 0 else float(100.0 * np.sum(np.abs(actual - predicted)) / total)


def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """The team metrics (MAE, RMSE, WAPE) plus RMSLE, all on predictions clipped to 0."""
    p = np.clip(predicted, 0, None)
    return {"val_mae": mae(actual, p), "val_rmse": rmse(actual, p),
            "val_wape": wape(actual, p), "val_rmsle": rmsle(actual, p)}


def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:''', rel + " rmsle")
    text = replace_once(text, '    qc_mask = (df["qc_zero_series"].fillna(0) == 0) & (df["qc_before_first_sale"].fillna(0) == 0)\n',
                        '''    qc_mask = (df["qc_zero_series"].fillna(0) == 0) & (df["qc_before_first_sale"].fillna(0) == 0)
    if "qc_sales_suspect" in df.columns:
        # suspected DATA ERRORS (extreme AND unexplained) are left out of training; real peaks stay in
        n_sus = int(((df["date"] <= pd.Timestamp(config.TRAIN_END)) & (df["qc_sales_suspect"].fillna(0) == 1)).sum())
        print(f"[TRAIN] Suspected data-error rows removed from training: {n_sus:,}")
        qc_mask = qc_mask & (df["qc_sales_suspect"].fillna(0) == 0)
''', rel + " qc_mask")
    text = replace_once(text, '        mlflow.set_experiment("StockWise-AI")\n',
                        '        mlflow.set_tracking_uri(MLFLOW_URI)\n        mlflow.set_experiment(MLFLOW_EXPERIMENT)\n',
                        rel + " set_experiment")
    text = replace_once(text, '    score = rmsle(y_val, preds)\n', '    score = rmsle(y_val, preds)\n    scores = all_metrics(y_val, preds)\n', rel + " score")
    text = replace_once(text, '    print(f"  RMSLE : {score:.5f}")\n',
                        '    print(f"  RMSLE : {score:.5f}")\n'
                        '    print(f"  MAE = {scores[\'val_mae\']:.3f}   RMSE = {scores[\'val_rmse\']:.3f}   WAPE = {scores[\'val_wape\']:.2f}%")\n',
                        rel + " print RMSLE")
    text = replace_once(text, '            metrics={"val_rmsle": score, "train_time_s": elapsed},\n',
                        '            metrics={**scores, "train_time_s": elapsed},\n', rel + " mlflow metrics")
    text = replace_once(text, '        "rmsle": score,\n', '        "rmsle": score,\n        "scores": scores,\n', rel + " result dict")
    text = replace_once(text, '    best = min(results, key=lambda r: r["rmsle"])\n',
                        '    best = min(results, key=lambda r: r["scores"]["val_wape"])      # WAPE is the main metric\n', rel + " best (artifacts)")
    text = replace_once(text, '                "rmsle": float(r["rmsle"]),\n',
                        '                "rmsle": float(r["rmsle"]),\n                "scores": {k: float(v) for k, v in r["scores"].items()},\n',
                        rel + " metadata")
    text = replace_once(text, '    best = min(results, key=lambda x: x["rmsle"])\n',
                        '    best = min(results, key=lambda x: x["scores"]["val_wape"])      # WAPE is the main metric\n', rel + " best (main)")
    text = replace_once(text, '''    print(f"Best model : {best['name']}  (RMSLE = {best['rmsle']:.5f})")\n''',
                        '''    print(f"Best model : {best['name']}  (WAPE = {best['scores']['val_wape']:.2f}%,  RMSLE = {best['rmsle']:.5f})")\n''',
                        rel + " best print")
    new_texts[ROOT / rel] = text


# ----------------------------------------------------------------------------- feature_engineering.py
def patch_features() -> None:
    rel = "src/feature_engineering.py"
    text = read(rel)
    if text is None:
        return
    if "sales_capped" in text:
        notes.append("feature_engineering.py: already patched, skipped")
        return
    text = replace_once(text, '_fast_lag(df, "sales", lag)', '_fast_lag(df, "sales_capped", lag)', rel + " lags")
    text = regex_once(text, r'(_fast_roll\(\s*df,\s*)"sales"(,\s*w,\s*"mean")', r'\1"sales_capped"\2', rel + " rolling")
    text = replace_once(text, 'df.groupby(KEY, sort=False)["sales"]', 'df.groupby(KEY, sort=False)["sales_capped"]', rel + " series mean")
    text = replace_once(text, 'for c in ["sales"] + config.QC_FLAGS:', 'for c in ["sales", "sales_capped"] + config.QC_FLAGS:', rel + " placeholders")
    text = replace_once(text, '    combined["date"] = pd.to_datetime(combined["date"])\n', '''    combined["date"] = pd.to_datetime(combined["date"])
    # History features are built from sales_capped (= sales, except suspected data errors are capped),
    # so one bad day is not copied forward into the lags / rolling means. Old cleaned file -> plain sales.
    if "sales_capped" not in combined.columns:
        combined["sales_capped"] = np.nan
    combined["sales_capped"] = combined["sales_capped"].fillna(combined["sales"])
''', rel + " fillna")
    new_texts[ROOT / rel] = text


# ----------------------------------------------------------------------------- small files
def patch_gitignore() -> None:
    p = ROOT / ".gitignore"
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    have = {line.strip() for line in text.splitlines()}
    add = [l for l in ["mlflow.db", "mlruns/", ".venv/", "__pycache__/", "data/raw/", "data/processed/"] if l not in have]
    if add:
        new_texts[p] = text.rstrip("\n") + ("\n" if text else "") + "\n# added by apply_team_fixes.py\n" + "\n".join(add) + "\n"


def patch_requirements() -> None:
    p = ROOT / "requirements.txt"
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    have = {re.split(r"[<>=!~\[ ]", l.strip().lower())[0] for l in text.splitlines() if l.strip() and not l.startswith("#")}
    add = [pkg for pkg in ["pandas", "numpy", "pyarrow", "matplotlib", "scikit-learn", "mlflow", "xgboost", "joblib"] if pkg not in have]
    if add:
        new_texts[p] = text.rstrip("\n") + ("\n" if text else "") + "\n".join(add) + "\n"


def main() -> int:
    if not (ROOT / "src").is_dir():
        print("Run this from the project root (the folder that contains src/).")
        return 1
    patch_config()
    patch_train_models()
    patch_features()
    patch_gitignore()
    patch_requirements()

    if problems:
        print("NOTHING WAS CHANGED, because these places were not found:\n")
        for p in problems:
            print("  -", p)
        print("\nThe file probably changed on main. Send me this list and the file, and I will adjust the patch.")
        return 1
    for path, text in new_texts.items():
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue                                  # nothing to change in this file
        path.write_text(text, encoding="utf-8")
        print("updated:", path.relative_to(ROOT))
    for n in notes:
        print("note   :", n)
    print("\nDone. Look at the changes with:  git diff")
    return 0


if __name__ == "__main__":
    sys.exit(main())