# generate_candidate_unified_mapping.py
import csv
import json
from pathlib import Path
import re
import pandas as pd

ROOT = Path.cwd().resolve()
DATASET_DIR = ROOT / "dataset"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OUT = LOGS_DIR / "unified_label_mapping.candidate.csv"

# label heuristics: map various textual signals to our 5 classes
HEURISTICS = [
    (re.compile(r"\b(myocard|infarct|mi|ischemi|ischemia)\b", re.I), "MI"),
    (re.compile(r"\b(atrial fibrill|afib|a\.fibrill|fibrillation)\b", re.I), "AF"),
    (re.compile(r"\b(bundle branch block|bbb|left bundle|right bundle|bundle-branch)\b", re.I), "BBB"),
    (re.compile(r"\b(normal|norm|sinus rhythm|no abnormalit)\b", re.I), "NORM"),
]

def apply_heuristics(text: str):
    if not text:
        return ""
    for pat, lab in HEURISTICS:
        if pat.search(text):
            return lab
    return ""

def extract_ptbxl(ptbxl_dir: Path, rows):
    csv_path = ptbxl_dir / "ptbxl_database.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, r in df.iterrows():
            # filename_lr often like "records100/00000/00001_lr"
            filename_lr = str(r.get("filename_lr", "")).strip()
            if filename_lr:
                rid = Path(filename_lr).name  # e.g. "00001_lr" but we'll preserve folder-less id
                rid = filename_lr.replace("\\", "/").strip("/")
            else:
                rid = str(r.get("ecg_id", "")).strip()
            report = str(r.get("report", "")).strip()
            scp = str(r.get("scp_codes", "")).strip()
            combined = " ".join([report, scp])
            mapped = apply_heuristics(combined)
            rows.append(("PTBXL", rid, combined, mapped))
    else:
        print("PTBXL metadata not found at", csv_path)

def extract_cinc(cinc_dir: Path, rows):
    # CinC 2017 uses REFERENCE files with labels like A,N,O
    # We'll collect training/validation/test reference files if present
    for sub in ["training", "validation", "test", ""]:
        base = cinc_dir if sub == "" else cinc_dir / sub
        if not base.exists():
            continue
        for ref in base.glob("REFERENCE*.csv"):
            try:
                df = pd.read_csv(ref, header=None, names=["record", "label"], dtype=str)
            except Exception:
                continue
            for _, r in df.iterrows():
                record = str(r["record"]).strip()
                label = str(r["label"]).strip().upper()
                # map A->AF? O->OTHER? N->NORM? This is heuristic: you may need to adjust
                mapped = ""
                if label == "A": mapped = "AF"
                elif label == "N": mapped = "NORM"
                elif label == "O": mapped = "OTHER"
                rows.append(("CinC_2017_AFDB", record, label, mapped))
    # If no REFERENCE files, also sweep .hea/.mat basenames
    for rec in cinc_dir.rglob("*"):
        if rec.suffix.lower() in (".hea", ".mat"):
            rel = rec.relative_to(DATASET_DIR).with_suffix("")
            rid = rel.as_posix()
            rows.append(("CinC_2017_AFDB", rid, "", ""))

def extract_folder_generic(ds_name: str, ds_dir: Path, rows):
    # walk files and try to collect any textual hints from accompanying .hea or .txt files
    for p in ds_dir.rglob("*"):
        if p.suffix.lower() in (".hea", ".mat", ".txt", ".csv"):
            rel = p.relative_to(DATASET_DIR).with_suffix("")
            rid = rel.as_posix()
            hint = ""
            # attempt to read small header or first lines for clues
            if p.suffix.lower() == ".hea":
                try:
                    text = p.read_text(errors="ignore")
                    hint = " ".join(text.splitlines()[:10])
                except Exception:
                    hint = ""
            elif p.suffix.lower() == ".csv":
                hint = ""
            rows.append((ds_name, rid, hint, ""))

def main():
    rows = []
    # PTBXL
    ptbxl_dir = DATASET_DIR / "ptb-xl"
    if ptbxl_dir.exists():
        extract_ptbxl(ptbxl_dir, rows)
    # CinC
    cinc_dir = DATASET_DIR / "CinC2017"
    if cinc_dir.exists():
        extract_cinc(cinc_dir, rows)
    # PTB Diagnostic folder (if present, we'll flatten names)
    ptbdiag = DATASET_DIR / "PTB_Diagnostic"
    if ptbdiag.exists():
        extract_folder_generic("PTB_Diagnostic", ptbdiag, rows)
    # Chapman_Shaoxing
    chap = DATASET_DIR / "Chapman_Shaoxing"
    if chap.exists():
        extract_folder_generic("Chapman_Shaoxing", chap, rows)

    # dedupe by dataset+record_id
    seen = set()
    unique = []
    for ds, rid, orig, mapped in rows:
        key = (ds, rid)
        if key in seen:
            continue
        seen.add(key)
        unique.append((ds, rid, orig.replace("\n", " ")[:400], mapped))

    # write CSV
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "record_id", "original_label_text", "mapped_label"])
        for ds, rid, orig, mapped in unique:
            w.writerow([ds, rid, orig, mapped])
    print("Wrote candidate mapping to", OUT)
    print("Rows:", len(unique))

if __name__ == "__main__":
    main()
