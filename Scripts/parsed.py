import argparse
import os
import re
import pandas as pd

QUERIES_21 = [
    "sadness", "pessimistic", "past failure", "loss of pleasure", "guilty feeling",
    "punishment feeling", "self dislike", "self critical", "suicidal thought or wish",
    "cry", "agitated", "loss of interest", "indecisiveness", "worthless",
    "loss of energy", "change in sleeping pattern", "irritability",
    "change in appetite", "concentration difficulty", "tiredness or fatigue",
    "loss of interest in sex",
]

def parse_trec_file(trec_path):

    doc_open = False
    doc_id = None
    text_parts = []
    capture_text = False

    def push():
        nonlocal doc_id, text_parts
        if doc_id and text_parts:
            text = " ".join(t.strip() for t in text_parts if t.strip())
            if text:
                yield (doc_id.strip(), text)
        doc_id = None
        text_parts = []

    with open(trec_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if "<DOC>" in line:
                doc_open = True
                doc_id = None
                text_parts = []
                capture_text = False
                continue

            if "</DOC>" in line:
                yield from push()
                doc_open = False
                capture_text = False
                continue

            if not doc_open:
                continue

            if "<DOCNO>" in line:
                m = re.search(r"<DOCNO>\s*(.*?)\s*</DOCNO>", line)
                if m:
                    doc_id = m.group(1).strip()
                else:
                    content = []
                    if not line.strip().endswith("</DOCNO>"):
                        for raw2 in f:
                            l2 = raw2.rstrip("\n")
                            content.append(l2.strip())
                            if "</DOCNO>" in l2:
                                break
                        joined = " ".join(content)
                        m2 = re.search(r"(.*?)\s*</DOCNO>", joined)
                        if m2:
                            doc_id = m2.group(1).strip()

            if "<TEXT>" in line:
                if "</TEXT>" in line:
                    m = re.search(r"<TEXT>(.*?)</TEXT>", line, flags=re.DOTALL)
                    if m:
                        text_parts.append(m.group(1).strip())
                    capture_text = False
                else:
                    after = line.split("<TEXT>", 1)[1]
                    if after.strip():
                        text_parts.append(after.strip())
                    capture_text = True
                continue

            if capture_text:
                if "</TEXT>" in line:
                    before = line.split("</TEXT>", 1)[0]
                    if before.strip():
                        text_parts.append(before.strip())
                    capture_text = False
                else:
                    if line.strip():
                        text_parts.append(line.strip())

def parse_trec_dir(trec_dir):
   
    all_rows = []
    for fname in os.listdir(trec_dir):
        if fname.lower().endswith(".trec"):
            fpath = os.path.join(trec_dir, fname)
            print(f"[INFO] Parsing {fpath}")
            for doc_id, text in parse_trec_file(fpath):
                all_rows.append((doc_id, text))
    df = pd.DataFrame(all_rows, columns=["doc_id", "text"])
    return df

def load_labels_flex(path):
   
    try:
        labels = pd.read_csv(path)
    except Exception:
        labels = pd.read_csv(path, sep=None, engine="python")

    if labels.shape[1] == 1:
        labels = pd.read_csv(path, sep="\t", header=None,
                             names=["topic", "query_id", "doc_id", "label"])

    cols = {c.lower(): c for c in labels.columns}
    if "doc_id" not in cols:
        if "document_id" in cols:
            labels.rename(columns={cols["document_id"]: "doc_id"}, inplace=True)
        elif "docno" in cols:
            labels.rename(columns={cols["docno"]: "doc_id"}, inplace=True)
        elif "id" in cols:
            labels.rename(columns={cols["id"]: "doc_id"}, inplace=True)

    if "label" not in cols and "relevant" in cols:
        labels.rename(columns={cols["relevant"]: "label"}, inplace=True)

    current_cols = {c.lower(): c for c in labels.columns}
    if "query" not in current_cols:
        if "query_id" in current_cols:
            qcol = current_cols["query_id"]
            labels["query"] = labels[qcol].astype(int).map(lambda i: QUERIES_21[i] if 0 <= i < 21 else f"q_{i}")
        else:
            labels["query"] = "unknown"

    if "doc_id" not in labels.columns:
        raise ValueError("Không tìm thấy cột doc_id (hoặc document_id/docno/id) trong file nhãn.")
    if "label" not in labels.columns:
        raise ValueError("Không tìm thấy cột label (hoặc relevant) trong file nhãn.")

    labels["doc_id"] = labels["doc_id"].astype(str).str.strip()
    labels["query"]  = labels["query"].astype(str).str.strip()
    try:
        labels["label"] = labels["label"].astype(int)
    except Exception:
        pass

    return labels

def build_dataset(trec_dir: str, label_file: str, output_file: str):
    docs_df = parse_trec_dir(trec_dir)
    print(f"[INFO] Parsed total {len(docs_df)} docs from TREC")
    if not docs_df.empty:
        print("[DEBUG] sample TREC doc_ids:", docs_df["doc_id"].head().tolist())

    labels_df = load_labels_flex(label_file)
    print(f"[INFO] Loaded {len(labels_df)} labels from {label_file}")
    print("[DEBUG] sample label doc_ids:", labels_df["doc_id"].head().tolist())

    docs_df["doc_id"] = docs_df["doc_id"].astype(str).str.strip()

    merged = labels_df.merge(docs_df, on="doc_id", how="left")

    missing = merged["text"].isna().sum()
    print(f"[INFO] Matched with text: {len(merged) - missing} / {len(merged)}")
    if missing > 0:
        print(f"[WARN] {missing} rows không tìm thấy text trong TREC.")
        miss_ids = merged.loc[merged["text"].isna(), "doc_id"].head().tolist()
        print("[DEBUG] ví dụ doc_id bị miss:", miss_ids)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[INFO] Saved dataset to {output_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build dataset từ nhiều TREC + labels (robust)")
    ap.add_argument("--trec_dir", required=True, help="Thư mục chứa các file .trec")
    ap.add_argument("--label_file", required=True, help="File nhãn (CSV/TSV). Hỗ trợ cả TSV không header.")
    ap.add_argument("--output_file", default="data/parsed/final_dataset.csv", help="CSV output")
    args = ap.parse_args()

    build_dataset(args.trec_dir, args.label_file, args.output_file)

