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
    """
    Parser an toàn cho .trec:
    - Bắt <DOC> ... </DOC>
    - Lấy <DOCNO>... </DOCNO> ở bất kỳ vị trí (inline cũng được)
    - Gom TẤT CẢ các đoạn <TEXT> ... </TEXT> (inline hoặc multi-line)
    Trả về generator (doc_id, text)
    """
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
        # reset
        doc_id = None
        text_parts = []

    with open(trec_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # Mở / đóng DOC
            if "<DOC>" in line:
                doc_open = True
                doc_id = None
                text_parts = []
                capture_text = False
                continue

            if "</DOC>" in line:
                # kết thúc 1 doc
                yield from push()
                doc_open = False
                capture_text = False
                continue

            if not doc_open:
                continue

            # Lấy DOCNO (inline)
            if "<DOCNO>" in line:
                m = re.search(r"<DOCNO>\s*(.*?)\s*</DOCNO>", line)
                if m:
                    doc_id = m.group(1).strip()
                else:
                    # DOCNO mở ở dòng này, đóng ở dòng sau (ít gặp)
                    content = []
                    if not line.strip().endswith("</DOCNO>"):
                        # thu tiếp cho đến khi gặp </DOCNO>
                        for raw2 in f:
                            l2 = raw2.rstrip("\n")
                            content.append(l2.strip())
                            if "</DOCNO>" in l2:
                                break
                        joined = " ".join(content)
                        m2 = re.search(r"(.*?)\s*</DOCNO>", joined)
                        if m2:
                            doc_id = m2.group(1).strip()

            # Bắt TEXT: xử lý cả inline <TEXT> ... </TEXT>
            if "<TEXT>" in line:
                # inline case
                if "</TEXT>" in line:
                    m = re.search(r"<TEXT>(.*?)</TEXT>", line, flags=re.DOTALL)
                    if m:
                        text_parts.append(m.group(1).strip())
                    capture_text = False
                else:
                    # mở capture, lấy phần sau <TEXT> nếu có
                    after = line.split("<TEXT>", 1)[1]
                    if after.strip():
                        text_parts.append(after.strip())
                    capture_text = True
                continue

            if capture_text:
                if "</TEXT>" in line:
                    # lấy phần trước </TEXT>
                    before = line.split("</TEXT>", 1)[0]
                    if before.strip():
                        text_parts.append(before.strip())
                    capture_text = False
                else:
                    if line.strip():
                        text_parts.append(line.strip())

def parse_trec_dir(trec_dir):
    """
    Parse toàn bộ .trec trong thư mục (không load all vào RAM)
    Trả về DataFrame (doc_id, text)
    """
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
    """
    Đọc labels linh hoạt:
    - Nếu là CSV có header ('doc_id', 'query', 'label' ...) -> giữ nguyên
    - Nếu là TSV không header 4 cột (topic, query_id, doc_id, label) -> thêm header + map query_id -> query text
    - Chuẩn hóa: đảm bảo có cột 'doc_id', 'label', và 'query'
    """
    # thử auto-detect delimiter
    try:
        labels = pd.read_csv(path)
    except Exception:
        labels = pd.read_csv(path, sep=None, engine="python")

    # Case: đọc ra 1 cột duy nhất -> khả năng TSV không header
    if labels.shape[1] == 1:
        # thử đọc lại là TSV không header
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

    # Ép kiểu, strip
    labels["doc_id"] = labels["doc_id"].astype(str).str.strip()
    labels["query"]  = labels["query"].astype(str).str.strip()
    # label -> int nếu có thể
    try:
        labels["label"] = labels["label"].astype(int)
    except Exception:
        pass

    return labels

def build_dataset(trec_dir: str, label_file: str, output_file: str):
    # Parse TREC
    docs_df = parse_trec_dir(trec_dir)
    print(f"[INFO] Parsed total {len(docs_df)} docs from TREC")
    if not docs_df.empty:
        print("[DEBUG] sample TREC doc_ids:", docs_df["doc_id"].head().tolist())

    # Load labels (linh hoạt)
    labels_df = load_labels_flex(label_file)
    print(f"[INFO] Loaded {len(labels_df)} labels from {label_file}")
    print("[DEBUG] sample label doc_ids:", labels_df["doc_id"].head().tolist())

    # Chuẩn hóa khóa (đảm bảo string, strip)
    docs_df["doc_id"] = docs_df["doc_id"].astype(str).str.strip()

    # Join: giữ nguyên thứ tự labels (left), ghép text từ TREC (right)
    merged = labels_df.merge(docs_df, on="doc_id", how="left")

    # Thống kê missing
    missing = merged["text"].isna().sum()
    print(f"[INFO] Matched with text: {len(merged) - missing} / {len(merged)}")
    if missing > 0:
        print(f"[WARN] {missing} rows không tìm thấy text trong TREC.")
        # In vài ví dụ không match để bạn so sánh doc_id
        miss_ids = merged.loc[merged["text"].isna(), "doc_id"].head().tolist()
        print("[DEBUG] ví dụ doc_id bị miss:", miss_ids)

    # Lưu
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

