import argparse
import pandas as pd
import re
import spacy
import contractions

nlp = spacy.load("en_core_web_sm")

def contract(text: str) -> str:
    return contractions.fix(str(text))

def remove_punct(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r'(\d+)\.(\d+)', r'\1<dot>\2', s)
    s = re.sub(r'(\d+)-(\d+)', r'\1 to \2', s)
    s = re.sub(r'[^a-z0-9\s<dot>]', '', s)
    s = s.replace("<dot>", ".")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def lemmatize_spacy(text: str) -> str:
    doc = nlp(str(text))
    return " ".join([token.lemma_ for token in doc])

def tokenize(corpus: str):
    corpus = str(corpus)
    return corpus.split()

def handling_text(corpus: str):
    corpus = contract(corpus)
    corpus = remove_punct(corpus)
    corpus = lemmatize_spacy(corpus)
    corpus = tokenize(corpus)
    return corpus

def clean_dataset(input_file: str, output_file: str):
    print(f"[INFO] Loading dataset {input_file}")
    df = pd.read_csv(input_file)

    print(f"[INFO] Cleaning text column...")
    df["clean_text"] = df["text"].astype(str).apply(handling_text)

    print(f"[INFO] Saving cleaned dataset to {output_file}")
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean text in dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Input dataset CSV (parsed)")
    parser.add_argument("--output_file", type=str, required=True, help="Output cleaned CSV")
    args = parser.parse_args()

    clean_dataset(args.input_file, args.output_file)
