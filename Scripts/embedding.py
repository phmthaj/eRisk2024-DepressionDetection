import argparse
import pandas as pd
from gensim.models import Word2Vec
import pickle
import os

def train_word2vec(input_file, model_file, vector_file, size=200, window=5, min_count=2, workers=4):
    print(f"[INFO] Loading dataset {input_file}")
    df = pd.read_csv(input_file)

    print("[INFO] Preparing corpus...")
    corpus = df["clean_text"].apply(eval).tolist() 

    print("[INFO] Training Word2Vec...")
    w2v = Word2Vec(
        sentences=corpus,
        vector_size=size,
        window=window,
        min_count=min_count,
        workers=workers
    )

    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    w2v.save(model_file)
    print(f"[INFO] Saved Word2Vec model to {model_file}")

    word2idx = {word: idx for idx, word in enumerate(w2v.wv.index_to_key)}
    embedding_matrix = w2v.wv.vectors

    with open(vector_file, "wb") as f:
        pickle.dump({"word2idx": word2idx, "embeddings": embedding_matrix}, f)

    print(f"[INFO] Saved embeddings + word2idx to {vector_file}")
    print(f"[DEBUG] vocab size = {len(word2idx)}, embedding dim = {embedding_matrix.shape[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Word2Vec embeddings")
    parser.add_argument("--input_file", type=str, required=True, help="Clean dataset CSV")
    parser.add_argument("--model_file", type=str, default="data/word2vec/word2vec.model", help="Output Word2Vec model")
    parser.add_argument("--vector_file", type=str, default="data/word2vec/embeddings.pkl", help="Pickle file with word2idx + embeddings")
    parser.add_argument("--size", type=int, default=200, help="Embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--min_count", type=int, default=2, help="Min word frequency")
    parser.add_argument("--workers", type=int, default=4, help="Number of threads")
    args = parser.parse_args()

    train_word2vec(args.input_file, args.model_file, args.vector_file, args.size, args.window, args.min_count, args.workers)

