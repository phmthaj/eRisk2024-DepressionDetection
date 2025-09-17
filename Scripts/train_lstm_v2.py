import argparse, pickle, os, ast
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


queries_tokens = [
    ["sadness"],                             # 1
    ["pessimistic"],                         # 2
    ["past", "failure"],                     # 3
    ["loss", "of", "pleasure"],              # 4
    ["guilty", "feeling"],                   # 5
    ["punishment", "feeling"],               # 6
    ["self", "dislike"],                     # 7
    ["self", "critical"],                    # 8
    ["suicidal", "thought", "or", "wish"],   # 9
    ["cry"],                                 # 10
    ["agitated"],                            # 11
    ["loss", "of", "interest"],              # 12
    ["indecisiveness"],                      # 13
    ["worthless"],                           # 14
    ["loss", "of", "energy"],                # 15
    ["change", "in", "sleeping", "pattern"], # 16
    ["irritability"],                        # 17
    ["change", "in", "appetite"],            # 18
    ["concentration", "difficulty"],         # 19
    ["tiredness", "or", "fatigue"],          # 20
    ["loss", "of", "interest", "in", "sex"]  # 21
]

PAD_IDX, UNK_IDX = 0, 1

def build_queries(word2idx):
    queries_sequence = []
    for s in queries_tokens:
        ids = [word2idx.get(t, UNK_IDX) for t in s]
        queries_sequence.append(torch.tensor(ids, dtype=torch.long))
    return pad_sequence(queries_sequence, batch_first=True, padding_value=PAD_IDX)


class TextDataset(Dataset):
    def __init__(self, df, word2idx, queries_padded):
        self.docs = df["clean_text"].apply(ast.literal_eval).tolist()
        self.q_idx = df["query"].tolist()   # 1..21
        self.labels = df["label"].astype(int).tolist()
        self.word2idx = word2idx
        self.pad_idx = PAD_IDX
        self.unk_idx = UNK_IDX
        self.queries_padded = queries_padded

    def encode_doc(self, tokens):
        ids = [self.word2idx.get(tok, self.unk_idx) for tok in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        doc_ids = self.encode_doc(self.docs[idx])
        q_id = int(self.q_idx[idx]) - 1      # chuyển 1..21 -> 0..20
        q_tensor = self.queries_padded[q_id]
        label = self.labels[idx]
        return doc_ids, q_tensor, label


def collate_fn(batch):
    docs, queries, labels = zip(*batch)

    docs_padded = pad_sequence(docs, batch_first=True, padding_value=PAD_IDX)
    queries_padded = pad_sequence(queries, batch_first=True, padding_value=PAD_IDX)

    doc_lengths = torch.tensor([len(x) for x in docs], dtype=torch.long)
    query_lengths = torch.tensor([len(x) for x in queries], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return docs_padded, doc_lengths, queries_padded, query_lengths, labels


class LSTM_v2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, hidden_size=64, pad_idx=0, pdrop=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.doc_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.query_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(pdrop)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, docs, docs_length, queries, queries_length):
        d_emb = self.dropout(self.embedding(docs))
        q_emb = self.dropout(self.embedding(queries))

        d_packed = pack_padded_sequence(d_emb, docs_length.cpu(), batch_first=True, enforce_sorted=False)
        q_packed = pack_padded_sequence(q_emb, queries_length.cpu(), batch_first=True, enforce_sorted=False)

        d_output, _ = self.doc_lstm(d_packed)
        d_output, _ = pad_packed_sequence(d_output, batch_first=True)

        _, (q_h, _) = self.query_lstm(q_packed)
        q_h = q_h[-1]

        q_vec = self.attention(q_h).unsqueeze(2)
        scores = torch.bmm(d_output, q_vec).squeeze(2)
        alpha = F.softmax(scores, dim=1)
        new_alpha = alpha.unsqueeze(1)
        represent_doc = torch.bmm(new_alpha, d_output).squeeze(1)

        comb = torch.cat([represent_doc, q_h], dim=1)
        comb = self.dropout(comb)
        logits = self.fc(comb)
        return logits


def main(args):
    df = pd.read_csv(args.clean_file)

    import ast
    df = df[df["clean_text"].apply(lambda x: len(ast.literal_eval(x)) > 0)].reset_index(drop=True)

    with open(args.embedding_file, "rb") as f:
        obj = pickle.load(f)
        word2idx, embeddings = obj["word2idx"], obj["embeddings"]

    queries_padded = build_queries(word2idx)
    dataset = TextDataset(df, word2idx, queries_padded)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTM_v2(len(word2idx), embeddings.shape[1], args.hidden_size,
                    pad_idx=word2idx.get("<pad>", 0), pdrop=args.dropout).to(device)
    model.embedding.weight.data.copy_(torch.tensor(embeddings))
    model.embedding.weight.requires_grad = True

    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    print("[INFO] Start training...")
    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, total_loss = 0, 0, 0
        for d, d_len, q, q_len, y in loader:
            d, d_len, q, q_len, y = d.to(device), d_len.to(device), q.to(device), q_len.to(device), y.to(device)

            out = model(d, d_len, q, q_len)
            loss = crit(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

        acc = correct / total
        print(f"Epoch {epoch}/{args.epochs} | Loss {total_loss/len(loader):.4f} | Acc {acc:.4f}")

    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print(f"[INFO] Saved model to {args.model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM_v2 with attention")
    parser.add_argument("--clean_file", type=str, required=True)
    parser.add_argument("--embedding_file", type=str, required=True)
    parser.add_argument("--model_output", type=str, default="data/models/lstm_v2.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    args = parser.parse_args()
    main(args)





