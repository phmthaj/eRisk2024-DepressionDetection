# 🧠 Depression Detection – eRisk 2024 Task 1

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Overview
This repository implements a **text classification system** for **CLEF eRisk 2024 – Task 1: Search for Symptoms of Depression**.  

The goal of Task 1 is to determine whether a given **text segment** (sentence) is **relevant** to a specific **depression symptom** (query), based on the 21 symptoms from the **BDI-II questionnaire**.

---

## 📂 Project Structure
```
├── dataset/                # Datasets
│   ├── dataset2024.csv  # Final processed dataset (~15k samples)
│   └── majority_erisk_2024.csv  # Original labels
├── model_jupyter/              # model trained in jupyter
├── src/                 # Source code
│   ├── dataset.py       # Dataset class & DataLoader
│   ├── model.py         # LSTM + Attention model
│   ├── train.py         # Training pipeline
│   ├── evaluate.py      # Evaluation metrics
│   └── utils.py         # Preprocessing utilities
├── data/ # Data storage
│ ├── raw/ # Raw TREC + label files
│ ├── parsed/ # Parsed + cleaned datasets
│ └── models/ # Trained embeddings & LSTM models
├── Scripts/ # Pipeline scripts
│ ├── parsed.py # Join TREC + labels → final_dataset.csv
│ ├── text_cleaning.py # Preprocess & tokenize text
│ ├── embedding.py # Train Word2Vec embeddings
│ └── train_lstm_v2.py # Train LSTM model
└── README.md
|__ requirement.txt
```

---

## 📂 Dataset
- **Source**: [eRisk 2024 official data](https://drive.google.com/drive/folders/13SVrWMuZxyqLjPFREkNJO3seB_SYYjFS)  
- **Processing Steps**:
  - Extracted sentences and queries from TREC files  
  - Merged judgments from `majority_erisk_2024.csv`  
  - Final dataset: `dataset2024.csv` with ~15,000 labeled samples  

**Label Meaning**:  
- `1` → Sentence expresses symptom (relevant)  
- `0` → Sentence not related (irrelevant)  

---

## 🧹 Preprocessing
Before training, text samples are normalized with:
- Lowercasing 
- Contraction expansion (e.g., *"can't"* → *"cannot"*)  
- Lemmatization (words reduced to their base form)  
- Tokenization (split into word tokens)  

---

## 🏗️ Model
The classification model is based on **LSTM with Attention**:

- **Embedding Layer** → transforms tokens into dense vectors  
- **LSTM Encoder** → captures contextual information  
- **Attention Mechanism** → focuses on words most relevant to the query  
- **Classifier** → outputs relevance score (`0/1`)  

---

## ⚙️ Installation
```bash
# Clone repository
git clone https://github.com/phmthaj/depression-detection-erisk2024.git
cd depression-detection-erisk2024

# Create environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Training 

You need to put the .Trec file from erisk2024.zip (get by this link https://drive.google.com/drive/folders/13SVrWMuZxyqLjPFREkNJO3seB_SYYjFS) and put it in data/raw file in order for the pipeline to work. 

Then run the pipeline step by step:

```bash
# Step 1: Parse TREC + Labels
python Scripts/parsed.py --trec_dir data/raw/erisk2024 \
  --label_file data/raw/majority_erisk_24_clean.csv \
  --output_file data/parsed/final_dataset.csv

# Step 2: Clean Text
python Scripts/text_cleaning.py --input_file data/parsed/final_dataset.csv \
  --output_file data/parsed/clean_dataset.csv

# Step 3: Train Word2Vec Embeddings
python Scripts/embedding.py --input_file data/parsed/clean_dataset.csv \
  --model_file data/models/word2vec.model \
  --vector_file data/models/embeddings.pkl

# Step 4: Train LSTM Model
python Scripts/train_lstm_v2.py --clean_file data/parsed/clean_dataset.csv \
  --embedding_file data/models/embeddings.pkl \
  --model_output data/models/lstm_v2.pt \
  --epochs 10 --batch_size 64

You can change epochs size. After training, the model will be  in the file data/models.

📊 Pipeline Diagram
flowchart TD
  A[Raw TREC files] --> B[parsed.py]
  B -->|Merge with labels| C[final_dataset.csv]
  C --> D[text_cleaning.py]
  D --> E[clean_dataset.csv]
  E --> F[embedding.py]
  F --> G[embeddings.pkl]
  G --> H[train_lstm_v2.py]
  H --> I[lstm_v2.pt]


## 📊 Results (Example)
| Model            | Accuracy | F1-score |
|------------------|----------|----------|
| LSTM + Attention | 0.88     | 0.87     |

---

## 📚 References
- [CLEF eRisk 2024 Lab Notebook](https://ceur-ws.org/Vol-3740/paper-72.pdf)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  


