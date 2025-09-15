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
├── data/                # Datasets
│   ├── dataset2024.csv  # Final processed dataset (~15k samples)
│   └── majority_erisk_2024.csv  # Original labels
├── models/              # Saved checkpoints
├── notebooks/           # Experiments & analysis
├── src/                 # Source code
│   ├── dataset.py       # Dataset class & DataLoader
│   ├── model.py         # LSTM + Attention model
│   ├── train.py         # Training pipeline
│   ├── evaluate.py      # Evaluation metrics
│   └── utils.py         # Preprocessing utilities
└── README.md
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
- Lowercasing (`text.lower()`)
- Contraction expansion (e.g., *"can't"* → *"cannot"*)  
- Lemmatization (words reduced to their base form)  
- Tokenization (split into word tokens)  

---

## 🏗️ Model
The classification model is based on **LSTM with Attention**:

- **Embedding Layer** → transforms tokens into dense vectors  
- **BiLSTM Encoder** → captures contextual information  
- **Attention Mechanism** → focuses on words most relevant to the query  
- **Classifier** → outputs relevance score (`0/1`)  

---

## ⚙️ Installation
```bash
# Clone repository
git clone https://github.com/yourname/depression-detection-erisk2024.git
cd depression-detection-erisk2024

# Create environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Training & Evaluation

### Training
```bash
python src/train.py --data data/dataset2024.csv --epochs 20 --batch_size 64 --lr 1e-3
```

### Evaluation
```bash
python src/evaluate.py --model models/bilstm_attention.pt
```

### Prediction
```python
from src.model import BiLSTMWithAttention
from src.utils import predict_one

print(predict_one("I feel so empty and hopeless..."))
# Output: 1 (relevant to depression symptom)
```

---

## 📊 Results (Example)
| Model              | Accuracy | F1-score | MAP  |
|--------------------|----------|----------|------|
| BiLSTM             | 0.82     | 0.80     | 0.74 |
| BiLSTM + Attention | 0.85     | 0.83     | 0.78 |

---

## 📚 References
- [CLEF eRisk 2024 Lab Notebook](https://ceur-ws.org/Vol-3740/paper-72.pdf)  
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  

---

## 📜 License
This project is licensed under the MIT License.
