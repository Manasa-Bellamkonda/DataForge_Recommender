# 🎬 Fair and Diverse Movie Recommendation System

## 👥 Team 40 (DataForge)

---

## 📌 Project Overview
This project builds a movie recommendation system using the MovieLens dataset.  
The goal is to predict user ratings and generate personalized recommendations  
while analyzing trade-offs between accuracy, diversity, and coverage.

---

## ⚙️ Models Implemented

### 🔹 Baseline Models
- Global Mean Predictor  
- User-Item Bias Model  

### 🔹 Collaborative Filtering
- User-Based kNN  
- Item-Based kNN  

### 🔹 Matrix Factorization
- Latent factor model (SVD-style)

### 🔹 Hybrid Model
- Combines collaborative filtering and matrix factorization  
- Uses hyperparameter tuning (alpha, beta)

---

## 📊 Results

| Model | RMSE |
|------|------|
| Matrix Factorization (Test) | **0.9361** |
| Hybrid Model (Best) | **0.9291** |

👉 Hybrid model achieved the best performance.

---

## 📁 Project Structure

```
DataForge_Recommender/
│
├── data/              # Raw and processed data (not included)
├── notebooks/         # Jupyter notebooks
├── src/               # Core implementation
├── results/           # Metrics, figures, tables
├── reports/           # Project reports
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📥 Dataset

MovieLens dataset:  
https://grouplens.org/datasets/movielens/

(Note: Dataset is not included in this repository.)

---

## ▶️ How to Run

### 1. Create virtual environment (optional)

```
python -m venv venv
venv\Scripts\activate
```
### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the project

- Open notebooks in `/notebooks`
- OR run scripts from `/src`
---

## 🚀 Current Progress (Phase 2)

- ✅ Data preprocessing completed  
- ✅ Baseline models implemented  
- ✅ Collaborative filtering implemented  
- ✅ Matrix factorization implemented  
- ✅ Hybrid model developed  
- ✅ RMSE evaluation completed  

---

## 🔮 Future Work

- MAP@K and NDCG@K  
- Coverage and diversity metrics  
- Fairness-aware recommendation  
- Final evaluation and comparison  

---

## 🧠 Challenges

- Data sparsity  
- Cold-start problem  
- Hyperparameter tuning  
- Accuracy vs diversity trade-off  

---

## 👨‍💻 Author

Team 40 — DataForge

---

## 🔗 Repository
https://github.com/Manasa-Bellamkonda/DataForge_Recommender

---

## 📘 Course

CSC 4740/6740 Data Mining