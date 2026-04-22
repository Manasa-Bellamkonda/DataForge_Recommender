# 🎬 Adaptive Hybrid Recommender System with Dynamic Weighting for Movie Recommendations

## 👤 Author
Manasa Bellamkonda 

## 📌 Project Overview
This project presents a movie recommendation system using the MovieLens dataset.  
The objective is to predict user ratings and generate personalized recommendations by combining global rating trends and similarity-based collaborative filtering methods.

The project introduces a **novel adaptive hybrid model** that dynamically adjusts model weights based on user activity, improving personalization and addressing sparsity and cold-start challenges.

## ⚙️ Models Implemented

### 🔹 Baseline Models
- Global Mean Predictor  
- User-Item Bias Model  

### 🔹 Collaborative Filtering
- User-Based kNN (k = 20)  
- Item-Based kNN (k = 10)  

### 🔹 Matrix Factorization
- Latent factor model (SVD-style)

### 🔹 Hybrid Models
- Static Hybrid Model (fixed weights)  
- **Adaptive Hybrid Model (proposed contribution)**  
  - Dynamically adjusts weights based on user activity  

## 📊 Results

| Model | Test RMSE |
|------|----------|
| Matrix Factorization | 0.9361 |
| Static Hybrid | **0.9291** |
| Adaptive Hybrid | 0.9617 |

👉 The static hybrid model achieved the best performance, while the adaptive model demonstrates dynamic personalization capabilities.

## 📁 Project Structure
```
Sources/
│
├── data/
│ └── processed/
│ ├── train.csv
│ ├── validation.csv
│ └── test.csv
│
├── notebooks/ # Jupyter notebooks (step-by-step execution)
├── src/ # Core model implementations
├── results/ # Figures, metrics
├── requirements.txt
└── README.md
```

## 📥 Dataset

The dataset is included in:
Sources/data/processed/


No additional download is required.

## ▶️ How to Run

### 1. Navigate to project folder
cd Sources

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run notebooks in order
- 01_data_exploration.ipynb
- 02_baseline_models.ipynb
- 03_knn_models.ipynb
- 04_hybrid_models.ipynb
- 05_matrix_factorization.ipynb

## 📈 Key Features

- Hybrid recommendation framework  
- Adaptive weighting mechanism  
- User activity-based personalization  
- RMSE-based evaluation  
- Top-N recommendation generation  

## 🧠 Challenges Addressed

- Data sparsity  
- Cold-start problem  
- Model combination complexity  
- Trade-off between global and local patterns  

## 🎯 Learning Outcomes

- Implementation of multiple recommendation techniques  
- Understanding hybrid model design  
- Practical experience in model evaluation  
- Handling real-world data challenges  

## 📘 Course

CSC 4740/6740 Data Mining  

## 🎥 Demo Video

👉 Watch the project demo here:  

## 🔗 Repository
https://github.com/Manasa-Bellamkonda/DataForge_Recommender