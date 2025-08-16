# 🏡 House Price Prediction with Explainable AI

This project demonstrates a **Machine Learning pipeline** for predicting house prices, while emphasizing **explainability** so that predictions are transparent and interpretable.

It includes all major steps in a real-world ML workflow:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training with multiple algorithms
- Model evaluation with standard regression metrics
- Explainable AI (XAI) using **SHAP** and **LIME**
- (Optional) Interactive demo with **Streamlit**

---

## 🚀 Features

- Predict house prices using features such as number of rooms, location, square footage, etc.
- Compare performance of models: Linear Regression, Random Forest, Gradient Boosting, XGBoost
- Evaluate models with **RMSE**, **MAE**, and **R²**
- Interpret predictions with **SHAP values** and **LIME explanations**
- Streamlit app to input custom features and see predicted price + explanations

---

## 🗂 Project Structure

```
house-price-prediction/
│
├── data/                     # Dataset (e.g., housing.csv)
│
├── notebooks/                 # Jupyter notebooks for EDA and prototyping
│
├── src/                       # Source code for ML pipeline
│   ├── preprocess.py           # Data cleaning & preprocessing
│   ├── train.py                # Model training & evaluation
│   ├── explain.py              # SHAP/LIME explainability
│   └── app.py                  # (Optional) Streamlit app
│
├── models/                     # Saved trained models
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `pandas`, `numpy` — Data handling
- `matplotlib`, `seaborn` — Visualization
- `scikit-learn` — ML models & preprocessing
- `xgboost` — Gradient boosting
- `shap`, `lime` — Explainable AI
- `streamlit` — (Optional) Web app

### 3. Dataset

Download your dataset (e.g., Kaggle’s **House Prices: Advanced Regression Techniques**) and place it in the `data/` folder.

---

## 🛠 Usage

### Train the Model

```bash
python src/train.py --data data/housing.csv
```

### Generate Explanations

```bash
python src/explain.py --model models/best_model.pkl --data data/housing.csv
```

### Run the Streamlit App

```bash
streamlit run src/app.py
```

---

## 📊 Example Output

- **Model Evaluation**  
  - Linear Regression: RMSE = 42,000, R² = 0.72  
  - Random Forest: RMSE = 28,000, R² = 0.88  
  - XGBoost: RMSE = 25,000, R² = 0.91  

- **Explainability (SHAP)**  
  ![SHAP Plot](docs/shap_example.png)

---

## 🧩 How It Works

1. **Preprocessing**: Handles missing values, encodes categorical variables, scales numeric features.  
2. **Training**: Fits multiple models and selects the best based on validation metrics.  
3. **Evaluation**: Reports RMSE, MAE, R².  
4. **Explainability**: Uses SHAP and LIME to show feature contributions for predictions.  
5. **App**: Lets users enter property details and view predicted price + SHAP explanation.  

---

## 📈 Roadmap

- ✅ Baseline regression models
- ✅ SHAP & LIME explainability
- 🔄 Hyperparameter tuning with Optuna
- 🌍 Deployment on Streamlit Cloud / HuggingFace Spaces
- 🖥 REST API wrapper with FastAPI

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📜 License

MIT License © 2025 
