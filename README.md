# 🧠 Parkinson's Disease Detection using SVM and XGBoost

## 📌 Problem Statement
Parkinson’s Disease is a progressive neurological disorder. Early detection is crucial for timely treatment. The goal is to build an accurate model that can classify whether a patient has Parkinson’s based on voice measurements.

## 📂 Dataset

* <strong>Source</strong>: <a href="https://github.com/Sanchitsingh05/Parkinson-s-Disease-Detection-SVM-XGBoost/blob/main/parkinsons.csv">Click here to view the dataset</a>s
* **Features**: Biomedical voice measurements like pitch, jitter, shimmer, and Harmonic-to-Noise Ratio
* **Target**: `status` (1 = Parkinson's Disease, 0 = Healthy)

## 🧰 Tech Stack

* **Python**
* **Pandas & NumPy** for data handling
* **Matplotlib & Seaborn** for data visualization
* **scikit-learn** for SVM and preprocessing
* **XGBoost** for advanced classification
* **Jupyter Notebook**

## ⚙️ Workflow
1. **Data Loading and Exploration**

   * Load CSV into Pandas
   * Check for null values, unique counts, class balance

2. **Exploratory Data Analysis (EDA)**

   * Correlation matrix
   * Distribution of features
   * Class-wise feature comparison

3. **Feature Selection**

   * Dropped unrelated or weakly correlated features
   * Kept key predictors like `MDVP:Fo(Hz)`, `spread1`, `PPE`, etc.

4. **Data Preprocessing**

   * Scaling using `StandardScaler`
   * Train-test split

5. **Model Building**

   * **Support Vector Machine (SVM)**: Good for small datasets and high-dimensional features
   * **XGBoost**: Boosting algorithm for handling complex patterns

6. **Model Evaluation**

   * Accuracy Score
   * Confusion Matrix
   * Ensemble Testing (optional)

## 📊 Results

| Model   | Accuracy |
| ------- | -------- |
| SVM     | \~92%    |
| XGBoost | \~94%    |

Both models performed well, with **XGBoost outperforming SVM**, making it suitable for production-level deployment in healthcare tools for early PD detection.

## 💡 Future Improvements
* Use more clinical features beyond vocal data
* Implement deep learning models like LSTM or CNN for time-series voice data
* Try stacking or voting classifiers for ensemble prediction
* Build a front-end UI for real-time prediction

## 👨‍💻 Owner
This project is created and maintained by **Sanchit Singh** as part of a practical machine learning journey in health diagnostics.

## 🚀 Final Note
This project illustrates the power of ML in assisting early medical diagnosis. With accurate predictions and clear interpretability, models like SVM and XGBoost can pave the way for smarter, faster, and more reliable healthcare solutions.
