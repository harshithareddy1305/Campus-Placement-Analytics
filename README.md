# Campus Placement Analytics & Salary Prediction Feasibility Study

## 📌 Project Overview

This project explores campus placement data to understand the factors influencing placement outcomes and salary offers. It evaluates the feasibility of predicting salaries using academic and profile-based features through exploratory data analysis and basic machine learning models.

The focus of this work is on pattern discovery, model evaluation, and understanding the limitations of predictive modeling in real-world datasets.

---

## 🎯 Problem Statement

Students and institutions are interested in understanding how academic performance, specialization, and work experience influence placement outcomes and salary offers. Using historical placement data, this project investigates these relationships and assesses whether salary prediction is feasible with the available features.

---

## 📊 Dataset

Student placement dataset containing:

- Academic performance (SSC, HSC, Degree, MBA percentages)  
- Educational specialization  
- Work experience  
- Placement status  
- Salary (available only for placed students)  

> Salary values are missing for unplaced students, reflecting real-world data constraints.

---

## 🧹 Data Preparation

- Filtered only placed students for salary analysis  
- Removed records with missing salary values  
- Dropped non-informative identifier columns  
- Encoded categorical variables using label encoding  
- Performed minimal cleaning to preserve real-world data characteristics  

---

## 📈 Exploratory Data Analysis (EDA)

Key analyses included:

- Salary distribution analysis  
- Work experience vs salary comparison  
- Academic performance vs salary relationships  
- Correlation analysis across numerical features  

---

## 🤖 Machine Learning Models

### Models Evaluated
- **Linear Regression** (baseline model)  
- **Random Forest Regressor** (non-linear model)  

---

## 📏 Evaluation Metrics

- Mean Absolute Error (MAE)  
- R² Score  

---

## 🔍 Model Interpretation & Findings

- Academic and profile-based features show weak correlation with salary  
- Random Forest did not outperform the linear baseline  
- Negative R² scores indicate limited predictive capability  
- Salary outcomes are influenced by external factors not captured in the dataset  

---

## ❓ Why Salary Prediction Is Challenging

- Company-specific and role-based factors are unavailable  
- Interview performance and negotiation details are missing  
- High salary variance reduces model effectiveness  

---

## 🛠 Tools & Technologies

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## ✅ Conclusion

This project demonstrates the importance of feature relevance, honest model evaluation, and domain understanding in applied machine learning. It highlights why strong predictive performance is not always achievable with limited real-world data and emphasizes responsible interpretation of model results.
