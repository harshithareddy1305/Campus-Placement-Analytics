# Campus Placement Analytics & Salary Prediction Feasibility Study

## Overview
This project analyzes campus placement data to understand factors influencing placement outcomes and salary offers. It also evaluates the feasibility of predicting salary using academic and profile-based features through exploratory data analysis and basic machine learning models.

The goal of this project is not to achieve high prediction accuracy, but to understand patterns, limitations, and real-world complexity in campus placement salary data.

---

## Problem Statement
Colleges and students are interested in understanding how academic performance, specialization, and work experience influence placement outcomes and salary. Using historical placement data, this project explores these relationships and assesses whether salary prediction is feasible using available features.

---

## Dataset
The dataset contains records of students with the following information:
- Academic performance (SSC, HSC, Degree, MBA percentages)
- Educational background and specialization
- Work experience
- Placement status
- Salary (available only for placed students)

**Note:** Salary values are missing for unplaced students.

---

## Approach

### Data Preparation
- Filtered only placed students for salary analysis
- Removed rows with missing salary values
- Dropped non-informative ID column
- Encoded categorical variables using label encoding
- Performed minimal cleaning to preserve real-world data characteristics

### Feature Engineering
- Created a simple engineered feature:
  - `avg_academic_score`: Average of SSC, HSC, Degree, and MBA percentages

---

## Exploratory Data Analysis (EDA)
The following analyses were performed:
- Salary distribution analysis
- Work experience vs salary comparison
- Academic performance vs salary relationships
- Correlation heatmap of features

### Key Observations
- Salary distribution is right-skewed with a few high-paying outliers
- Work experience generally improves salary outcomes but does not guarantee higher pay
- Academic scores show weak correlation with salary
- No single feature strongly explains salary variation

---

## Machine Learning Models

### Models Evaluated
- **Linear Regression** (Baseline model)
- **Random Forest Regressor** (Non-linear model)

### Evaluation Metrics
- Mean Absolute Error (MAE)
- R² Score

### Model Comparison

| Model | MAE (₹) | R² |
|------|--------|----|
| Linear Regression | ~67,000 | -0.18 |
| Random Forest Regressor | ~81,000 | -0.50 |

---

## Model Comparison and Interpretation
The non-linear Random Forest model did not outperform the linear baseline. This indicates that salary outcomes are not strongly captured by linear or non-linear interactions within the available academic and profile features.

---

## Why Salary Prediction Is Challenging
- Salary is influenced by company-specific, role-based, and market factors not present in the dataset
- Academic and profile features alone have weak relationships with salary
- High variance and skewed salary distribution reduce predictive performance
- Important factors such as interview performance, company brand, and negotiation are unavailable

---

## Conclusion
This project demonstrates that while campus placement data is useful for understanding general trends, salary prediction using academic-only features has limited feasibility. The analysis highlights the importance of domain understanding, honest evaluation, and interpreting model limitations in real-world data science problems.

---

## Technology Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
- **Tools:** VS Code, Command Prompt, Git, GitHub

---

## Project Structure
