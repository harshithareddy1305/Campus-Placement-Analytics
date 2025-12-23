# Campus Placement Analytics & Salary Forecasting

## Problem Statement
This project analyzes campus placement data to understand factors influencing placement outcomes and salary, and builds a basic machine learning model to predict salary for placed students.

## Dataset
The dataset contains academic performance, work experience, specialization, placement status, and salary information of students. Salary data is available only for placed students.

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Approach
- Filtered placed students for salary analysis
- Cleaned missing values and encoded categorical variables
- Performed exploratory data analysis to identify trends
- Built a Linear Regression model to predict salary
- Evaluated the model using MAE and R² score

## Key Insights
- Work experience generally leads to higher salary outcomes
- Academic scores show limited correlation with salary
- Salary distribution is skewed with a few high-paying offers
- Salary prediction is influenced by multiple external factors

## Model Performance
- Mean Absolute Error (MAE): ~₹67,000
- R² Score: Negative, indicating limited predictive power with available features

## Conclusion
The results reflect real-world placement scenarios where salary is influenced by factors beyond academic performance. The model is suitable for understanding trends rather than making precise predictions.

## How to Run
1. Install dependencies from `requirements.txt`
2. Navigate to the `notebooks` folder
3. Run `python analysis.py`
