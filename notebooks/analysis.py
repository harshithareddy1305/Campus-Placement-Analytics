import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


# =========================
# Load Dataset
# =========================
df = pd.read_csv("../data/Placement_Data_Full_Class.csv")

print("Original Dataset Shape:", df.shape)


# =========================
# Data Cleaning (MVP)
# =========================

# Use only placed students (salary available)
df_placed = df[df["status"] == "Placed"].copy()

# Drop rows with missing salary
df_placed.dropna(subset=["salary"], inplace=True)

# Drop ID column
df_placed.drop(columns=["sl_no"], inplace=True)

# Encode categorical features
le = LabelEncoder()
cat_cols = df_placed.select_dtypes(include="object").columns

for col in cat_cols:
    df_placed[col] = le.fit_transform(df_placed[col])

print("Cleaned Dataset Shape:", df_placed.shape)


# =========================
# Exploratory Data Analysis
# =========================

# 1. MBA % vs Salary
plt.figure()
plt.scatter(df_placed["mba_p"], df_placed["salary"])
plt.xlabel("MBA Percentage")
plt.ylabel("Salary")
plt.title("MBA Percentage vs Salary")
plt.savefig("../images/plots/mba_vs_salary.png")
plt.close()

# 2. Work Experience vs Salary
plt.figure()
df_placed.boxplot(column="salary", by="workex")
plt.xlabel("Work Experience (0 = No, 1 = Yes)")
plt.ylabel("Salary")
plt.title("Work Experience vs Salary")
plt.suptitle("")
plt.savefig("../images/plots/workex_vs_salary.png")
plt.close()

# 3. Degree % vs Salary
plt.figure()
plt.scatter(df_placed["degree_p"], df_placed["salary"])
plt.xlabel("Degree Percentage")
plt.ylabel("Salary")
plt.title("Degree Percentage vs Salary")
plt.savefig("../images/plots/degree_vs_salary.png")
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df_placed.corr()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.savefig("../images/plots/correlation_heatmap.png")
plt.close()

# 5. Salary Distribution
plt.figure()
plt.hist(df_placed["salary"], bins=10)
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution")
plt.savefig("../images/plots/salary_distribution.png")
plt.close()


# =========================
# Machine Learning Model
# =========================

X = df_placed.drop(columns=["salary", "status"])
y = df_placed["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)
