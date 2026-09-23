"""
Retail Sales Data Analytics & AI-Based Demand Forecasting
-----------------------------------------------------------
Author : <Your Name>
Program: AICTE | IBM SkillsBuild - Data Analytics with AI Internship 2026

Description:
This project performs an end-to-end data analytics pipeline on a
retail sales dataset:
    1. Data Generation / Loading
    2. Data Cleaning & Preprocessing
    3. Exploratory Data Analysis (EDA) with visualizations
    4. Feature Engineering
    5. AI Model 1: Sales Demand Forecasting (Regression)
    6. AI Model 2: Customer Segmentation (KMeans Clustering)
    7. Insights & Model Evaluation Report

Run:
    python sales_analytics_ai.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

sns.set_style("whitegrid")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# 1. DATA GENERATION (synthetic but realistic retail sales dataset)
# ----------------------------------------------------------------------
def generate_dataset(n_rows: int = 2000) -> pd.DataFrame:
    """Creates a realistic synthetic retail sales dataset."""
    categories = ["Electronics", "Clothing", "Groceries", "Furniture", "Beauty", "Sports"]
    regions = ["North", "South", "East", "West"]
    payment_modes = ["Credit Card", "UPI", "Cash", "Debit Card"]

    dates = pd.date_range(start="2024-01-01", periods=n_rows, freq="6h")

    df = pd.DataFrame({
        "OrderID": range(1001, 1001 + n_rows),
        "Date": dates,
        "Category": np.random.choice(categories, n_rows, p=[0.22, 0.2, 0.25, 0.13, 0.12, 0.08]),
        "Region": np.random.choice(regions, n_rows),
        "PaymentMode": np.random.choice(payment_modes, n_rows),
        "CustomerAge": np.random.randint(18, 65, n_rows),
        "UnitPrice": np.round(np.random.uniform(5, 500, n_rows), 2),
        "Quantity": np.random.randint(1, 10, n_rows),
        "DiscountPct": np.round(np.random.uniform(0, 0.3, n_rows), 2),
    })

    # Seasonal & category-driven signal so the ML model has real patterns to learn
    month_factor = df["Date"].dt.month.map(
        {1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.1,
         7: 1.15, 8: 1.1, 9: 1.05, 10: 1.2, 11: 1.35, 12: 1.5}
    )
    category_factor = df["Category"].map(
        {"Electronics": 1.4, "Furniture": 1.3, "Clothing": 1.0,
         "Groceries": 0.7, "Beauty": 0.9, "Sports": 1.1}
    )

    base_sales = df["UnitPrice"] * df["Quantity"] * (1 - df["DiscountPct"])
    noise = np.random.normal(0, 15, n_rows)
    df["Sales"] = np.round(base_sales * month_factor.values * category_factor.values + noise, 2)
    df["Sales"] = df["Sales"].clip(lower=5)

    # Introduce a few missing values & duplicates to demonstrate cleaning
    for col in ["CustomerAge", "DiscountPct"]:
        idx = np.random.choice(df.index, size=int(0.02 * n_rows), replace=False)
        df.loc[idx, col] = np.nan
    df = pd.concat([df, df.sample(10, random_state=RANDOM_SEED)], ignore_index=True)

    return df


# ----------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="OrderID", keep="first").copy()
    df["CustomerAge"] = df["CustomerAge"].fillna(df["CustomerAge"].median())
    df["DiscountPct"] = df["DiscountPct"].fillna(df["DiscountPct"].mean())
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month_name()
    df["MonthNum"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.day_name()
    print(f"[Cleaning] Removed {before - len(df)} duplicate rows. Missing values imputed.")
    return df


# ----------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    # 3.1 Monthly sales trend
    monthly = df.groupby("MonthNum")["Sales"].sum().reset_index()
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=monthly, x="MonthNum", y="Sales", marker="o", color="#2E86AB")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales (₹)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_monthly_sales_trend.png", dpi=150)
    plt.close()

    # 3.2 Sales by category
    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=cat_sales.values, y=cat_sales.index, palette="viridis")
    plt.title("Total Sales by Category")
    plt.xlabel("Total Sales (₹)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_sales_by_category.png", dpi=150)
    plt.close()

    # 3.3 Regional distribution
    plt.figure(figsize=(6, 6))
    df.groupby("Region")["Sales"].sum().plot.pie(autopct="%1.1f%%", startangle=90)
    plt.title("Sales Share by Region")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_sales_by_region.png", dpi=150)
    plt.close()

    # 3.4 Correlation heatmap
    plt.figure(figsize=(6, 5))
    numeric_cols = ["CustomerAge", "UnitPrice", "Quantity", "DiscountPct", "Sales"]
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_correlation_heatmap.png", dpi=150)
    plt.close()

    print("[EDA] Charts saved to outputs/ folder.")


# ----------------------------------------------------------------------
# 4 & 5. AI MODEL 1 - DEMAND / SALES FORECASTING (Regression)
# ----------------------------------------------------------------------
def train_forecasting_model(df: pd.DataFrame):
    data = df.copy()
    le_cat = LabelEncoder()
    le_region = LabelEncoder()
    le_pay = LabelEncoder()

    data["Category_enc"] = le_cat.fit_transform(data["Category"])
    data["Region_enc"] = le_region.fit_transform(data["Region"])
    data["PaymentMode_enc"] = le_pay.fit_transform(data["PaymentMode"])

    features = ["Category_enc", "Region_enc", "PaymentMode_enc", "CustomerAge",
                "UnitPrice", "Quantity", "DiscountPct", "MonthNum"]
    X = data[features]
    y = data["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"[Forecasting Model] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2 Score: {r2:.3f}")

    # Feature importance chart
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(7, 4.5))
    importance.plot.barh(color="#2E86AB")
    plt.title("Feature Importance - Sales Forecasting Model")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_feature_importance.png", dpi=150)
    plt.close()

    # Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.4, color="#A23B72")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_actual_vs_predicted.png", dpi=150)
    plt.close()

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ----------------------------------------------------------------------
# 6. AI MODEL 2 - CUSTOMER SEGMENTATION (KMeans Clustering)
# ----------------------------------------------------------------------
def customer_segmentation(df: pd.DataFrame):
    cust = df.groupby("CustomerAge").agg(
        TotalSpend=("Sales", "sum"),
        AvgOrderValue=("Sales", "mean"),
        OrderCount=("OrderID", "count")
    ).reset_index()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cust[["TotalSpend", "AvgOrderValue", "OrderCount"]])

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
    cust["Segment"] = kmeans.fit_predict(scaled)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=cust, x="TotalSpend", y="AvgOrderValue",
                     hue="Segment", palette="Set2", s=80)
    plt.title("Customer Segmentation (AI - KMeans Clustering)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/7_customer_segmentation.png", dpi=150)
    plt.close()

    print("[Segmentation] 3 customer segments identified and saved.")
    return cust


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RETAIL SALES DATA ANALYTICS & AI FORECASTING PIPELINE")
    print("=" * 60)

    df_raw = generate_dataset()
    df_raw.to_csv(f"{OUTPUT_DIR}/raw_sales_data.csv", index=False)

    df = clean_data(df_raw)
    df.to_csv(f"{OUTPUT_DIR}/cleaned_sales_data.csv", index=False)

    run_eda(df)
    metrics = train_forecasting_model(df)
    customer_segmentation(df)

    with open(f"{OUTPUT_DIR}/model_metrics.txt", "w") as f:
        f.write("Sales Forecasting Model Performance\n")
        f.write("------------------------------------\n")
        f.write(f"MAE  : {metrics['mae']:.2f}\n")
        f.write(f"RMSE : {metrics['rmse']:.2f}\n")
        f.write(f"R2   : {metrics['r2']:.3f}\n")

    print("\nPipeline completed successfully. All results saved in 'outputs/' folder.")


if __name__ == "__main__":
    main()
