# Retail Sales Data Analytics & AI-Based Demand Forecasting

**Program:** AICTE | IBM SkillsBuild – Data Analytics with AI Internship 2026
**Organization:** BharatCares

## 📌 Project Overview
This project analyzes retail sales data end-to-end and applies AI/Machine
Learning to (1) forecast future sales demand and (2) segment customers
based on purchasing behavior. It demonstrates the complete data analytics
lifecycle: data cleaning, exploratory data analysis (EDA), feature
engineering, model building, and evaluation.

## 🎯 Objectives
- Clean and preprocess raw retail transaction data
- Perform EDA to uncover sales trends across time, category, and region
- Build an AI regression model to forecast sales demand
- Apply unsupervised learning (KMeans) to segment customers
- Present actionable business insights

## 🗂️ Project Structure
```
├── sales_analytics_ai.py      # Main code (data pipeline + AI models)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
├── Project_Report.docx        # Detailed project report
└── outputs/                   # Generated charts, cleaned data & metrics
    ├── raw_sales_data.csv
    ├── cleaned_sales_data.csv
    ├── 1_monthly_sales_trend.png
    ├── 2_sales_by_category.png
    ├── 3_sales_by_region.png
    ├── 4_correlation_heatmap.png
    ├── 5_feature_importance.png
    ├── 6_actual_vs_predicted.png
    ├── 7_customer_segmentation.png
    └── model_metrics.txt
```

## ⚙️ Tech Stack
- **Language:** Python 3
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **AI Techniques:** Random Forest Regression (forecasting), KMeans
  Clustering (customer segmentation)

## 🚀 How to Run
1. Clone this repository
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline
   ```bash
   python sales_analytics_ai.py
   ```
4. Check the `outputs/` folder for all generated charts, cleaned data,
   and the model performance report.

## 📊 Key Results
- **Forecasting Model Accuracy (R²):** ~0.95+
- Sales peak strongly in **October–December** (festive/seasonal effect)
- **Electronics and Furniture** categories drive the highest revenue
- Identified **3 distinct customer segments** — low, medium, and
  high-value customers — useful for targeted marketing

## 💡 Business Insights
- Inventory should be scaled up ahead of Q4 (Oct–Dec) to meet seasonal demand
- High-value customer segments should be targeted with loyalty offers
- Discount strategy has a measurable but secondary effect on sales
  compared to category and seasonality

## 👤 Author
Submitted as part of the AICTE | IBM SkillsBuild Data Analytics with AI
Internship Program 2026 (17th August – 30th September 2026).
