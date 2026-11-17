# 🛍️ E-Commerce Intelligence System

An end-to-end data analytics and machine learning project built using Python, SQL, Power BI, and Streamlit to analyze e-commerce business performance and predict potential delivery delays.

## 📌 Project Overview

This project uses the Brazilian Olist E-Commerce dataset to explore customer behavior, sales performance, payment patterns, product trends, and delivery performance.

It combines data analytics, machine learning, and interactive dashboards to generate meaningful business insights.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- SQL and SQLite
- Scikit-learn
- Power BI
- Streamlit
- Plotly

## 📂 Project Structure

```text
E-Commerce Intelligence System/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
├── src/
├── models/
├── app.py
├── requirements.txt
├── E-Commerce Intelligence Dashboard.pbix
└── README.md
```

## 📊 Project Features

### 1. Data Cleaning and Exploration
- Cleaned and processed e-commerce datasets.
- Handled missing values and duplicate records.
- Explored customer, product, order, payment, and seller data.

### 2. SQL Analysis
- Analyzed order status and revenue.
- Identified top-selling product categories.
- Explored monthly revenue and payment methods.
- Analyzed customer locations and delivery performance.

### 3. Machine Learning
- Built a Random Forest classification model to predict delivery delays.
- Compared it with Logistic Regression.
- Evaluated model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Saved the trained model for application use.

### 4. Streamlit Application
- Interactive business overview.
- Sales analytics.
- Customer insights.
- Delivery performance.
- Delivery delay prediction.

### 5. Power BI Dashboard
- 5 KPI cards.
- 7 interactive charts.
- Order status distribution.
- Monthly revenue trends.
- Payment method distribution.
- Top customer states and cities.
- Delivery performance.
- Review score distribution.

## 📈 Key Findings

- Total orders: 99,441
- Total product revenue: R$13,591,643.70
- Average delivery time: 12.56 days
- Average review score: 4.09/5
- Approximately 8.1% of delivered orders were late.

## 🤖 Model Performance

The tuned Random Forest model achieved:

- Accuracy: 83.25%
- ROC-AUC: 0.706

The model is experimental. Its ability to identify late deliveries is limited, so predictions should not be treated as guaranteed outcomes.

## 🚀 Run the Streamlit Application

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python -m streamlit run app.py
```

## 📁 Dataset

Brazilian Olist E-Commerce Public Dataset from Kaggle.

## 🎯 Project Objective

To demonstrate an end-to-end data science workflow involving data preparation, SQL analytics, machine learning, and interactive business intelligence.

---

**Developed as a Data Science portfolio project.** 