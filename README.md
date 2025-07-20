# 🛒 Market Basket Analysis & Recommendation System

An interactive web application built with **Streamlit** for performing Market Basket Analysis (MBA) and generating product recommendations. This tool allows users to upload their transactional data, explore it through visualizations, and apply association rule mining algorithms like **Apriori** and **FP-Growth** to uncover hidden patterns in customer purchasing behavior.

---

## ✨ Features

### 📁 Flexible Data Input

* Upload your own transaction data in **CSV**, **XLSX**, or **JSON** formats.
* Use a built-in **sample dataset** to explore the app's features.

### 📊 Interactive Data Exploration (EDA)

* Dataset summaries with key metrics (total transactions, unique products, etc.)
* Analysis of transaction sizes and product frequencies.
* Visualizations of product, category, and brand distributions using **Plotly**.
* Price and rating analysis across different product categories.

### 🧠 Multiple MBA Algorithms

* **Apriori**: A classic algorithm for frequent itemset mining.
* **FP-Growth**: A more efficient alternative to Apriori for large datasets.
* Adjustable parameters: `min_support`, `min_confidence`, `min_lift`.

### 🔍 Association Rule Generation

* Rules displayed in an intuitive **"If-Then"** format.
* Interactive rule table with sorting and filtering.
* Visualizations: **Confidence vs. Lift scatter**, **network graphs** of product associations.

### 💡 Product Recommendation Engine

* **Basket Simulator**: Select items and get real-time recommendations.
* **Product Similarity**: Find frequently co-purchased products.
* **Heatmaps** to visualize product co-occurrence.

### 🎓 Educational Content

* In-app explanations of algorithms with **pseudocode**, **support/confidence/lift** concepts.
* Helpful for students and newcomers to MBA.

### ⬇️ Export Results

* Download frequent itemsets and association rules as **CSV** files.

### 🎨 Customizable UI

* Includes a **Dark Mode** toggle for user comfort.

---

## 📸 Screenshots

> *Add screenshots of your application UI here*

* **Data Exploration Dashboard**
* **Association Rules Network**
* *(Image of EDA section)*
* *(Image of Network Graph)*
* **Recommendation Simulator**
* *(Image of Reco System)*
* *(Image of Algorithm Explanation)*

---

## 🛠️ Technical Stack

* **Backend**: Python
* **Web Framework**: Streamlit
* **Data Manipulation**: Pandas, NumPy
* **MBA Algorithms**: `mlxtend` (Apriori, FP-Growth, Rule Generation)
* **Visualization**: Plotly, Matplotlib, Seaborn
* **Network Analysis**: NetworkX

---

## 🚀 Getting Started

### ✅ Prerequisites

Ensure you have **Python 3.8+** installed.

### 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/mohitkapoor19/market-basket-analysis.git
cd market-basket-analysis
```

Create and activate a virtual environment:

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### ▶️ Running the Application

```bash
streamlit run sample.py
```

The app will open in your default browser.

---

## 📂 File Structure

```
market-basket-analysis/
│
├── .devcontainer/
│   └── devcontainer.json      # Dev Container settings (optional)
│
├── README.md                  # This file
├── requirements.txt           # Project dependencies
└── sample.py                  # Streamlit app entry point
```

---
!
