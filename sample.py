# Set page configuration
import streamlit as st
st.set_page_config(
    page_title="Market Basket Analysis & Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import pairwise
import base64
from io import BytesIO
import time
from collections import Counter
import re
import math
import warnings
warnings.filterwarnings('ignore')

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #0D47A1;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #666;
        font-size: 0.8rem;
    }
    
    /* Dark mode toggle */
    .dark-mode {
        background-color: #121212;
        color: #e0e0e0;
    }
    .dark-mode .card {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .dark-mode .metric-card {
        background-color: #2c2c2c;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
algo_descriptions = {
    "apriori": """
    ### Apriori Algorithm
    The Apriori algorithm identifies frequent itemsets in a dataset and derives association rules. It uses two key principles:
    1. **If an itemset is frequent, then all of its subsets must also be frequent**
    2. **If an itemset is infrequent, then all of its supersets must also be infrequent**
    
    #### Key Steps:
    1. Generate frequent 1-itemsets
    2. Using frequent k-itemsets, generate candidate (k+1)-itemsets
    3. Prune candidate itemsets if any subset is not frequent
    4. Repeat until no more frequent itemsets are found
    
    #### Pros and Cons:
    - **Pros**: Simple to understand and implement
    - **Cons**: Computationally expensive for large datasets
    """,
    
    "fpgrowth": """
    ### FP-Growth Algorithm
    FP-Growth (Frequent Pattern Growth) is a more efficient alternative to Apriori. Instead of generating candidate itemsets, it uses a compact data structure called an FP-tree.
    
    #### Key Steps:
    1. Build an FP-tree by scanning the dataset twice
    2. Extract frequent itemsets directly from the FP-tree
    
    #### Pros and Cons:
    - **Pros**: More efficient than Apriori, especially for large datasets
    - **Cons**: More complex implementation, requires more memory for the FP-tree
    """,
    
    "association_rules": """
    ### Association Rule Mining
    Association rules are derived from frequent itemsets and represent relationships between items in a dataset.
    
    #### Key Metrics:
    - **Support**: Frequency of itemset occurrence (how popular an itemset is)
    - **Confidence**: Conditional probability of finding the consequent given the antecedent
    - **Lift**: Ratio of observed support to expected support if items were independent
    
    #### Rule Format:
    {Antecedent} → {Consequent} [Support, Confidence, Lift]
    
    #### Application:
    Rules with high confidence and lift are used for product recommendations.
    """
}

# Color palette
colors = {
    'primary': '#1E88E5',
    'secondary': '#FF9800',
    'highlight': '#FFC107',
    'background': '#f9f9f9',
    'text': '#333333',
    'categorical': ['#1E88E5', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4', '#FFEB3B', '#795548', '#607D8B']
}

def load_data(file):
    """
    Load data from an uploaded file. Handles data with generic or missing headers
    as shown in the transaction dataset screenshot.
    """
    try:
        file_name = file.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file_name.endswith(('.xls', '.xlsx')):
            # Try to load Excel file - if headers are all the same or missing
            df = pd.read_excel(file, header=0)
            
            # Check if all columns have the same name (like "InvoiceNo")
            if len(df.columns) > 1 and df.columns.nunique() <= 2:
                # Try reading again without header and assign proper column names
                df = pd.read_excel(file, header=None)
                # Assign meaningful column names based on position
                column_names = ['transaction_id', 'stock_code', 'product_name', 'quantity', 
                               'date_time', 'sale_price', 'customer_id', 'country']
                # If dataframe has fewer columns than our names, truncate the list
                if len(df.columns) < len(column_names):
                    column_names = column_names[:len(df.columns)]
                # If dataframe has more columns, add generic names
                elif len(df.columns) > len(column_names):
                    for i in range(len(column_names), len(df.columns)):
                        column_names.append(f'column_{i+1}')
                df.columns = column_names
        elif file_name.endswith('.json'):
            df = pd.read_json(file)
        else:
            st.error(f"Unsupported file format: {file_name}")
            return None
        
        # If we still have columns with the same name, try to detect column types
        if df.columns.duplicated().any():
            # Use positional logic to rename columns
            original_cols = df.columns.tolist()
            new_cols = []
            for i, col in enumerate(original_cols):
                if i == 0:
                    new_cols.append('transaction_id')
                elif i == 1:
                    new_cols.append('stock_code')
                elif i == 2:
                    new_cols.append('product_name')
                elif i == 3:
                    new_cols.append('quantity')
                elif i == 4:
                    new_cols.append('date_time')
                elif i == 5:
                    new_cols.append('sale_price')
                elif i == 6:
                    new_cols.append('customer_id')
                elif i == 7:
                    new_cols.append('country')
                else:
                    new_cols.append(f'column_{i+1}')
            df.columns = new_cols
        
        # Strip extra whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Handle data quality issues in specific columns
        if 'transaction_id' in df.columns:
            # Convert transaction_id to string to ensure consistent handling
            df['transaction_id'] = df['transaction_id'].astype(str)
        
        # If your data has product code and product description in separate columns,
        # combine them to create a more informative product_name
        if 'stock_code' in df.columns and 'product_name' in df.columns:
            # Combine stock code and description to create a more detailed product name
            df['product_name'] = df['stock_code'] + ' ' + df['product_name'].astype(str)
        
        # Basic data cleaning
        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].fillna('Unknown')
        numeric_columns = df.select_dtypes(include=['float', 'int']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Ensure transaction_id column exists
        if 'transaction_id' not in df.columns:
            st.error("Could not identify a transaction ID column in your data.")
            return None
            
        # Check if we need to create artificial transaction groups
        if df['transaction_id'].nunique() == len(df):
            st.warning("""
            Your data has one product per transaction, which isn't ideal for market basket analysis.
            Creating artificial transaction groups based on categories for demonstration purposes.
            """)
            
            # Check if we have a category column to group by
            if 'country' in df.columns:  # Use country as a proxy for category if needed
                df['transaction_group'] = df['country']
                # Add some randomness to create more interesting transaction groups
                if len(df) > 20:
                    random_groups = np.random.randint(1, 4, size=len(df))
                    df['transaction_group'] = df['country'] + "_" + random_groups.astype(str)
            else:
                # Create random transaction groups
                num_groups = max(5, len(df) // 3)  # Aim for ~3 items per group
                df['transaction_group'] = np.random.randint(1, num_groups+1, size=len(df))
                
            # Preserve original transaction_id
            df.rename(columns={'transaction_id': 'original_transaction_id'}, inplace=True)
            df['transaction_id'] = df['transaction_group']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Create a sample dataset when no file is uploaded
@st.cache_data
def create_sample_dataset(n_rows=100):
    """
    Create a sample dataset for market basket analysis with configurable size.
    
    Parameters:
    n_rows (int): Number of rows to generate. Default: 100
    
    Returns:
    pd.DataFrame: Sample transaction dataset
    """
    # Create a sample dataset based on the provided data schema
    data = {
        'transaction_id': [1, 2, 3, 4, 5, 1, 2, 3],
        'product': [
            'Garlic Oil - Vegetarian Capsule 500 mg',
            'Water Bottle - Orange',
            'Brass Angle Deep - Plain, No.2',
            'Organic Turmeric Powder',
            'LED Light Bulb - 9W',
            'Aloe Vera Gel',
            'Stainless Steel Water Bottle',
            'Room Freshener - Lavender'
        ],
        'category': [
            'Beauty & Hygiene',
            'Kitchen, Garden & Pets',
            'Cleaning & Household',
            'Grocery & Staples',
            'Electronics',
            'Beauty & Hygiene',
            'Kitchen, Garden & Pets',
            'Cleaning & Household'
        ],
        'sub_category': [
            'Hair Care',
            'Storage & Accessories',
            'Pooja Needs',
            'Spices & Masalas',
            'Lighting',
            'Skin Care',
            'Storage & Accessories',
            'Air Fresheners'
        ],
        'brand': [
            'Sri Sri Ayurveda',
            'Mastercook',
            'Trm',
            'Organic India',
            'Philips',
            'Patanjali',
            'Milton',
            'Godrej'
        ],
        'sale_price': [220.00, 180.00, 119.00, 150.00, 199.00, 180.00, 250.00, 90.00],
        'market_price': [220.0, 180.0, 250.0, 175.0, 220.0, 200.0, 300.0, 120.0],
        'type': [
            'Hair Oil & Serum',
            'Water & Fridge Bottles',
            'Lamp & Lamp Oil',
            'Organic',
            'LED Bulbs',
            'Face Gel',
            'Water Bottles',
            'Room Freshener'
        ],
        'rating': [4.1, 2.3, 3.4, 4.5, 4.0, 3.8, 4.2, 3.9],
        'description': [
            'This Product contains Garlic Oil that is known...',
            'Each product is microwave safe (without lid), ...',
            'A perfect gift for all occasions, festivals...',
            'Organic turmeric with high curcumin content...',
            'Energy saving bulb with 2 years warranty...',
            'Pure aloe vera gel with no harmful chemicals...',
            'Vacuum insulated, keeps water hot/cold for 24hrs...',
            'Long lasting fragrance, eliminates odors...'
        ]
    }
    
    # Add more products for larger sample sizes to increase variety
    additional_products = [
        ('Organic Honey', 'Grocery & Staples', 'Organic', 'Organic India', 350, 400, 'Honey', 4.7, 'Pure organic honey sourced from...'),
        ('Yoga Mat', 'Sports & Fitness', 'Yoga', 'Adidas', 999, 1200, 'Exercise Mat', 4.3, 'Anti-slip yoga mat for comfortable...'),
        ('Bluetooth Headphones', 'Electronics', 'Audio', 'JBL', 1499, 1999, 'Wireless Headphones', 4.4, 'Premium sound quality with...'),
        ('Protein Bar', 'Sports & Fitness', 'Nutrition', 'Muscle Blaze', 80, 99, 'Energy Bar', 3.5, 'High protein low sugar energy bar...'),
        ('Scented Candle', 'Home Decor', 'Candles', 'Ikea', 299, 399, 'Aromatherapy', 4.1, 'Long lasting scented candle with...'),
        ('Dish Soap', 'Cleaning & Household', 'Dishwashing', 'Vim', 70, 90, 'Liquid Soap', 4.0, 'Removes tough stains and grease...'),
        ('Face Mask', 'Beauty & Hygiene', 'Skin Care', 'Himalaya', 120, 150, 'Facial', 3.9, 'Purifying face mask with natural...'),
        ('Almonds', 'Grocery & Staples', 'Dry Fruits', 'Nutraj', 450, 499, 'Nuts', 4.6, 'Premium California almonds...'),
        ('USB Cable', 'Electronics', 'Accessories', 'Anker', 299, 399, 'Charging Cable', 4.5, 'Fast charging durable braided cable...'),
        ('Hand Sanitizer', 'Beauty & Hygiene', 'Hand Care', 'Dettol', 50, 60, 'Sanitizer', 4.2, 'Kills 99.9% of germs without water...')
    ]
    
    # Add these additional products to the base data
    for prod in additional_products:
        data['product'].append(prod[0])
        data['category'].append(prod[1])
        data['sub_category'].append(prod[2])
        data['brand'].append(prod[3])
        data['sale_price'].append(prod[4])
        data['market_price'].append(prod[5])
        data['type'].append(prod[6])
        data['rating'].append(prod[7])
        data['description'].append(prod[8])
        # Add a reasonable transaction ID for these products
        data['transaction_id'].append(np.random.randint(6, 15))
    
    # Calculate how many more rows we need to generate
    base_size = len(data['product'])
    additional_needed = max(0, n_rows - base_size)
    
    if additional_needed > 0:
        # Generate transaction IDs with realistic distribution (some transactions have multiple items)
        # For larger datasets, create more unique transaction IDs
        n_transactions = min(30, max(15, n_rows // 10))  # Scale transactions with dataset size
        transaction_ids = np.random.choice(range(1, n_transactions + 1), size=additional_needed, 
                                          p=np.array([1/(i**0.5) for i in range(1, n_transactions + 1)])/
                                            sum([1/(i**0.5) for i in range(1, n_transactions + 1)]))
        
        # Expand the sample data with variations
        for i in range(additional_needed):
            transaction_id = transaction_ids[i]
            product_idx = np.random.randint(0, base_size)
            
            # Add some variation to products
            product = data['product'][product_idx]
            if np.random.random() < 0.5:
                product = f"{product} - Variant {np.random.randint(1, 5)}"
            
            # Add the data point with some variation
            data['transaction_id'].append(transaction_id)
            data['product'].append(product)
            data['category'].append(data['category'][product_idx])
            data['sub_category'].append(data['sub_category'][product_idx])
            data['brand'].append(data['brand'][product_idx])
            
            # Add small variations to price and rating
            base_price = data['sale_price'][product_idx]
            data['sale_price'].append(round(base_price * (0.9 + np.random.random() * 0.2), 2))
            data['market_price'].append(round(data['market_price'][product_idx] * (0.9 + np.random.random() * 0.2), 2))
            data['type'].append(data['type'][product_idx])
            data['rating'].append(round(min(5, max(1, data['rating'][product_idx] + (np.random.random() - 0.5))), 1))
            data['description'].append(data['description'][product_idx])
    
    # Create more realistic market basket patterns by adding similar items to the same transactions
    df = pd.DataFrame(data)
    
    # For large datasets, enhance transaction patterns to create more realistic associations
    if n_rows >= 500:
        # Create association patterns: customers who buy X often buy Y
        common_pairs = [
            ('Water Bottle', 'Yoga Mat'),
            ('Organic Turmeric', 'Organic Honey'),
            ('Aloe Vera Gel', 'Face Mask'),
            ('LED Light Bulb', 'USB Cable'),
            ('Garlic Oil', 'Almonds')
        ]
        
        for pair in common_pairs:
            # Find transactions with first item
            transactions_with_first = df[df['product'].str.contains(pair[0], case=False)]['transaction_id'].unique()
            if len(transactions_with_first) > 0:
                # Add second item to some of these transactions
                for trans_id in transactions_with_first[:len(transactions_with_first)//2]:
                    # Find the second item
                    second_items = df[df['product'].str.contains(pair[1], case=False)]
                    if not second_items.empty:
                        second_item = second_items.iloc[0].copy()
                        second_item['transaction_id'] = trans_id
                        df = pd.concat([df, pd.DataFrame([second_item])], ignore_index=True)
    
    return df

# EDA Functions
def display_dataset_summary(df):
    """Display dataset summary with metrics and sample data"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Dataset Overview</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Rows", f"{df.shape[0]:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Unique Transactions", f"{df['transaction_id'].nunique():,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        product_col = 'product_name' if 'product_name' in df.columns else 'product'
        st.metric("Unique Products", f"{df[product_col].nunique():,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if 'category' in df.columns:
            st.metric("Unique Categories", f"{df['category'].nunique():,}")
        else:
            st.metric("Unique Categories", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def analyze_transaction_data(df):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Transaction Analysis</h3>", unsafe_allow_html=True)
    
    # Define key columns
    transaction_col = 'transaction_id'
    product_col = 'product_name' if 'product_name' in df.columns else 'product'
    
    try:
        transactions_per_product = df.groupby(product_col)[transaction_col].nunique()
        products_per_transaction = df.groupby(transaction_col)[product_col].nunique()
    except Exception as e:
        st.error(f"Error in grouping data: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg Products per Transaction", f"{products_per_transaction.mean():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Max Products per Transaction", f"{products_per_transaction.max()}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg Transactions per Product", f"{transactions_per_product.mean():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Transaction size distribution
    st.subheader("Transaction Size Distribution")
    fig = px.histogram(
        products_per_transaction, 
        nbins=20,
        labels={'value': 'Number of Products', 'count': 'Number of Transactions'},
        title='Distribution of Products per Transaction'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Transaction sample
    st.subheader("Sample Transactions")
    sample_transactions = df.groupby(transaction_col)[product_col].apply(list).reset_index()
    sample_transactions.columns = ['Transaction ID', 'Products']
    sample_transactions['Product Count'] = sample_transactions['Products'].apply(len)
    st.dataframe(sample_transactions.head(10), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def analyze_product_data(df):
    """Analyze product frequency and category distribution"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Product Analysis</h3>", unsafe_allow_html=True)
    
    # Define key columns
    product_col = 'product_name' if 'product_name' in df.columns else 'product'
    
    # Product frequency
    product_counts = df[product_col].value_counts().reset_index()
    product_counts.columns = ['Product', 'Count']
    
    # Top products
    st.subheader("Top Products by Frequency")
    top_n = st.slider("Number of top products to display", min_value=5, max_value=30, value=10)
    
    fig = px.bar(
        product_counts.head(top_n),
        x='Product',
        y='Count',
        title=f'Top {top_n} Products by Frequency'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Category analysis if available
    if 'category' in df.columns:
        st.subheader("Category Distribution")
        category_counts = df['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category bar chart
            fig = px.bar(
                category_counts,
                x='Category',
                y='Count',
                title='Number of Products by Category'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category pie chart
            fig = px.pie(
                category_counts,
                names='Category',
                values='Count',
                title='Category Distribution'
            )
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3))
            st.plotly_chart(fig, use_container_width=True)
        
        # Price analysis by category
        if 'sale_price' in df.columns:
            st.subheader("Price Analysis by Category")
            
            price_by_category = df.groupby('category')['sale_price'].agg(['mean', 'min', 'max']).reset_index()
            price_by_category.columns = ['Category', 'Average Price', 'Min Price', 'Max Price']
            price_by_category = price_by_category.sort_values('Average Price', ascending=False)
            
            fig = px.bar(
                price_by_category,
                x='Category',
                y='Average Price',
                error_y=price_by_category['Max Price'] - price_by_category['Average Price'],
                error_y_minus=price_by_category['Average Price'] - price_by_category['Min Price'],
                title='Price Distribution by Category'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sub-category analysis if available
        if 'sub_category' in df.columns:
            st.subheader("Sub-category Distribution")
            
            # Count products by sub-category
            subcategory_counts = df['sub_category'].value_counts().reset_index()
            subcategory_counts.columns = ['Sub-category', 'Count']
            
            # Create treemap of category > subcategory
            category_subcategory = df.groupby(['category', 'sub_category']).size().reset_index()
            category_subcategory.columns = ['Category', 'Sub-category', 'Count']
            
            fig = px.treemap(
                category_subcategory,
                path=['Category', 'Sub-category'],
                values='Count',
                title='Category and Sub-category Hierarchy'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Brand analysis if available
    if 'brand' in df.columns:
        st.subheader("Brand Analysis")
        
        brand_counts = df['brand'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        
        top_brands = st.slider("Number of top brands to display", min_value=5, max_value=20, value=10)
        
        fig = px.bar(
            brand_counts.head(top_brands),
            x='Brand',
            y='Count',
            title=f'Top {top_brands} Brands by Frequency'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Rating analysis if available
    if 'rating' in df.columns:
        st.subheader("Rating Distribution")
        
        fig = px.histogram(
            df,
            x='rating',
            nbins=10,
            title='Distribution of Product Ratings'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average rating by category if both columns exist
        if 'category' in df.columns:
            avg_rating_by_category = df.groupby('category')['rating'].mean().reset_index()
            avg_rating_by_category.columns = ['Category', 'Average Rating']
            avg_rating_by_category = avg_rating_by_category.sort_values('Average Rating', ascending=False)
            
            fig = px.bar(
                avg_rating_by_category,
                x='Category',
                y='Average Rating',
                title='Average Rating by Category'
            )
            fig.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 5]))
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def prepare_transactions_for_mba(df):
    """
    Transform raw data into transaction format for market basket analysis.
    Returns:
        - transactions_list: List of lists, each inner list contains products from one transaction
        - formatted_df: One-hot encoded DataFrame ready for MBA algorithms
    """
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Data Preparation for Market Basket Analysis</h3>", unsafe_allow_html=True)
    
    # Define key columns
    transaction_col = 'transaction_id'
    product_col = 'product_name' if 'product_name' in df.columns else 'product'
    
    # Group products by transaction
    transactions = df.groupby(transaction_col)[product_col].apply(list).reset_index()
    transactions.columns = ['Transaction ID', 'Products']
    
    # Display sample transactions
    st.subheader("Sample Transactions")
    transactions['Product Count'] = transactions['Products'].apply(len)
    st.dataframe(transactions.head(5), use_container_width=True)
    
    # Get list of transaction lists
    transactions_list = transactions['Products'].tolist()
    
    # One-hot encode transactions
    st.subheader("One-hot Encoded Transactions")
    
    # Create one-hot encoded matrix
    te = TransactionEncoder()
    te_data = te.fit_transform(transactions_list)
    formatted_df = pd.DataFrame(te_data, columns=te.columns_)
    
    # Display one-hot encoded sample
    st.markdown("First 5 rows and 10 columns of one-hot encoded transactions:")
    st.dataframe(formatted_df.iloc[:5, :10], use_container_width=True)
    
    # Visualize transaction density
    st.subheader("Transaction Density Heatmap")
    
    # Create a smaller subset for visualization
    sample_size = min(20, formatted_df.shape[0])
    feature_size = min(20, formatted_df.shape[1])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        formatted_df.iloc[:sample_size, :feature_size],
        cmap='viridis',
        cbar_kws={'label': 'Present in Transaction'},
        ax=ax
    )
    plt.title('Transaction-Product Matrix (Sample)')
    plt.xlabel('Products')
    plt.ylabel('Transactions')
    st.pyplot(fig)
    
    # Add a prominent button to proceed to Market Basket Analysis
    st.markdown("<h3>Ready for Market Basket Analysis</h3>", unsafe_allow_html=True)
    st.markdown("Your data is now prepared for market basket analysis. Click below to proceed.")
    
    # Store data in session state so it's accessible after navigation
    if 'transactions_list' not in st.session_state:
        st.session_state.transactions_list = transactions_list
    if 'formatted_df' not in st.session_state:
        st.session_state.formatted_df = formatted_df
    
    # Button to navigate to Market Basket Analysis
    proceed_button = st.button("Proceed to Market Basket Analysis", key="proceed_to_mba", 
                              help="Click to run market basket analysis on the prepared data")
    
    if proceed_button:
        st.session_state.app_mode = "Market Basket Analysis"
        st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return transactions_list, formatted_df

def run_apriori_algorithm(formatted_df, min_support=0.01):
    """Run Apriori algorithm and display results"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Apriori Algorithm</h3>", unsafe_allow_html=True)
    
    # Show algorithm description
    with st.expander("How Apriori Algorithm Works"):
        st.markdown(algo_descriptions["apriori"], unsafe_allow_html=True)
    
    # Support threshold slider
    min_support = st.slider(
        "Minimum Support Threshold",
        min_value=0.001,
        max_value=0.5,
        value=min_support,
        format="%.3f",
        help="Minimum frequency threshold for itemsets to be considered frequent"
    )
    
    # Run Apriori with progress indicator
    with st.spinner("Running Apriori algorithm..."):
        start_time = time.time()
        
        # Run Apriori
        frequent_itemsets = apriori(
            formatted_df,
            min_support=min_support,
            use_colnames=True,
            verbose=0
        )
        
        execution_time = time.time() - start_time
    
    # Display results
    if frequent_itemsets.empty:
        st.warning(f"No frequent itemsets found with minimum support = {min_support}. Try lowering the threshold.")
    else:
        # Add length of itemsets
        frequent_itemsets['itemset_length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Frequent Itemsets Found", len(frequent_itemsets))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Max Itemset Size", frequent_itemsets['itemset_length'].max())
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Execution Time", f"{execution_time:.2f} seconds")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display frequent itemsets
        st.subheader("Frequent Itemsets")
        
        # Convert itemsets to string for better display
        frequent_itemsets_display = frequent_itemsets.copy()
        frequent_itemsets_display['itemsets_str'] = frequent_itemsets_display['itemsets'].apply(lambda x: ', '.join(list(x)))
        
        # Sort by support and display
        frequent_itemsets_display = frequent_itemsets_display.sort_values('support', ascending=False)
        st.dataframe(
            frequent_itemsets_display[['itemsets_str', 'support', 'itemset_length']].rename(
                columns={'itemsets_str': 'Items', 'support': 'Support', 'itemset_length': 'Size'}
            ),
            use_container_width=True
        )
        
        # Visualize support distribution
        st.subheader("Support Distribution by Itemset Size")
        
        fig = px.box(
            frequent_itemsets,
            x='itemset_length',
            y='support',
            color='itemset_length',
            labels={'itemset_length': 'Itemset Size', 'support': 'Support'},
            title='Support Distribution by Itemset Size'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top frequent itemsets bar chart
        st.subheader("Top Frequent Itemsets")
        
        top_n = min(10, len(frequent_itemsets_display))
        top_itemsets = frequent_itemsets_display.head(top_n)
        
        fig = px.bar(
            top_itemsets,
            x='itemsets_str',
            y='support',
            color='itemset_length',
            labels={'itemsets_str': 'Items', 'support': 'Support', 'itemset_length': 'Size'},
            title=f'Top {top_n} Frequent Itemsets by Support'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return frequent_itemsets

def run_fpgrowth_algorithm(formatted_df, min_support=0.01):
    """Run FP-Growth algorithm and display results"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>FP-Growth Algorithm</h3>", unsafe_allow_html=True)
    
    # Show algorithm description
    with st.expander("How FP-Growth Algorithm Works"):
        st.markdown(algo_descriptions["fpgrowth"], unsafe_allow_html=True)
    
    # Support threshold slider
    min_support = st.slider(
        "Minimum Support Threshold (FP-Growth)",
        min_value=0.001,
        max_value=0.5,
        value=min_support,
        format="%.3f",
        help="Minimum frequency threshold for itemsets to be considered frequent"
    )
    
    # Run FP-Growth with progress indicator
    with st.spinner("Running FP-Growth algorithm..."):
        start_time = time.time()
        
        # Run FP-Growth
        frequent_itemsets = fpgrowth(
            formatted_df,
            min_support=min_support,
            use_colnames=True
        )
        
        execution_time = time.time() - start_time
    
    # Display results
    if frequent_itemsets.empty:
        st.warning(f"No frequent itemsets found with minimum support = {min_support}. Try lowering the threshold.")
    else:
        # Add length of itemsets
        frequent_itemsets['itemset_length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Frequent Itemsets Found", len(frequent_itemsets))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Max Itemset Size", frequent_itemsets['itemset_length'].max())
            # Check for dataset unique values before displaying
            if 'itemset_length' in frequent_itemsets:
                max_size = frequent_itemsets['itemset_length'].max()
                st.metric("Max Itemset Size", max_size)
            else:
                st.warning("Unable to determine max itemset size")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Execution Time", f"{execution_time:.2f} seconds")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display frequent itemsets
        st.subheader("Frequent Itemsets (FP-Growth)")
        
        # Convert itemsets to string for better display
        frequent_itemsets_display = frequent_itemsets.copy()
        frequent_itemsets_display['itemsets_str'] = frequent_itemsets_display['itemsets'].apply(lambda x: ', '.join(list(x)))
        
        # Sort by support and display
        frequent_itemsets_display = frequent_itemsets_display.sort_values('support', ascending=False)
        st.dataframe(
            frequent_itemsets_display[['itemsets_str', 'support', 'itemset_length']].rename(
                columns={'itemsets_str': 'Items', 'support': 'Support', 'itemset_length': 'Size'}
            ),
            use_container_width=True
        )
        
        # Visualize frequent itemsets by size
        st.subheader("Frequent Itemsets by Size")
        
        size_counts = frequent_itemsets['itemset_length'].value_counts().sort_index().reset_index()
        size_counts.columns = ['Size', 'Count']
        
        fig = px.bar(
            size_counts,
            x='Size',
            y='Count',
            title='Number of Frequent Itemsets by Size',
            labels={'Size': 'Itemset Size', 'Count': 'Number of Itemsets'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualize FP-Tree structure (simplified representation)
        st.subheader("FP-Tree Structure Visualization")
        st.markdown("""
        This visualization shows a simplified representation of the FP-Tree structure.
        Nodes represent items, and edge weights represent their co-occurrence frequency.
        """)
        
        # Create a graph of frequent pairs to represent the FP-Tree structure
        G = nx.Graph()
        
        # Add edges for 2-itemsets
        pairs = frequent_itemsets[frequent_itemsets['itemset_length'] == 2]
        
        if not pairs.empty:
            for _, row in pairs.iterrows():
                items = list(row['itemsets'])
                G.add_edge(items[0], items[1], weight=row['support'])
            
            # Keep only top edges for visualization clarity
            if len(G.edges) > 50:
                edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:50]
                G = nx.Graph()
                for u, v, data in edges:
                    G.add_edge(u, v, weight=data['weight'])
            
            # Create positions
            pos = nx.spring_layout(G, seed=42)
            
            # Create node sizes based on frequency in 1-itemsets
            singles = frequent_itemsets[frequent_itemsets['itemset_length'] == 1]
            node_sizes = {}
            for item in G.nodes():
                support = singles.loc[singles['itemsets'].apply(lambda x: item in x), 'support'].values
                node_sizes[item] = 300 * (support[0] * 10) if len(support) > 0 else 100
            
            # Create edge widths based on support
            edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]
            
            # Draw the graph
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, 
                node_size=[node_sizes[node] for node in G.nodes()],
                node_color='skyblue',
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos, 
                width=edge_widths,
                edge_color='gray',
                alpha=0.5
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos, 
                font_size=10,
                font_family='sans-serif'
            )
            
            plt.title('Frequent Itemset Network (Simplified FP-Tree Representation)')
            plt.axis('off')
            st.pyplot(fig)
        else:
            st.info("Not enough 2-itemsets to visualize FP-Tree structure. Try lowering the support threshold.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.0):
    """Generate and display association rules from frequent itemsets"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Association Rules</h3>", unsafe_allow_html=True)
    
    # Show algorithm description
    with st.expander("How Association Rule Mining Works"):
        st.markdown(algo_descriptions["association_rules"], unsafe_allow_html=True)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=min_confidence,
            format="%.2f",
            help="Minimum confidence threshold for rules"
        )
    
    with col2:
        min_lift = st.slider(
            "Minimum Lift",
            min_value=0.1,
            max_value=10.0,
            value=min_lift,
            format="%.2f",
            help="Minimum lift threshold for rules"
        )
    
    # Check if we have frequent itemsets
    if frequent_itemsets.empty:
        st.warning("No frequent itemsets available. Please run Apriori or FP-Growth algorithm first.")
        rules_df = pd.DataFrame()
    else:
        # Generate rules with progress indicator
        with st.spinner("Generating association rules..."):
            try:
                # Generate rules
                rules_df = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=min_confidence
                )
                
                # Filter by lift if the column exists and DataFrame is not empty
                if not rules_df.empty and 'lift' in rules_df.columns:
                    rules_df = rules_df[rules_df['lift'] >= min_lift]
                else:
                    st.warning("No rules could be generated with the current confidence threshold.")
                    rules_df = pd.DataFrame()
            except Exception as e:
                st.error(f"Error generating association rules: {str(e)}")
                rules_df = pd.DataFrame()
    
    # Display results
    if rules_df.empty:
        st.warning("No association rules found with the current thresholds. Try lowering confidence or lift.")
    else:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Rules", len(rules_df))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg Confidence", f"{rules_df['confidence'].mean():.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg Lift", f"{rules_df['lift'].mean():.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display rules in If-Then format
        st.subheader("Top Association Rules (If-Then Format)")
        
        # Get top rules sorted by lift
        top_rules = rules_df.sort_values('lift', ascending=False).head(15)
        
        for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            st.markdown(f"""
            <div style="margin-bottom: 10px; padding: 10px; border-left: 4px solid #1E88E5; background-color: #f5f5f5;">
                <b>Rule {i}:</b> If a customer buys <b>{', '.join(antecedents)}</b>, then they will likely buy <b>{', '.join(consequents)}</b>
                <br>
                <span style="font-size: 0.9em; color: #555;">
                    Support: {rule['support']:.3f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # Prepare rules for display in table
        rules_display = rules_df.copy()
        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Display rules in table format
        st.subheader("Association Rules Table")
        
        # Sort the dataframe before renaming columns to avoid KeyError
        if 'lift' in rules_display.columns:
            rules_display = rules_display.sort_values('lift', ascending=False)
        
        st.dataframe(
            rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].rename(
                columns={
                    'antecedents': 'Antecedent (If)',
                    'consequents': 'Consequent (Then)',
                    'support': 'Support',
                    'confidence': 'Confidence',
                    'lift': 'Lift'
                }
            ),
            use_container_width=True
        )
        
        # Visualize metrics
        st.subheader("Rule Metrics Visualization")
        
        # Create a Plotly-friendly version of the dataframe
        plotly_df = rules_df.copy()
        # Convert frozensets to strings to make them JSON serializable
        plotly_df['antecedents_str'] = plotly_df['antecedents'].apply(lambda x: ', '.join(list(x)))
        plotly_df['consequents_str'] = plotly_df['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Confidence vs Lift scatter plot
        fig = px.scatter(
            plotly_df,
            x='confidence',
            y='lift',
            size='support',
            color='support',
            hover_data=['antecedents_str', 'consequents_str'],  # Use string versions for hover
            labels={
                'confidence': 'Confidence', 
                'lift': 'Lift', 
                'support': 'Support',
                'antecedents_str': 'Antecedents',
                'consequents_str': 'Consequents'
            },
            title='Rule Metrics: Confidence vs Lift',
            color_continuous_scale='viridis',
        )
        fig.update_layout(coloraxis_colorbar=dict(title='Support'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Network visualization of rules
        st.subheader("Association Rules Network")
        st.markdown("""
        This network visualization shows the relationships between products based on association rules.
        Nodes are products, and directed edges represent rules (antecedent → consequent).
        """)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add top N rules for visualization clarity
        top_n = min(50, len(rules_df))
        top_rules = rules_df.sort_values('lift', ascending=False).head(top_n)
        
        for _, row in top_rules.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            
            # Add edges for each antecedent-consequent pair
            for a in antecedents:
                for c in consequents:
                    G.add_edge(a, c, weight=row['lift'], confidence=row['confidence'])
        
        # Create positions
        pos = nx.spring_layout(G, seed=42)
        
        # Create node sizes based on degree
        node_sizes = {node: 200 + 100 * G.degree(node) for node in G.nodes()}
        
        # Create edge widths based on lift
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        
        # Draw the graph
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=[node_sizes[node] for node in G.nodes()],
            node_color='lightblue',
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            width=edge_widths,
            edge_color='gray',
            alpha=0.6,
            arrowsize=15,
            arrowstyle='->'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, 
            font_size=9,
            font_family='sans-serif'
        )
        
        plt.title('Association Rules Network')
        plt.axis('off')
        st.pyplot(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return rules_df

def build_recommendation_system(rules_df, transactions_list, formatted_df):
    """Build and display product recommendation system"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Product Recommendation System</h3>", unsafe_allow_html=True)
    
    # Check if we have rules
    if rules_df.empty:
        st.warning("No association rules available. Please generate rules first.")
        return
    
    # Get unique products
    unique_products = formatted_df.columns.tolist()
    
    # Create basket simulator
    st.subheader("Basket Simulator")
    st.markdown("""
    Select products to add to your basket, and the system will recommend other products 
    based on association rules.
    """)
    
    # Multi-select for products
    selected_products = st.multiselect(
        "Select products for your basket",
        options=unique_products,
        help="Select one or more products to add to your basket"
    )
    
    # Display recommendations if products are selected
    if selected_products:
        st.subheader("Recommendations")
        
        # Find rules that match the selected products
        matching_rules = []
        
        for _, rule in rules_df.iterrows():
            antecedents = set(rule['antecedents'])
            if antecedents.issubset(set(selected_products)):
                consequents = set(rule['consequents'])
                if not consequents.issubset(set(selected_products)):
                    matching_rules.append(rule)
        
        if matching_rules:
            # Convert to DataFrame
            recommendations_df = pd.DataFrame(matching_rules)
            
            # Sort by metrics
            metric_option = st.radio(
                "Sort recommendations by:",
                options=['Lift', 'Confidence', 'Support'],
                horizontal=True
            )
            
            recommendations_df = recommendations_df.sort_values(metric_option.lower(), ascending=False)
            
            # Convert sets to strings for display
            recommendations_df['antecedents_str'] = recommendations_df['antecedents'].apply(lambda x: ', '.join(list(x)))
            recommendations_df['consequents_str'] = recommendations_df['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Display top recommendations
            st.subheader("Top Recommendations")
            
            # Create recommendation cards
            for i, (_, rec) in enumerate(recommendations_df.head(5).iterrows()):
                consequents_list = list(rec['consequents'])
                for item in consequents_list:
                    if item not in selected_products:
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px 0;">
                            <h4>{item}</h4>
                            <p><b>Based on:</b> {rec['antecedents_str']}</p>
                            <p><b>Lift:</b> {rec['lift']:.2f} | <b>Confidence:</b> {rec['confidence']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            # Visualize recommendations
            st.subheader("Recommendation Visualization")
                
            # Create a graph of selected products and recommendations
            G = nx.DiGraph()
            
            # Add selected products as nodes
            for product in selected_products:
                G.add_node(product, selected=True)
            
            # Add recommendations and edges
            for _, rec in recommendations_df.head(10).iterrows():
                for item in rec['consequents']:
                    if item not in selected_products:
                        G.add_node(item, selected=False)
                        for source in rec['antecedents']:
                            G.add_edge(source, item, weight=rec['lift'])
            
            # Create positions
            pos = nx.spring_layout(G, seed=42)
            
            # Draw the graph
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Draw selected product nodes
            selected_nodes = [node for node, data in G.nodes(data=True) if data.get('selected', False)]
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=selected_nodes,
                node_size=500,
                node_color='#1E88E5',
                alpha=0.8,
                label='Selected Products'
            )
            
            # Draw recommended product nodes
            recommended_nodes = [node for node, data in G.nodes(data=True) if not data.get('selected', False)]
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=recommended_nodes,
                node_size=400,
                node_color='#FF9800',
                alpha=0.8,
                label='Recommended Products'
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos, 
                width=1.5,
                edge_color='gray',
                alpha=0.6,
                arrowsize=15,
                arrowstyle='->'
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos, 
                font_size=10,
                font_family='sans-serif'
            )
            
            plt.title('Product Recommendations Network')
            plt.axis('off')
            plt.legend()
            st.pyplot(fig)
            
            # Product co-occurrence analysis
            st.subheader("Product Co-occurrence Analysis")
            
            # Calculate co-occurrence matrix
            product_matrix = formatted_df.T.dot(formatted_df)
            np.fill_diagonal(product_matrix.values, 0)  # Remove self-connections
            
            # Filter to selected products and top recommendations
            focus_products = selected_products.copy()
            for _, rec in recommendations_df.head(5).iterrows():
                for item in rec['consequents']:
                    if item not in focus_products:
                        focus_products.append(item)
            
            # Create co-occurrence heatmap
            if len(focus_products) > 1:
                focus_matrix = product_matrix.loc[focus_products, focus_products]
                
                fig = px.imshow(
                    focus_matrix,
                    color_continuous_scale="Viridis",
                    labels=dict(x="Product", y="Product", color="Co-occurrence Count"),
                    title="Product Co-occurrence Heatmap"
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No recommendations found for the selected products. Try selecting different or more products.")
    
    # Advanced: Product similarity search
    st.subheader("Product Similarity Search")
    st.markdown("""
    Find products similar to a selected product based on their co-occurrence patterns.
    """)
    
    # Select a product
    selected_product = st.selectbox(
        "Select a product to find similar items",
        options=unique_products
    )
    
    if selected_product:
        # Calculate product similarity
        product_matrix = formatted_df.T.dot(formatted_df)
        np.fill_diagonal(product_matrix.values, 0)  # Remove self-connections
        
        # Get similarity scores for the selected product
        similarity_scores = product_matrix[selected_product].sort_values(ascending=False)
        
        # Display top similar products
        st.subheader(f"Products Similar to {selected_product}")
        
        similarity_df = similarity_scores.head(10).reset_index()
        similarity_df.columns = ['Product', 'Co-occurrence Score']
        
        # Visualize similar products
        fig = px.bar(
            similarity_df,
            x='Co-occurrence Score',
            y='Product',
            orientation='h',
            title=f"Top Products Similar to {selected_product}",
            labels={'Co-occurrence Score': 'Co-occurrence Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_technical_details():
    """Display technical details about the algorithms"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Technical Details</h3>", unsafe_allow_html=True)
    
    # Algorithm tabs
    tab1, tab2, tab3 = st.tabs(["Apriori Algorithm", "FP-Growth Algorithm", "Association Rules"])
    
    with tab1:
        st.markdown(algo_descriptions["apriori"], unsafe_allow_html=True)
        
        # Apriori pseudocode
        st.subheader("Apriori Algorithm Pseudocode")
        st.code("""
        Apriori(dataset, min_support):
            # First iteration: find frequent 1-itemsets
            L1 = {frequent 1-itemsets}
            
            # Subsequent iterations: generate candidate k-itemsets and find frequent k-itemsets
            for k = 2; Lk-1 is not empty; k++:
                # Generate candidate itemsets
                Ck = generate_candidates(Lk-1)
                
                # Prune candidates
                for each transaction t in dataset:
                    increment the count of all candidates in Ck that are contained in t
                
                # Extract frequent itemsets
                Lk = {c in Ck | c.support >= min_support}
            
            return the union of all Lk
        """)
        
        # Apriori example
        st.subheader("Apriori Example")
        st.markdown("""
        Consider a dataset with the following transactions:
        
        T1: {Bread, Milk}
        T2: {Bread, Diapers, Beer, Eggs}
        T3: {Milk, Diapers, Beer, Cola}
        T4: {Bread, Milk, Diapers, Beer}
        T5: {Bread, Milk, Diapers, Cola}
        
        With min_support = 0.4 (40% of transactions):
        
        1. Find frequent 1-itemsets:
           - {Bread}: 4/5 = 0.8 ✓
           - {Milk}: 4/5 = 0.8 ✓
           - {Diapers}: 4/5 = 0.8 ✓
           - {Beer}: 3/5 = 0.6 ✓
           - {Cola}: 2/5 = 0.4 ✓
           - {Eggs}: 1/5 = 0.2 ✗
        
        2. Generate candidate 2-itemsets from L1:
           - {Bread, Milk}: 3/5 = 0.6 ✓
           - {Bread, Diapers}: 3/5 = 0.6 ✓
           - {Bread, Beer}: 2/5 = 0.4 ✓
           - {Bread, Cola}: 1/5 = 0.2 ✗
           - {Milk, Diapers}: 3/5 = 0.6 ✓
           - {Milk, Beer}: 2/5 = 0.4 ✓
           - {Milk, Cola}: 2/5 = 0.4 ✓
           - {Diapers, Beer}: 3/5 = 0.6 ✓
           - {Diapers, Cola}: 2/5 = 0.4 ✓
           - {Beer, Cola}: 1/5 = 0.2 ✗
        
        3. Continue with 3-itemsets...
        """)
    
    with tab2:
        st.markdown(algo_descriptions["fpgrowth"], unsafe_allow_html=True)
        
        # FP-Growth visualization
        st.subheader("FP-Tree Construction")
        st.markdown("""
        The FP-Growth algorithm constructs an FP-Tree data structure that compactly represents the transaction database.
        
        **Steps to build the FP-Tree:**
        1. Scan the dataset to find frequent 1-itemsets and their support counts
        2. Sort frequent items in descending order of their support
        3. Scan the database again to construct the FP-Tree:
           - For each transaction, sort its frequent items by the global order
           - Insert the sorted transaction into the tree
        
        **Benefits of FP-Tree:**
        - Avoids generating candidate itemsets
        - Compresses the database into a compact structure
        - Only needs to scan the database twice
        """)
        
        # Add FP-Tree visualization
        st.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2020/11/FP_Tree.png", 
                 caption="Example of an FP-Tree Structure", use_column_width=True)
    
    with tab3:
        st.markdown(algo_descriptions["association_rules"], unsafe_allow_html=True)
        
        # Rule generation
        st.subheader("Association Rule Generation")
        st.markdown("""
        Association rules are generated from frequent itemsets using the following metrics:
        
        **Support:** The frequency of an itemset in the database
        ```
        Support(X) = (Number of transactions containing X) / (Total number of transactions)
        ```
        
        **Confidence:** The likelihood that Y is purchased when X is purchased
        ```
        Confidence(X → Y) = Support(X ∪ Y) / Support(X)
        ```
        
        **Lift:** How much more likely Y is purchased when X is purchased compared to random chance
        ```
        Lift(X → Y) = Confidence(X → Y) / Support(Y)
        ```
        
        **Interpretation:**
        - Lift > 1: Positive correlation (X and Y appear together more than expected)
        - Lift = 1: No correlation (X and Y are independent)
        - Lift < 1: Negative correlation (X and Y appear together less than expected)
        """)
        
        # Association rules example
        st.subheader("Association Rules Example")
        st.markdown("""
        Given frequent itemset {Bread, Milk, Diapers} with support = 0.6, we can generate these rules:
        
        1. {Bread, Milk} → {Diapers}
           - Support = 0.6 (appears in 60% of transactions)
           - Confidence = Support(Bread, Milk, Diapers) / Support(Bread, Milk) = 0.6 / 0.6 = 1.0 (100%)
           - Lift = Confidence / Support(Diapers) = 1.0 / 0.8 = 1.25
        
        2. {Bread, Diapers} → {Milk}
           - Support = 0.6
           - Confidence = Support(Bread, Milk, Diapers) / Support(Bread, Diapers) = 0.6 / 0.6 = 1.0 (100%)
           - Lift = Confidence / Support(Milk) = 1.0 / 0.8 = 1.25
        
        3. {Milk, Diapers} → {Bread}
           - Support = 0.6
           - Confidence = Support(Bread, Milk, Diapers) / Support(Milk, Diapers) = 0.6 / 0.6 = 1.0 (100%)
           - Lift = Confidence / Support(Bread) = 1.0 / 0.8 = 1.25
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

def export_results(rules_df, frequent_itemsets_df):
    """Provide options to export analysis results"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Export Results</h3>", unsafe_allow_html=True)
    
    # Check if there's data to export
    if rules_df.empty and frequent_itemsets_df.empty:
        st.warning("No results available for export. Please run the analysis first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Create tabs for different exports
    tab1, tab2 = st.tabs(["Export Rules", "Export Frequent Itemsets"])
    
    with tab1:
        if rules_df.empty:
            st.warning("No association rules available for export.")
        else:
            # Prepare rules for export
            export_rules = rules_df.copy()
            export_rules['antecedents'] = export_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            export_rules['consequents'] = export_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Select columns to export
            st.subheader("Export Association Rules")
            export_columns = st.multiselect(
                "Select columns to export",
                options=export_rules.columns,
                default=['antecedents', 'consequents', 'support', 'confidence', 'lift']
            )
            
            if export_columns:
                # Create CSV
                csv = export_rules[export_columns].to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download Association Rules (CSV)",
                    data=csv,
                    file_name="association_rules.csv",
                    mime="text/csv"
                )
        
        with tab2:
            if frequent_itemsets_df.empty:
                st.warning("No frequent itemsets available for export.")
            else:
                # Prepare itemsets for export
                export_itemsets = frequent_itemsets_df.copy()
                export_itemsets['itemsets'] = export_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
                
                # Select columns to export
                st.subheader("Export Frequent Itemsets")
                export_columns = st.multiselect(
                    "Select columns to export",
                    options=export_itemsets.columns,
                    default=['itemsets', 'support', 'itemset_length']
                )
            
            if export_columns:
                # Create CSV
                csv = export_itemsets[export_columns].to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download Frequent Itemsets (CSV)",
                    data=csv,
                    file_name="frequent_itemsets.csv",
                    mime="text/csv"
                )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main application
def main():
    # Initialize session state for app navigation
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Introduction"
    
    # Header
    st.markdown("<h1 class='main-header'>🛒 Market Basket Analysis & Product Recommendation System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
    if dark_mode:
        st.markdown("""
        <style>
            .main {background-color: #121212 !important}
            .main-header {color: #4CAF50 !important}
            .sub-header {color: #8BC34A !important}
            .card {background-color: #1E1E1E !important; color: #E0E0E0 !important}
            .metric-card {background-color: #2A2A2A !important; color: #E0E0E0 !important}
            .stMarkdown {color: #E0E0E0 !important}
            .stText {color: #E0E0E0 !important}
            .stMarkdownContainer {color: #E0E0E0 !important}
            .stSelectbox>div>div {background-color: #1E1E1E !important; color: #E0E0E0 !important}
            .stSlider>div>div {background-color: #1E1E1E !important; color: #E0E0E0 !important}
        </style>
        """, unsafe_allow_html=True)
    
    # Algorithm parameters
    st.sidebar.header("Algorithm Parameters")
    default_min_support = st.sidebar.slider(
        "Default Min Support",
        min_value=0.001,
        max_value=0.5,
        value=0.05,
        format="%.3f"
    )
    
    default_min_confidence = st.sidebar.slider(
        "Default Min Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        format="%.2f"
    )
    
    default_min_lift = st.sidebar.slider(
        "Default Min Lift",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        format="%.2f"
    )
    
    # File upload or sample data
    st.sidebar.header("Data Input")
    data_option = st.sidebar.radio(
        "Choose data source:",
        options=["Upload your data", "Use sample data"]
    )
    
    # Initialize variables
    df = None
    transactions_list = None
    formatted_df = None
    frequent_itemsets_df = None
    rules_df = None
    
    # Load data based on selection
    if data_option == "Upload your data":
        uploaded_file = st.sidebar.file_uploader("Upload your transaction data", type=["csv", "xlsx", "xls", "json"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
    else:
        # Add sample size selection
        sample_size = st.sidebar.radio(
            "Choose sample dataset size:",
            options=[100, 500, 1000],
            format_func=lambda x: f"{x} rows"
        )
        df = create_sample_dataset(n_rows=sample_size)
        st.sidebar.info(f"Using generated sample dataset with {sample_size} rows that mimics grocery store transactions.")
    
    # Navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a section:",
        ["Introduction", "Data Exploration", "Market Basket Analysis", "Recommendation System", "Technical Details"],
        index=["Introduction", "Data Exploration", "Market Basket Analysis", "Recommendation System", "Technical Details"].index(st.session_state.app_mode)
    )
    
    # Update session state if changed from sidebar
    if app_mode != st.session_state.app_mode:
        st.session_state.app_mode = app_mode
    
    # Display selected section
    if st.session_state.app_mode == "Introduction":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        # Welcome to the Market Basket Analysis & Product Recommendation System
        
        This application helps you discover relationships between products in transaction data using association rule mining techniques.
        
        ## What you can do:
        - **Explore transaction data** with interactive visualizations
        - **Run market basket analysis** using Apriori and FP-Growth algorithms
        - **Generate association rules** to find product relationships
        - **Build product recommendations** based on discovered patterns
        
        ## Getting Started:
        1. Upload your transaction data or use the sample dataset
        2. Navigate through the sections using the sidebar
        3. Adjust algorithm parameters as needed
        
        ## About the Algorithms:
        - **Apriori**: Identifies frequent itemsets using a bottom-up approach
        - **FP-Growth**: More efficient algorithm using a compact data structure
        - **Association Rules**: Generates "if-then" relationships from frequent itemsets
        
        Ready to discover insights in your transaction data? Select "Data Exploration" in the sidebar to begin!
        """, unsafe_allow_html=True)
        
        # Add a button to start exploring data
        if st.button("Start Data Exploration"):
            st.session_state.app_mode = "Data Exploration"
            st.experimental_rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif df is not None:
        if st.session_state.app_mode == "Data Exploration":
            display_dataset_summary(df)
            analyze_transaction_data(df)
            analyze_product_data(df)
            
            # Use session state values if available, otherwise generate new ones
            if 'transactions_list' in st.session_state and 'formatted_df' in st.session_state:
                transactions_list = st.session_state.transactions_list
                formatted_df = st.session_state.formatted_df
            else:
                transactions_list, formatted_df = prepare_transactions_for_mba(df)
        
        elif st.session_state.app_mode == "Market Basket Analysis":
            # Use session state values if available
            if 'formatted_df' in st.session_state:
                formatted_df = st.session_state.formatted_df
            
            if formatted_df is None:
                st.warning("Please go to 'Data Exploration' section first to prepare the transaction data.")
                if st.button("Go to Data Exploration"):
                    st.session_state.app_mode = "Data Exploration"
                    st.experimental_rerun()
            else:
                # Run Apriori
                frequent_itemsets_df = run_apriori_algorithm(formatted_df, min_support=default_min_support)
                
                # Run FP-Growth
                run_fpgrowth_algorithm(formatted_df, min_support=default_min_support)
                
                # Generate rules
                rules_df = generate_association_rules(
                    frequent_itemsets_df, 
                    min_confidence=default_min_confidence,
                    min_lift=default_min_lift
                )
                
                # Store in session state
                st.session_state.rules_df = rules_df
                st.session_state.frequent_itemsets_df = frequent_itemsets_df
                
                # Export results
                export_results(rules_df, frequent_itemsets_df)
                
                # Add a button to proceed to recommendation system
                if st.button("Proceed to Recommendation System"):
                    st.session_state.app_mode = "Recommendation System"
                    st.experimental_rerun()
        
        elif st.session_state.app_mode == "Recommendation System":
            # Use session state values if available
            if 'formatted_df' in st.session_state:
                formatted_df = st.session_state.formatted_df
            if 'transactions_list' in st.session_state:
                transactions_list = st.session_state.transactions_list
            if 'rules_df' in st.session_state:
                rules_df = st.session_state.rules_df
            
            if formatted_df is None:
                st.warning("Please go to 'Data Exploration' section first to prepare the transaction data.")
                if st.button("Go to Data Exploration"):
                    st.session_state.app_mode = "Data Exploration"
                    st.experimental_rerun()
            elif rules_df is None:
                st.warning("Please run Market Basket Analysis first to generate association rules.")
                if st.button("Go to Market Basket Analysis"):
                    st.session_state.app_mode = "Market Basket Analysis"
                    st.experimental_rerun()
            else:
                build_recommendation_system(rules_df, transactions_list, formatted_df)
        
        elif st.session_state.app_mode == "Technical Details":
            display_technical_details()
    
    else:
        st.warning("Please upload data or select the sample dataset to begin.")
    
    # Footer
    st.markdown("""
    <div class='footer'>
        Market Basket Analysis & Product Recommendation System | Created with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

