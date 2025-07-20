🛒 Market Basket Analysis & Recommendation SystemAn interactive web application built with Streamlit for performing Market Basket Analysis (MBA) and generating product recommendations. This tool allows users to upload their transactional data, explore it through various visualizations, and apply association rule mining algorithms like Apriori and FP-Growth to uncover hidden patterns in customer purchasing behavior.✨ Features📁 Flexible Data Input: Upload your own transaction data in CSV, XLSX, or JSON formats, or use a built-in sample dataset to explore the app's features.📊 Interactive Data Exploration (EDA):Dataset summaries with key metrics (total transactions, unique products, etc.).Analysis of transaction sizes and product frequencies.Visualizations of product, category, and brand distributions using Plotly charts.Price and rating analysis across different product categories.🧠 Multiple MBA Algorithms:Apriori: A classic algorithm for frequent itemset mining.FP-Growth: A more efficient alternative to Apriori for large datasets.Adjustable parameters (min_support, min_confidence, min_lift) to fine-tune the analysis.🔍 Association Rule Generation:Generates and displays association rules in an intuitive "If-Then" format.Interactive table to sort and filter the generated rules.Visualizations of rule metrics (Confidence vs. Lift) and a network graph of product associations.💡 Product Recommendation Engine:Basket Simulator: Select products and get instant recommendations for other items.Product Similarity: Find products that are frequently purchased together.Co-occurrence heatmaps to visualize product relationships.🎓 Educational Content: In-app explanations of the underlying algorithms and key concepts (Support, Confidence, Lift) with pseudocode and examples.⬇️ Export Results: Download the generated frequent itemsets and association rules as CSV files for further analysis.🎨 Customizable UI: Includes a dark mode toggle for user comfort.📸 Screenshots(Add screenshots of your application here to showcase the UI)Data Exploration DashboardAssociation Rules Network(Image of EDA section)(Image of Network Graph)Recommendation SimulatorTechnical Details View(Image of Reco System)(Image of Algo Explanation)🛠️ Technical StackBackend: PythonWeb Framework: StreamlitData Manipulation: Pandas, NumPyMBA Algorithms: mlxtend (for Apriori, FP-Growth, and association rules)Data Visualization: Plotly, Matplotlib, SeabornNetwork Analysis: NetworkX🚀 Getting StartedPrerequisitesEnsure you have Python 3.8 or higher installed on your system.InstallationClone the repository:git clone [https://github.com/mohitkapoor19/market-basket-analysis.git](https://github.com/mohitkapoor19/market-basket-analysis.git)
cd market-basket-analysis
Create and activate a virtual environment (recommended):# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install the required dependencies:pip install -r requirements.txt
Running the ApplicationOnce the dependencies are installed, you can run the Streamlit application with the following command:streamlit run sample.py
The application will open in your default web browser.📂 File Structuremohitkapoor19-market-basket-analysis/
│
├── .devcontainer/
│   └── devcontainer.json      # Configuration for VS Code Dev Containers
│
├── README.md                  # This README file
├── requirements.txt           # Python dependencies for the project
└── sample.py                  # The main Streamlit application script
Created by Mohit Kapoor.
