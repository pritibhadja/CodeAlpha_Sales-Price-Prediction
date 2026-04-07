# 📊 Sales Price Prediction

A machine learning-based web application that predicts sales revenue based on advertising expenditure across different channels (TV, Radio, and Newspaper). The project features a dynamic dashboard, comprehensive analytics, and an intelligent budget optimizer.

## 🚀 Features

- **Predictive Modeling**: Uses Linear Regression to predict sales based on TV, Radio, and Newspaper advertising budgets.
- **Interactive Dashboard**: View summary statistics and historical sales trends.
- **Deep Analytics**: 
  - Correlation analysis between advertising channels and sales.
  - Spend distribution metrics.
  - Heatmaps showing "local neighbourhood" correlations.
- **Budget Optimizer**: Automatically suggests the most efficient budget allocation for a given total investment to maximize predicted sales.
- **Real-time Feedback**: Get instant predictions and actionable suggestions as you adjust budget values.
- **Exportable Reports**: Download detailed CSV reports comparing actual data with model predictions.

## 📈 Machine Learning Model

The model is built using **Multiple Linear Regression**. 
- **Training Data**: The Advertising dataset containing 200 records of advertising spend on TV, Radio, and Newspaper.
- **Target Variable**: Sales (in thousands of units).
- **Metric**: The coefficients indicate the marginal impact of each channel on total sales.
- Model performance evaluated using **R² Score** and **Mean Squared Error (MSE)**.  
- Trained model saved using **Pickle (.pkl)** for reuse. 

## 🛠️ Technologies Used

### 💻 Frontend
- HTML5  
- CSS3  
- JavaScript
- Chart.js

### ⚙️ Backend
- Python  
- Flask  

### 🗄️ Machine Learning
- Pandas  
- NumPy  
- Scikit-learn (Linear Regression)

### 🧰 Tools
- Jupyter Notebook (`model.ipynb`)  
- Pickle (for model saving)  
- VS Code  

---

## 📊 Data Visualization

- Pie Chart → Shows contribution of each advertising medium  
- Graphs → Show trends and prediction insights  

## 📁 Project Structure

```text
Sales-Price-Prediction/
├── app.py                # Flask application backend (Routes & Logic)
├── model.ipynb           # Model training & EDA notebook
├── model.pkl             # Serialized Linear Regression model
├── advertising.csv       # Dataset (TV, Radio, Newspaper, Sales)
├── templates/            # HTML templates (JinJa2)
│   ├── index.html        # Smart Dashboard Home
│   ├── analytics.html    # Pro Analytics & Heatmaps
│   └── predict.html      # Prediction Tool & Optimizer
├── static/               # CSS and JS assets
└── README.md             # Project documentation
```
