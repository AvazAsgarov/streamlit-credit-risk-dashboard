# Credit Risk Dashboard

A comprehensive **Credit Risk Dashboard** built with Streamlit, Pandas, NumPy, and Plotly. This interactive web application generates a synthetic dataset to simulate credit risk analysis, providing insights into customer credit profiles, loan portfolios, and risk factors through visualizations and interactive controls.

Explore the live dashboard here.

## Table of Contents

- Overview
- Features
- Screenshots
- Installation
- Usage
- Dataset
- Contributing
- License

## Overview

The Credit Risk Dashboard is designed to help financial analysts and risk managers explore and analyze credit risk data. It generates a synthetic dataset of customer loan applications and provides interactive visualizations, filters, and data export capabilities. The dashboard is divided into five tabs: Overview, Risk Analysis, Portfolio Statistics, Data Explorer, and Data Summary, each offering unique insights into the credit portfolio.

## Features

- **Interactive Visualizations**: Includes pie charts, bar charts, histograms, violin plots, scatter plots, and a correlation heatmap powered by Plotly.
- **Dynamic Filters**: Filter data by age, income, risk category, loan purpose, and employment type using sidebar controls.
- **Key Performance Indicators (KPIs)**: Displays metrics like total customers, average loan amount, default rate, average credit score, and total exposure.
- **Data Export**: Download filtered data as CSV or Excel files.
- **Data Explorer**: Interactive scatter plots with customizable axes and customer ID search functionality.
- **Comprehensive Summaries**: Statistical summaries for numeric and categorical variables, along with dataset metadata.

## Screenshots

Below are screenshots of the Credit Risk Dashboard:

### Overview Tab
<img width="1751" height="747" alt="image" src="https://github.com/user-attachments/assets/212a4e50-ecf2-4818-92df-516a2c807ccd" />

![Overview Tab](screenshots/overview_tab.png)*Displays KPIs, risk distribution, loan purpose distribution, and monthly trends.*

### Risk Analysis Tab

![Risk Analysis Tab](screenshots/risk_analysis_tab.png)*Shows default rates by credit score and income, correlation heatmap, and risk indicators.*

### Portfolio Statistics Tab

![Portfolio Statistics Tab](screenshots/portfolio_stats_tab.png)*Includes loan amount distribution, age distribution by risk, and analyses by employment and home ownership.*

### Data Explorer Tab

![Data Explorer Tab](screenshots/data_explorer_tab.png)*Features interactive scatter plots and data preview with column selection.*

### Data Summary Tab

![Data Summary Tab](screenshots/data_summary_tab.png)*Provides dataset information, data types, and statistical summaries.*

*Note*: Replace the placeholder screenshot paths (`screenshots/*.png`) with actual screenshot files in your repository's `screenshots/` folder.

## Installation

To run the dashboard locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Dependencies**: Ensure you have Python 3.8+ installed, then install the required packages:

   ```bash
   pip install streamlit pandas numpy plotly openpyxl
   ```

3. **Run the Application**: Launch the Streamlit app:

   ```bash
   streamlit run credit_risk_dashboard.py
   ```

   The dashboard will open in your default web browser.

## Usage

1. **Access the Live App**: Visit https://credit-risk-dashboard-avaz-asgarov.streamlit.app/ to explore the dashboard online.
2. **Adjust Dataset Size**: Use the sidebar slider to set the number of records (100–5000).
3. **Apply Filters**: Use sidebar controls to filter by age, income, risk category, loan purpose, or employment type.
4. **Explore Tabs**:
   - **Overview**: View KPIs and high-level portfolio metrics.
   - **Risk Analysis**: Analyze default rates and risk factor correlations.
   - **Portfolio Statistics**: Explore distributions and segment analyses.
   - **Data Explorer**: Create custom scatter plots and search for customers.
   - **Data Summary**: Review dataset statistics and variable summaries.
5. **Export Data**: Download filtered data as CSV or Excel files from the sidebar.

## Dataset

The dashboard generates a synthetic dataset with the following features:

- **Customer Attributes**: Customer ID, age, annual income, employment years, number of dependents.
- **Loan Details**: Loan amount, term, purpose, home ownership, cosigner status.
- **Risk Metrics**: Credit score, debt-to-income ratio, previous defaults, default probability, risk category.
- **Temporal Data**: Application date (from 2020 to 2024).

The dataset is generated using realistic statistical distributions (e.g., normal for age, lognormal for income) and includes a calculated risk score and default probability. Data is clipped to ensure realistic ranges (e.g., age 18–80, credit score 300–850).
