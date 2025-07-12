import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import random

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Dashboard", 
    page_icon="📊", 
    layout="wide", 
    initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_credit_risk_dataset(sample_size=1000):
    np.random.seed(42)
    random.seed(42)

    data = {
        'customer_id': [f'CUST-{i:06d}' for i in range(1, sample_size + 1)],
        'age': np.random.normal(40, 12, sample_size).astype(int), # mean=40, std=12
        'annual_income': np.random.lognormal(10.5, 0.5, sample_size).astype(int), # mean=10.5, std=0.5
        'employment_years': np.random.exponential(5, sample_size).astype(int),
        'loan_amount': np.random.lognormal(9.5, 0.8, sample_size).astype(int),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 72], sample_size),
        'credit_score': np.random.normal(650, 100, sample_size).astype(int),
        'debt_to_income_ratio': np.random.beta(2, 5, sample_size) * 100,
        'number_of_dependents': np.random.poisson(1.5, sample_size),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], sample_size, p=[0.3, 0.5, 0.15, 0.05]),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Contract'], sample_size, p=[0.7, 0.1, 0.15, 0.05]),
        'home_ownership': np.random.choice(['Own', 'Rent', 'Mortgage'], sample_size, p=[0.4, 0.3, 0.3]),
        'loan_purpose': np.random.choice(['Personal', 'Auto', 'Home', 'Business', 'Education'], sample_size, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'has_cosigner': np.random.choice([True, False], sample_size, p=[0.2, 0.8]),
        'previous_defaults': np.random.poisson(0.3, sample_size),
        'application_date': pd.date_range(start='2020-01-01', end='2024-01-01', periods=sample_size)
    }

    df = pd.DataFrame(data)

    # Realistic Ranges
    df['age'] = np.clip(df['age'], 18, 80)
    df['annual_income'] = np.clip(df['annual_income'], 20000, 500000)
    df['employment_years'] = np.clip(df['employment_years'], 0, 40)
    df['loan_amount'] = np.clip(df['loan_amount'], 1000, 100000)
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    df['debt_to_income_ratio'] = np.clip(df['debt_to_income_ratio'], 0, 80)
    df['number_of_dependents'] = np.clip(df['number_of_dependents'], 0, 8)
    df['previous_defaults'] = np.clip(df['previous_defaults'], 0, 5)

    # Risk Score Calculation
    risk_score = (
        (850 - df['credit_score']) / 550 * 30 +  # Credit score impact
        df['debt_to_income_ratio'] / 100 * 25 +  # DTI impact
        df['previous_defaults'] * 10 +  # Previous defaults
        np.where(df['annual_income'] < 30000, 15, 0) +  # Low income penalty
        np.where(df['employment_years'] < 2, 10, 0) +  # Employment stability
        np.where(df['has_cosigner'], -5, 0)  # Cosigner benefit
    )

    # Default Probability and Actual Default
    default_probability = 1 / (1 + np.exp(-(risk_score - 50) / 10))
    df['default_probability'] = default_probability
    df['is_default'] = np.random.binomial(1, default_probability, sample_size).astype(bool)

    # Risk Categories
    df['risk_category'] = pd.cut(df['default_probability'], 
                            bins=[0, 0.1, 0.3, 0.7, 1.0], 
                            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
    
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Credit Risk Data')
    return output.getvalue()

def main():
    st.title("Credit Risk Dashboard")
    st.markdown("Comprehensive analysis of credit risk portfolio with interactive visualizations and insights")

    # Sidebar
    st.sidebar.header("Dashboard Controls")

    # Dataset Size Slider
    sample_size = st.sidebar.slider("Dataset Size", min_value=100, max_value=5000, value=1000, step=100)

    # Load the Data
    df = generate_credit_risk_dataset(sample_size)

    # Sidebar Filters
    st.sidebar.header("Filters")

    # Age Range Filter
    age_range = st.sidebar.slider("Age Range", 
                                 min_value=int(df['age'].min()), 
                                 max_value=int(df['age'].max()), 
                                 value=(int(df['age'].min()), int(df['age'].max())))
    
    # Income Range Filter
    income_range = st.sidebar.slider("Annual Income Range", 
                                   min_value=int(df['annual_income'].min()), 
                                   max_value=int(df['annual_income'].max()), 
                                   value=(int(df['annual_income'].min()), int(df['annual_income'].max())),
                                   format="$%d")
    
    # Risk Category Multiselect
    risk_categories = st.sidebar.multiselect("Risk Categories", 
                                           options=df['risk_category'].cat.categories.tolist(),
                                           default=df['risk_category'].cat.categories.tolist())
    # Loan Purpose Multiselect
    loan_purposes = st.sidebar.multiselect("Loan Purpose", 
                                         options=df['loan_purpose'].unique().tolist(),
                                         default=df['loan_purpose'].unique().tolist())
    
    # Employment Type Multiselect
    employment_types = st.sidebar.multiselect("Employment Type", 
                                            options=df['employment_type'].unique().tolist(),
                                            default=df['employment_type'].unique().tolist())
    
    # Apply Filters
    filtered_df = df[
            (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
            (df['annual_income'] >= income_range[0]) & (df['annual_income'] <= income_range[1]) &
            (df['risk_category'].isin(risk_categories)) &
            (df['loan_purpose'].isin(loan_purposes)) &
            (df['employment_type'].isin(employment_types))
        ]   
    
    # Export Section
    st.sidebar.header("Export Data")

    # CSV Export
    csv_data = convert_df_to_csv(filtered_df)
    st.sidebar.download_button(
        label="📄 Download CSV",
        data=csv_data,
        file_name="credit_risk_data.csv",  # No date/time in the file name
        mime="text/csv"
    )

    # Excel Export
    excel_data = convert_df_to_excel(filtered_df)
    st.sidebar.download_button(
        label="📊 Download Excel",
        data=excel_data,
        file_name=f"credit_risk_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Risk Analysis", "Portfolio Stats", "Data Explorer", "Data Summary"])

    with tab1:
        st.header("Portfolio Overview")

        # KPIs
        col1, col2, col3, col4, col5 = st.columns(5)

        # Total Customers KPI
        with col1:
            total_customers = len(filtered_df)
            total_customers_delta = total_customers - len(df)
            st.metric(
                label="Total Customers",
                value=f"{total_customers:,}",
                delta=f"{total_customers_delta}" if total_customers_delta != 0 else None,
            )

        # Avg Loan Amount KPI
        with col2:
            avg_loan = filtered_df['loan_amount'].mean()
            avg_loan_delta = ((avg_loan / df['loan_amount'].mean()) - 1) * 100
            st.metric(
                label="Avg Loan Amount",
                value=f"${avg_loan:,.0f}",
                delta=f"{avg_loan_delta:.1f}%" if len(filtered_df) != len(df) else None,
            )

        # Default Rate KPI
        with col3:
            default_rate = filtered_df['is_default'].mean() * 100
            default_rate_delta = default_rate - (df['is_default'].mean() * 100)
            st.metric(
                label="Default Rate",
                value=f"{default_rate:.1f}%",
                delta=f"{default_rate_delta:.1f}%" if len(filtered_df) != len(df) else None,
                delta_color="inverse"
            )

        # Avg Credit Score KPI
        with col4:
            avg_credit_score = filtered_df['credit_score'].mean()
            avg_credit_score_delta = avg_credit_score - df['credit_score'].mean()
            st.metric(
                label="Avg Credit Score",
                value=f"{avg_credit_score:.0f}",
                delta=f"{avg_credit_score_delta:.0f}" if len(filtered_df) != len(df) else None,
            )

        # Total Exposure KPI
        with col5:
            total_exposure = filtered_df['loan_amount'].sum()
            total_exposure_delta = ((total_exposure / df['loan_amount'].sum()) - 1) * 100
            st.metric(
                label="Total Exposure",
                value=f"${total_exposure:,.0f}",
                delta=f"{total_exposure_delta:.1f}%" if len(filtered_df) != len(df) else None,
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Distribution
            risk_dist = filtered_df['risk_category'].value_counts()
            fig_risk = px.pie(values=risk_dist.values, names=risk_dist.index, 
                            title="Risk Distribution",
                            color_discrete_map={'Low Risk': '#2E8B57', 'Medium Risk': '#FFD700', 
                                              'High Risk': '#FF6347', 'Very High Risk': '#DC143C'})
            st.plotly_chart(fig_risk, use_container_width=True)     

        with col2:
            # Loan Purpose Distribution
            purpose_dist = filtered_df['loan_purpose'].value_counts()
            fig_purpose = px.bar(x=purpose_dist.index, y=purpose_dist.values,
                               title="Loan Purpose Distribution",
                               color=purpose_dist.values,
                               color_continuous_scale='viridis')
            fig_purpose.update_layout(showlegend=False)
            st.plotly_chart(fig_purpose, use_container_width=True)

        # Time Series Analysis
        monthly_data = filtered_df.groupby(filtered_df['application_date'].dt.to_period('M')).agg({
            'loan_amount': ['sum', 'count'],
            'is_default': 'mean'
        }).reset_index()
        
        monthly_data.columns = ['month', 'total_amount', 'loan_count', 'default_rate']
        monthly_data['month'] = monthly_data['month'].astype(str)
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['loan_count'],
                                    mode='lines+markers', name='Loan Count', yaxis='y'))
        fig_time.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['default_rate']*100,
                                    mode='lines+markers', name='Default Rate (%)', yaxis='y2'))
        
        fig_time.update_layout(
            title='Monthly Loan Origination and Default Trends',
            xaxis_title='Month',
            yaxis=dict(title='Loan Count', side='left'),
            yaxis2=dict(title='Default Rate (%)', side='right', overlaying='y'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_time, use_container_width=True)  

    with tab2:
        st.header("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Credit Score vs Default Rate
            score_bins = pd.cut(filtered_df['credit_score'], bins=10)
            score_analysis = filtered_df.groupby(score_bins)['is_default'].mean().reset_index()
            score_analysis['credit_score_range'] = score_analysis['credit_score'].astype(str)
            
            fig_score = px.bar(score_analysis, x='credit_score_range', y='is_default',
                             title='Default Rate by Credit Score Range',
                             labels={'is_default': 'Default Rate', 'credit_score_range': 'Credit Score Range'})
            fig_score.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_score, use_container_width=True)
        
        with col2:
            # Income vs Default Rate
            income_bins = pd.cut(filtered_df['annual_income'], bins=8)
            income_analysis = filtered_df.groupby(income_bins)['is_default'].mean().reset_index()
            income_analysis['income_range'] = income_analysis['annual_income'].astype(str)
            
            fig_income = px.bar(income_analysis, x='income_range', y='is_default',
                              title='Default Rate by Income Range',
                              labels={'is_default': 'Default Rate', 'income_range': 'Income Range'})
            fig_income.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_income, use_container_width=True)
        
        # Correlation Heatmap
        numeric_cols = ['age', 'annual_income', 'employment_years', 'loan_amount', 
                       'credit_score', 'debt_to_income_ratio', 'number_of_dependents', 
                       'previous_defaults', 'default_probability']
        
        corr_matrix = filtered_df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Correlation Matrix of Risk Factors")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Risk Factors Importance
        st.subheader("Key Risk Factors")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**High Risk Indicators:**")
            st.write("• Credit Score < 600")
            st.write("• Debt-to-Income > 40%")
            st.write("• Previous Defaults > 1")
            st.write("• Employment < 2 years")
        
        with col2:
            st.markdown("**Medium Risk Indicators:**")
            st.write("• Credit Score 600-700")
            st.write("• Debt-to-Income 25-40%")
            st.write("• Income < $30K")
            st.write("• Multiple dependents")
        
        with col3:
            st.markdown("**Low Risk Indicators:**")
            st.write("• Credit Score > 700")
            st.write("• Debt-to-Income < 25%")
            st.write("• Stable employment")
            st.write("• Has cosigner")

    with tab3:
        st.header("Portfolio Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loan Amount Distribution
            fig_loan_dist = px.histogram(filtered_df, x='loan_amount', nbins=30,
                                       title='Loan Amount Distribution',
                                       marginal='box')
            st.plotly_chart(fig_loan_dist, use_container_width=True)
        
        with col2:
            # Age Distribution by Risk
            fig_age_risk = px.violin(filtered_df, x='risk_category', y='age',
                                   title='Age Distribution by Risk Category',
                                   box=True)
            st.plotly_chart(fig_age_risk, use_container_width=True)
        
        # Employment analysis
        employment_stats = filtered_df.groupby('employment_type').agg({
            'loan_amount': ['mean', 'sum', 'count'],
            'is_default': 'mean',
            'credit_score': 'mean'
        }).round(2)
        
        employment_stats.columns = ['Avg Loan Amount', 'Total Exposure', 'Count', 'Default Rate', 'Avg Credit Score']
        
        st.subheader("Employment Type Analysis")
        st.dataframe(employment_stats, use_container_width=True)
        
        # Home Ownership Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            home_default = filtered_df.groupby('home_ownership')['is_default'].mean().reset_index()
            fig_home = px.bar(home_default, x='home_ownership', y='is_default',
                            title='Default Rate by Home Ownership',
                            color='is_default', color_continuous_scale='Reds')
            st.plotly_chart(fig_home, use_container_width=True)
        
        with col2:
            education_default = filtered_df.groupby('education_level')['is_default'].mean().reset_index()
            fig_edu = px.bar(education_default, x='education_level', y='is_default',
                           title='Default Rate by Education Level',
                           color='is_default', color_continuous_scale='Reds')
            st.plotly_chart(fig_edu, use_container_width=True)

    with tab4:
        st.header("Data Explorer")
        
        # Interactive Scatter Plot
        st.subheader("Interactive Scatter Plot")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis", options=['age', 'annual_income', 'credit_score', 'loan_amount', 'debt_to_income_ratio'])
        
        with col2:
            y_axis = st.selectbox("Y-axis", options=['default_probability', 'loan_amount', 'credit_score', 'debt_to_income_ratio', 'employment_years'])
        
        with col3:
            color_by = st.selectbox("Color by", options=['risk_category', 'loan_purpose', 'employment_type', 'home_ownership', 'is_default'])
        
        fig_scatter = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by,
                               title=f'{y_axis.title()} vs {x_axis.title()}',
                               hover_data=['customer_id', 'loan_amount', 'credit_score'])
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Data Preview
        st.subheader("Data Preview")
        
        # Show/hide Columns
        all_columns = filtered_df.columns.tolist()
        selected_columns = st.multiselect("Select columns to display", 
                                        options=all_columns, 
                                        default=all_columns[:8])
        
        if selected_columns:
            st.dataframe(filtered_df[selected_columns].head(100), use_container_width=True)
        
        # Search Functionality
        st.subheader("Search Customer")
        customer_search = st.text_input("Enter Customer ID")
        
        if customer_search:
            customer_data = filtered_df[filtered_df['customer_id'].str.contains(customer_search, case=False)]
            if not customer_data.empty:
                st.dataframe(customer_data, use_container_width=True)
            else:
                st.warning("No customer found with that ID")

    with tab5:
        st.header("Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Total Records:** {len(filtered_df):,}")
            st.write(f"**Total Columns:** {len(filtered_df.columns)}")
            st.write(f"**Missing Values:** {filtered_df.isnull().sum().sum()}")
            st.write(f"**Memory Usage:** {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with col2:
            st.subheader("Data Types")
            data_types = filtered_df.dtypes.value_counts()
            for dtype, count in data_types.items():
                st.write(f"**{dtype}:** {count} columns")
        
        # Statistical Summary
        st.subheader("Statistical Summary")
        
        # Select Numeric Columns for Summary
        numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_columns:
            summary_stats = filtered_df[numeric_columns].describe()
            st.dataframe(summary_stats, use_container_width=True)
        
        # Categorical Summary
        st.subheader("Categorical Variables Summary")
        
        categorical_columns = filtered_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if categorical_columns:
            cat_summary = []
            for col in categorical_columns:
                unique_count = filtered_df[col].nunique()
                most_frequent = filtered_df[col].mode().iloc[0] if not filtered_df[col].empty else 'N/A'
                most_frequent_count = filtered_df[col].value_counts().iloc[0] if not filtered_df[col].empty else 0
                
                cat_summary.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Most Frequent': most_frequent,
                    'Frequency': most_frequent_count,
                    'Percentage': f"{(most_frequent_count / len(filtered_df)) * 100:.1f}%"
                })
            
            cat_summary_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_summary_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "📊 **Credit Risk Dashboard** | "
        f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Showing {len(filtered_df):,} of {len(df):,} records"
    )

if __name__ == "__main__":
    main()