import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Helper function to cache data loading ---
@st.cache_data
def load_data(uploaded_file):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- Helper function to convert DataFrame to CSV for download ---
@st.cache_data
def convert_df_to_csv(df_to_convert):
    """Converts a DataFrame to a CSV string for downloading."""
    return df_to_convert.to_csv(index=True).encode('utf-8')

# --- Page Configuration ---
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("Data Analysis and Visualization Dashboard")

# --- 1. Dataset Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
else:
    df = load_data(uploaded_file)

    if df is not None:
        # Define column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # --- Create Tabs for Navigation ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Summary", 
            "Missing Data Analysis", 
            "Data Visualization", 
            "Download Report"
        ])

        # --- Tab 1: Data Summary ---
        with tab1:
            st.header("1. Dataset Preview")
            st.dataframe(df.head())

            st.header("2. Data Summary")
            col1, col2 = st.columns(2)
            col1.metric("Total Rows", df.shape[0])
            col2.metric("Total Columns", df.shape[1])

            st.subheader("Column Data Types & Non-Nulls")
            # Capture df.info() output
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)

            st.subheader("Numerical Statistics")
            if not numerical_cols.empty:
                st.dataframe(df.describe(include=[np.number]))
            else:
                st.warning("No numerical columns found.")

            st.subheader("Categorical Statistics")
            if not categorical_cols.empty:
                try:
                    st.dataframe(df.describe(include=['object']))
                except ValueError:
                    st.warning("No categorical columns to describe.")
            else:
                st.warning("No categorical columns found.")

        # --- Tab 2: Missing Data Handling ---
        with tab2:
            st.header("3. Missing Data Handling")
            st.subheader("Missing Data Report")
            missing_report = df.isnull().sum().to_frame('Missing Count')
            missing_report['Missing (%)'] = (missing_report['Missing Count'] / len(df)) * 100
            st.dataframe(missing_report[missing_report['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))

            st.subheader("Handling Option 1: Dropping Rows (Example)")
            df_dropped = df.dropna()
            st.text(f"Original rows: {len(df)} \nRows after drop: {len(df_dropped)}")

            st.subheader("Handling Option 2: Filling with Mean (Example)")
            df_filled = df.copy()
            if not numerical_cols.empty:
                df_filled[numerical_cols] = df_filled[numerical_cols].fillna(df_filled[numerical_cols].mean())
                st.text("Missing values count after filling (numerical columns only):")
                st.dataframe(df_filled.isnull().sum().to_frame('Missing Count After Fill'))
            else:
                st.warning("No numerical columns to fill with mean.")

        # --- Tab 3: Data Visualization ---
        with tab3:
            st.header("4. Data Visualization")

            if st.checkbox("Show Visualization Plots", True):
                
                st.subheader("Histogram")
                if not numerical_cols.empty:
                    fig_hist, ax_hist = plt.subplots()
                    sns.histplot(df[numerical_cols[0]], kde=True, bins=30, ax=ax_hist)
                    ax_hist.set_title(f"Distribution of {numerical_cols[0]}")
                    st.pyplot(fig_hist)
                else:
                    st.warning("No numerical columns for Histogram.")

                st.subheader("Bar Chart")
                if not categorical_cols.empty:
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                    sns.countplot(x=categorical_cols[0], data=df, order=df[categorical_cols[0]].value_counts().index, ax=ax_bar)
                    ax_bar.set_title(f"Frequency of {categorical_cols[0]}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_bar)
                else:
                    st.warning("No categorical columns for Bar Chart.")

                st.subheader("Box Plot")
                if not numerical_cols.empty:
                    fig_box, ax_box = plt.subplots(figsize=(10, 5))
                    sns.boxplot(x=df[numerical_cols[0]], ax=ax_box)
                    ax_box.set_title(f"Box Plot for {numerical_cols[0]}")
                    st.pyplot(fig_box)
                else:
                    st.warning("No numerical columns for Box Plot.")

                st.subheader("Scatter Plot")
                if len(numerical_cols) >= 2:
                    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=numerical_cols[0], y=numerical_cols[1], data=df, ax=ax_scatter)
                    ax_scatter.set_title(f"Relationship between {numerical_cols[0]} and {numerical_cols[1]}")
                    st.pyplot(fig_scatter)
                else:
                    st.warning("At least two numerical columns required for Scatter Plot.")

                st.subheader("Correlation Heatmap")
                if len(numerical_cols) > 1:
                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
                    corr = df[numerical_cols].corr(numeric_only=True)
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax_heatmap)
                    ax_heatmap.set_title("Correlation Matrix")
                    st.pyplot(fig_heatmap)
                else:
                    st.warning("At least two numerical columns required for Heatmap.")

                st.subheader("Pairplot")
                if len(numerical_cols) > 1:
                    st.text("Plotting Pairplot (this may take a moment)...")
                    fig_pairplot = sns.pairplot(df[numerical_cols])
                    fig_pairplot.fig.suptitle("Pairplot of Numerical Columns", y=1.02)
                    st.pyplot(fig_pairplot)
                else:
                    st.warning("At least two numerical columns required for Pairplot.")

        # --- Tab 4: Download Summary ---
        with tab4:
            st.header("5. Download Summary")
            st.text("Click the button below to download the full summary statistics report.")
            
            try:
                summary_report = df.describe(include='all').transpose()
                csv_data = convert_df_to_csv(summary_report)
                
                st.download_button(
                    label="Download Summary Report as CSV",
                    data=csv_data,
                    file_name="data_summary_report.csv",
                    mime="text/csv",
                )
                
                st.subheader("Report Preview:")
                st.dataframe(summary_report)
                
            except Exception as e:
                st.error(f"Error generating report: {e}")


