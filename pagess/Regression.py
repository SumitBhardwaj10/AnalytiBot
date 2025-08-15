import streamlit as st
import pandas as pd
import time
import numpy as np


# ----------------- MOCK MODULE DEFINITION -----------------
# Defines a placeholder class for the regression workflow. This class is used as a
# fallback if the actual processing modules cannot be imported, allowing the app
# to run in a demonstration mode without crashing.
class MockRegressionModule:
    def information(self, df):
        st.write("**Data Shape:**", df.shape)
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())

    def run_preprocessing_workflow_reg(self, df):
        st.write("### ⚙️ Regression Preprocessing")
        st.write("Select your target variable and other options to prepare the data.")

        if 'target_reg' not in st.session_state:
            st.session_state.target_reg = st.selectbox("Select Target Variable (Numerical)", options=df.columns,
                                                       key="reg_target_select")

        if st.button("Complete Preprocessing", type="primary", key="reg_complete_processing"):
            # Mock data generation for demonstration purposes
            mock_X = pd.DataFrame(np.random.rand(100, 5), columns=[f'F{i}' for i in range(5)])
            mock_y = pd.Series(np.random.rand(100) * 1000)
            return mock_X, mock_X, mock_y, mock_y
        return None, None, None, None

    def reset_preprocessing_state_reg(self):
        if 'target_reg' in st.session_state:
            del st.session_state['target_reg']

    def traning_testing_evaluating_reg(self, X_train, X_test, y_train, y_test):
        st.write("### 🧠 Model Selection")
        model_choice = st.selectbox(
            "Choose a regression model to train:",
            ("Linear Regression", "Random Forest Regressor", "XGBoost Regressor"),
            key="reg_model_choice"
        )

        if st.button(f"Train {model_choice}", type="primary", key="reg_train_model_btn"):
            with st.spinner("Training in progress..."):
                time.sleep(3) # Simulate model training time
                st.success(f"**{model_choice}** trained successfully!")

                st.write("### 📊 Model Performance")
                col1, col2, col3 = st.columns(3)
                col1.metric("R-squared (R²)", "0.88")
                col2.metric("Mean Absolute Error (MAE)", "15.43")
                col3.metric("Root Mean Squared Error (RMSE)", "21.98")

                st.write("#### Predictions vs. Actuals Plot")
                # Create a placeholder plot with random data for demonstration.
                chart_data = pd.DataFrame(
                    {'Actual': np.random.rand(20) * 100, 'Predicted': np.random.rand(20) * 100}
                )
                st.line_chart(chart_data)


# ----------------- MODULE IMPORT & FALLBACK -----------------
# This try-except block attempts to import the actual regression modules.
# If an ImportError occurs (e.g., modules not found), it assigns the
# MockRegressionModule to the variables, ensuring the app remains functional.
try:
    from regression_process import load_plot_reg as lp
    from regression_process import preprocess_reg as prp
    from regression_process import models_reg as ml
except ImportError:
    st.warning("Custom regression modules not found. Using placeholder modules for demonstration.", icon="⚠️")
    lp = MockRegressionModule()
    prp = MockRegressionModule()
    ml = MockRegressionModule()

# ----------------- PAGE STYLING (CSS) -----------------
# Injects a block of custom CSS to style the application for a consistent and modern UI.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .stApp { background-color: #1E1E1E; }
    body, .stApp, p, label, input, textarea, div[data-testid="stSelectbox"] > div, [data-baseweb="menu"] li {
        font-family: 'Roboto', sans-serif;
        color: #E0E0E0 !important;
    }
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    h1 {
        font-size: 48px !important;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #00C49A, #33FFC7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .step-container {
        background: rgba(40, 40, 40, 0.85);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    div[data-testid="stButton"] > button {
        border-radius: 10px; border: 2px solid #00C49A; background-color: transparent;
        color: #00C49A; transition: all 0.3s ease; padding: 10px 25px; font-weight: bold;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #00C49A; color: #1E1E1E; transform: scale(1.05);
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #00C49A; color: #1E1E1E;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #00C49A; background-color: rgba(0, 196, 154, 0.1);
        padding: 1rem; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
# Initializes all required session state variables for the regression page.
# Using unique keys (e.g., ending in '_reg') prevents conflicts with other pages.
def initialize_state():
    if 'df_uploaded_reg' not in st.session_state: st.session_state.df_uploaded_reg = None
    if 'df_processed_reg' not in st.session_state: st.session_state.df_processed_reg = None
    if 'preprocessing_started_reg' not in st.session_state: st.session_state.preprocessing_started_reg = "sleep"
    if "model_training_reg" not in st.session_state: st.session_state.model_training_reg = "sleep"
    if 'X_train_reg' not in st.session_state: st.session_state.X_train_reg = None
    if 'X_test_reg' not in st.session_state: st.session_state.X_test_reg = None
    if 'y_train_reg' not in st.session_state: st.session_state.y_train_reg = None
    if 'y_test_reg' not in st.session_state: st.session_state.y_test_reg = None

initialize_state()

# ----------------- PAGE HEADER & INTRODUCTION -----------------
st.markdown("<h1>📈 AnalytiBot Regression Engine</h1>", unsafe_allow_html=True)

# An expander provides a brief explanation of regression to the user.
with st.expander("🤔 What is Regression & How Does It Work?", expanded=True):
    st.markdown(
        "<p>Welcome to the regression engine! This is where you can predict continuous numerical values like prices, sales, or temperatures.</p>",
        unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("⚙️ The Process")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1. Provide Data**", icon="💾")
        st.write("The model learns from your historical data, finding the mathematical relationship between features and the numerical outcome.")
    with c2:
        st.info("**2. Learn Patterns**", icon="🧠")
        st.write("An algorithm like Linear Regression or XGBoost fits a model to the data, ready to generalize to new inputs.")
    with c3:
        st.info("**3. Make Predictions**", icon="🎯")
        st.write("The trained model can now predict a numerical value for any new data point you provide.")

# ----------------- MAIN APPLICATION LOGIC -----------------
# This section controls the multi-step workflow of the regression page.

# Step 1: File Uploader. This is the initial view of the page.
if st.session_state.df_uploaded_reg is None:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🏁 Step 1: Upload Your Data")
    st.write("Upload your dataset in **CSV or Excel** format. Ensure it contains numerical features and a numerical target column for prediction.")
    uploaded_file = st.file_uploader("Upload your regression dataset", type=['csv', 'xlsx', 'xls'], key="reg_uploader", label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            with st.spinner('Reading file...'):
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            # Store the uploaded dataframe in session state to persist it.
            st.session_state.df_uploaded_reg = df
            st.session_state.df_processed_reg = df.copy()
            st.rerun() # Rerun the script to advance to the next step.
        except Exception as e:
            st.error(f"Error reading file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# This block executes only after a file has been successfully uploaded.
if st.session_state.df_uploaded_reg is not None:
    # Step 2: Data Preview and Inspection.
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🔍 Step 2: Preview & Inspect")
    st.dataframe(st.session_state.df_uploaded_reg.head())
    with st.expander("📊 Show Full Data Information"):
        lp.information(st.session_state.df_uploaded_reg)
    if st.button("🔄 Upload a Different File", key="reg_new_upload"):
        # Clear only the regression-related keys from session state to reset the page.
        reg_keys = [key for key in st.session_state.keys() if key.endswith('_reg')]
        for key in reg_keys:
            del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Data Preprocessing Workflow.
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🛠️ Step 3: Preprocess Data")
    # Manages the state of the preprocessing step (sleep, active, complete).
    if st.session_state.preprocessing_started_reg == "sleep":
        st.write("Ready to prepare your data for modeling?")
        if st.button("▶️ Start Preprocessing", type="primary", key="reg_start_processing"):
            st.session_state.preprocessing_started_reg = "active"
            st.rerun()
    elif st.session_state.preprocessing_started_reg == "active":
        st.info("You are in the preprocessing stage. Use the tools below.", icon="⚙️")
        X_train, X_test, y_train, y_test = prp.run_preprocessing_workflow_reg(st.session_state.df_processed_reg)
        # If the preprocessing workflow returns data, update the session state.
        if X_train is not None:
            st.session_state.X_train_reg, st.session_state.X_test_reg = X_train, X_test
            st.session_state.y_train_reg, st.session_state.y_test_reg = y_train, y_test
            st.session_state.preprocessing_started_reg = "complete"
            st.success("✅ Preprocessing complete!")
            time.sleep(2)
            st.rerun()
    elif st.session_state.preprocessing_started_reg == "complete":
        st.success("✅ Preprocessing is complete and the data is ready!")
        st.write("Final Processed Training Data Shape:", st.session_state.X_train_reg.shape)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Proceed to Model Training", type="primary", key="reg_proceed_training"):
                st.session_state.model_training_reg = "active"
                st.rerun()
        with col2:
            if st.button("↺ Reset Preprocessing", key="reg_reset_processing"):
                prp.reset_preprocessing_state_reg()
                for key in ['X_train_reg', 'X_test_reg', 'y_train_reg', 'y_test_reg']:
                    if key in st.session_state: del st.session_state[key]
                st.session_state.preprocessing_started_reg = "active"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 4: Model Training and Evaluation.
    # This block is only visible when preprocessing is complete and training is active.
    if st.session_state.model_training_reg == "active" and st.session_state.X_train_reg is not None:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("🧠 Step 4: Train & Evaluate Models")
        st.info("Select a regression model and see how well it predicts your target.", icon="🎯")
        st.subheader("Processed Training Data Preview:")
        st.dataframe(st.session_state.X_train_reg.head())
        st.markdown("---")
        # Call the model training function from the imported module.
        ml.traning_testing_evaluating_reg(st.session_state.X_train_reg, st.session_state.X_test_reg,
                                          st.session_state.y_train_reg, st.session_state.y_test_reg)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    # Provides a button to fully reset the regression page state and start over.
    if st.button("🚪 Exit and Start Over", type="primary", key="reg_reset_app"):
        st.balloons()
        st.success("### Thanks for using AnalytiBot! 👋 Resetting page...")
        time.sleep(2)
        reg_keys = [key for key in st.session_state.keys() if key.endswith('_reg')]
        for key in reg_keys:
            del st.session_state[key]
        st.rerun()