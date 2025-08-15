import streamlit as st
import pandas as pd
import time
import numpy as np


# ----------------- MOCK MODULE DEFINITION -----------------
# Defines a placeholder class with mock functions. This class is used as a fallback
# if the actual data processing modules fail to import, allowing the app to run
# in a demonstration mode without crashing.
class MockModule:
    def information(self, df):
        st.write("**Data Shape:**", df.shape)
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())

    def run_preprocessing_workflow(self, df):
        st.write("### ⚙️ Preprocessing Steps")
        st.write("Follow the steps below to prepare your data.")

        # Mocking some preprocessing steps
        if 'target' not in st.session_state:
            st.session_state.target = st.selectbox("Select target variable", options=df.columns)

        if st.button("Complete Preprocessing & Generate Data", type="primary"):
            # Mock data generation for demonstration
            mock_X = pd.DataFrame(np.random.rand(100, 5), columns=[f'F{i}' for i in range(5)])
            mock_y = pd.Series(np.random.randint(0, 2, 100))
            return mock_X, mock_X, mock_y, mock_y
        return None, None, None, None

    def reset_preprocessing_state(self):
        if 'target' in st.session_state:
            del st.session_state['target']

    def traning_testing_evaluating(self, X_train, X_test, y_train, y_test):
        st.write("### 🧠 Model Selection")
        model_choice = st.selectbox(
            "Choose a classification model to train:",
            ("Logistic Regression", "Random Forest", "XGBoost Classifier")
        )

        if st.button(f"Train {model_choice}", type="primary"):
            with st.spinner("Training in progress... Please wait."):
                time.sleep(3) # Simulate training time
                st.success(f"**{model_choice}** trained successfully!")

                st.write("### 📊 Model Performance")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", "92.5%", "1.5%")
                col2.metric("Precision", "94.2%", "-0.8%")
                col3.metric("Recall", "91.8%", "2.1%")

                st.write("#### Classification Report")
                st.code("""
                          precision    recall  f1-score   support
                Class 0       0.91      0.95      0.93       102
                Class 1       0.94      0.90      0.92        98
                ----------------------------------------------------
                accuracy                           0.92       200
                """)


# ----------------- MODULE IMPORT & FALLBACK -----------------
# This try-except block attempts to import the actual data processing modules.
# If an ImportError occurs, it assigns the MockModule to the variables, ensuring
# the app remains functional.
try:
    from classification_process import load_plot_clf as lp
    from classification_process import preprocess_clf as prp
    from classification_process import models_clf as ml
except ImportError:
    st.warning("Custom classification modules not found. Using placeholder modules for demonstration.", icon="⚠️")
    lp = MockModule()
    prp = MockModule()
    ml = MockModule()

# ----------------- PAGE STYLING (CSS) -----------------
# Injects custom CSS to style the Streamlit application for a unique look and feel.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* --- General App Styling --- */
    .stApp {
        background-color: #1E1E1E; /* Dark background */
    }

    body, .stApp, p, label, input, textarea, div[data-testid="stSelectbox"] > div, [data-baseweb="menu"] li {
        font-family: 'Roboto', sans-serif;
        color: #E0E0E0 !important; /* Light grey text for readability */
    }

    /* --- Headers --- */
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

    /* --- Container Styling for Steps --- */
    .step-container {
        background: rgba(40, 40, 40, 0.85);
        backdrop-filter: blur(5px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* --- Button Styling --- */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 2px solid #00C49A;
        background-color: transparent;
        color: #00C49A;
        transition: all 0.3s ease;
        padding: 10px 25px;
        font-weight: bold;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #00C49A;
        color: #1E1E1E;
        transform: scale(1.05);
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #00C49A;
        color: #1E1E1E;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #33FFC7;
        border-color: #33FFC7;
    }

    /* --- Other Widgets --- */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #00C49A;
        background-color: rgba(0, 196, 154, 0.1);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
# Initializes all required session state variables if they don't already exist.
# This ensures the app state is maintained across reruns and user interactions.
def initialize_state():
    if 'df_uploaded' not in st.session_state: st.session_state.df_uploaded = None
    if 'df_processed' not in st.session_state: st.session_state.df_processed = None
    if 'preprocessing_started' not in st.session_state: st.session_state.preprocessing_started = "sleep"
    if "model_training" not in st.session_state: st.session_state.model_training = "sleep"
    if 'X_train' not in st.session_state: st.session_state.X_train = None
    if 'X_test' not in st.session_state: st.session_state.X_test = None
    if 'y_train' not in st.session_state: st.session_state.y_train = None
    if 'y_test' not in st.session_state: st.session_state.y_test = None

initialize_state()

# ----------------- PAGE HEADER & INTRODUCTION -----------------
st.markdown("<h1>🤖 AnalytiBot Classification Engine</h1>", unsafe_allow_html=True)

# An expander to provide users with an explanation of the classification process.
with st.expander("🤔 What is Classification & How Does It Work?", expanded=True):
    st.markdown("""
    <p>Welcome to the classification engine! This is where your data learns to make decisions. Classification is a machine learning task that teaches a model to categorize items into predefined groups.</p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🏎️ The Process")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1. Data & Training**", icon="📊")
        st.write("It all starts with your labeled data. The model studies this historical data to find patterns associated with each category.")
    with c2:
        st.info("**2. Learning Patterns**", icon="🧠")
        st.write("The algorithm analyzes the relationship between input features and their class labels, building a set of rules to form a predictive model.")
    with c3:
        st.info("**3. Making Predictions**", icon="🎯")
        st.write("Once trained, the model can take new, unseen data and predict which class it belongs to based on the patterns it learned.")

# ----------------- MAIN APPLICATION LOGIC -----------------
# This section controls the multi-step workflow of the application,
# using session state to manage the user's progress.

# Step 1: File Uploader. This is the entry point for the user.
if st.session_state.df_uploaded is None:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🏁 Step 1: Upload Your Data")
    st.write("Upload your dataset in **CSV or Excel** format to begin. For best results, ensure your data is clean and well-structured.")
    uploaded_file = st.file_uploader("Drag and drop your file here or click to browse", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            with st.spinner('Reading your file...'):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            # Store the uploaded dataframe in session state to persist it.
            st.session_state.df_uploaded = df
            st.session_state.df_processed = df.copy()
            st.rerun() # Rerun the script to move to the next step.
        except Exception as e:
            st.error(f"❌ Error: The file could not be read. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# This block executes only after a file has been successfully uploaded.
if st.session_state.df_uploaded is not None:

    # Step 2: Data Preview and Inspection.
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🔍 Step 2: Preview & Inspect")
    st.dataframe(st.session_state.df_uploaded.head())
    with st.expander("📊 Show Full Data Information"):
        lp.information(st.session_state.df_uploaded) # Call the information function from the imported module.
    if st.button("🔄 Upload a Different File"):
        # Clear all session state variables to reset the app.
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Data Preprocessing Workflow.
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🛠️ Step 3: Preprocess Data")
    # Manages the state of the preprocessing step (sleep, active, complete).
    if st.session_state.preprocessing_started == "sleep":
        st.write("Ready to clean, transform, and prepare your data for the models?")
        if st.button("▶️ Start Preprocessing", type="primary", key="clf_preprocessing"):
            st.session_state.preprocessing_started = "active"
            st.rerun()
    elif st.session_state.preprocessing_started == "active":
        st.info("You are in the preprocessing stage. Use the tools below to prepare your data.", icon="⚙️")
        X_train, X_test, y_train, y_test = prp.run_preprocessing_workflow(st.session_state.df_processed)
        # If the preprocessing workflow returns data, update the session state.
        if X_train is not None:
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.y_train, st.session_state.y_test = y_train, y_test
            st.session_state.preprocessing_started = "complete"
            st.success("✅ Preprocessing complete! Your data is now ready for training.")
            time.sleep(2)
            st.rerun()
    elif st.session_state.preprocessing_started == "complete":
        st.success("✅ Preprocessing is complete!")
        st.write(f"**Training Data Shape (X_train):** `{st.session_state.X_train.shape}`")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Proceed to Model Training", type="primary", key="clf_start"):
                st.session_state.model_training = "active"
                st.rerun()
        with col2:
            if st.button("↺ Reset Preprocessing", key="clf_reset"):
                prp.reset_preprocessing_state()
                for key in ['X_train', 'X_test', 'y_train', 'y_test']:
                    if key in st.session_state: del st.session_state[key]
                st.session_state.preprocessing_started = "active"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 4: Model Training and Evaluation.
    # This block is only visible when preprocessing is complete and training is active.
    if st.session_state.model_training == "active" and st.session_state.X_train is not None:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("🧠 Step 4: Train & Evaluate Models")
        st.info("It's time to train! Select a model and see how it performs on your data.", icon="🎯")
        st.subheader("Processed Training Data Preview:")
        st.dataframe(st.session_state.X_train.head())
        st.markdown("---")
        # Call the model training function from the imported module.
        ml.traning_testing_evaluating(
            st.session_state.X_train, st.session_state.X_test,
            st.session_state.y_train, st.session_state.y_test
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    # Provides a button to fully reset the application state and start over.
    if st.button("🚪 Exit and Start Over", type="primary", key="clf_exit"):
        st.balloons()
        st.success("### Thanks for using AnalytiBot! 👋 Resetting the app...")
        time.sleep(2)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()