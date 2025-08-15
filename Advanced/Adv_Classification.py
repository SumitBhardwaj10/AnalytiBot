import time
import streamlit as st
import pandas as pd
import numpy as np
from Advanced import load_plot_adv_clf as lp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from itertools import combinations
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform,randint



st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    /* --- General App Styling --- */
    .stApp { background-color: #1E1E1E; }
    body, .stApp, p, label, input, textarea, div[data-testid="stSelectbox"] > div, [data-baseweb="menu"] li {
        font-family: 'Roboto', sans-serif;
        color: #E0E0E0 !important;
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
        border-radius: 10px; border: 2px solid #00C49A; background-color: transparent;
        color: #00C49A; transition: all 0.3s ease; padding: 10px 25px; font-weight: bold;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #00C49A; color: #1E1E1E; transform: scale(1.05);
    }
    div[data-testid="stButton"] > button[kind="primary"] { background-color: #00C49A; color: #1E1E1E; }
    div[data-testid="stButton"] > button[kind="primary"]:hover { background-color: #33FFC7; border-color: #33FFC7; }
    /* --- Other Widgets --- */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #00C49A; background-color: rgba(0, 196, 154, 0.1);
        padding: 1rem; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_state():
    """Initializes all necessary session state variables."""
    # State for dataframes
    if 'df_uploaded_adv_clf' not in st.session_state: st.session_state.df_uploaded_adv_clf = None
    if 'df_processed_adv_clf' not in st.session_state: st.session_state.df_processed_adv_clf = None
    if 'df_target_adv_clf' not in st.session_state: st.session_state.df_target_adv_clf = None

    # State for split data
    if 'X_train_adv_clf' not in st.session_state: st.session_state.X_train_adv_clf = None
    if 'X_test_adv_clf' not in st.session_state: st.session_state.X_test_adv_clf = None
    if 'y_train_adv_clf' not in st.session_state: st.session_state.y_train_adv_clf = None
    if 'y_test_adv_clf' not in st.session_state: st.session_state.y_test_adv_clf = None

    # State for the three feature sets
    if 'X_train1_adv_clf' not in st.session_state: st.session_state.X_train1_adv_clf = None
    if 'X_train2_adv_clf' not in st.session_state: st.session_state.X_train2_adv_clf = None
    if 'X_train3_adv_clf' not in st.session_state: st.session_state.X_train3_adv_clf = None
    if 'X_test1_adv_clf' not in st.session_state: st.session_state.X_test1_adv_clf = None
    if 'X_test2_adv_clf' not in st.session_state: st.session_state.X_test2_adv_clf = None
    if 'X_test3_adv_clf' not in st.session_state: st.session_state.X_test3_adv_clf = None

    # State for column names and preprocessors
    if "n_name1_adv_clf" not in st.session_state: st.session_state.n_name1_adv_clf = None
    if "c_name1_adv_clf" not in st.session_state: st.session_state.c_name1_adv_clf = None
    if "label_name_adv_clf" not in st.session_state: st.session_state.label_name_adv_clf = None
    if "imputer_adv_clf" not in st.session_state: st.session_state.imputer_adv_clf = None

    # State for workflow control
    if 'pointer_adv_clf' not in st.session_state: st.session_state.pointer_adv_clf = 0
    flags = ["split_adv_clf", "split_data_viewer_adv_clf", "miss_adv_clf", "feature_adv_clf", "encoding_adv_clf"
             ,"normal_adv_clf","normal_model_adv_clf","normal_result_adv_clf","normal_corr_adv_clf",
             "tuned_adv_clf","tuned_model_adv_clf","normal_result_adv_clf","normal_corr_adv_clf",
             "voting_adv_clf","voting_model_adv_clf","voting_result_adv_clf","voting_corr_adv_clf",
             "stacking_adv_clf","stacking_adv_clf"]
    for flag in flags:
        if flag not in st.session_state:
            st.session_state[flag] = "sleep"


initialize_state()


# Core Logic Functions

def splitter(data):
    """Splits data into training and testing sets."""
    X = data
    y = st.session_state.df_target_adv_clf
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.session_state.X_train_adv_clf = X_train
    st.session_state.X_test_adv_clf = X_test
    st.session_state.y_train_adv_clf = y_train
    st.session_state.y_test_adv_clf = y_test
    st.session_state.split_adv_clf = "sleep"
    st.session_state.split_data_viewer_adv_clf = "active"
    st.rerun()


def filler(data):
    """Fills missing values using the fitted imputer for numerical and mode for categorical."""
    num_cat = st.session_state.n_name1_adv_clf
    alp_cat = st.session_state.c_name1_adv_clf


    data_filled = data.copy()

    # Impute numerical data
    if num_cat:
        imputed_numerical = st.session_state.imputer_adv_clf.transform(data_filled[num_cat])
        data_filled[num_cat] = imputed_numerical

    # Impute categorical data
    for val in alp_cat:
        mode_val = st.session_state.X_train_adv_clf[val].mode()[0]
        data_filled[val].fillna(mode_val, inplace=True)

    return data_filled


def feature_maker2(data):
    """Creates interaction features between numerical columns and between categorical columns."""
    df = data.copy()
    num_cat = list(df.select_dtypes(include=[np.number]).columns)
    alp_cat = [col for col in df.columns if col not in num_cat]

    # Num * Num interactions
    new_features = {}
    if len(num_cat) > 1:
        for i in range(2, len(num_cat) + 1):
            for j, comb in enumerate(combinations(num_cat, i)):
                Q = df[comb[0]].astype(float) + 1
                for x in range(1, i):
                    Q *= (df[comb[x]].astype(float) + 1)
                new_features[f"num_interact_len{i}_num{j}"] = Q

    # Cat + Cat interactions
    if len(alp_cat) > 1:
        for i in range(2, len(alp_cat) + 1):
            for j, comb in enumerate(combinations(alp_cat, i)):
                Q = df[comb[0]].astype(str)
                for x in range(1, i):
                    Q += "_" + df[comb[x]].astype(str)
                new_features[f"cat_interact_len{i}_num{j}"] = Q

    if new_features:
        new_df = pd.DataFrame(new_features)
        df = pd.concat([df, new_df], axis=1)
    return df


def feature_maker3(data):
    """Creates interaction features by combining all columns as strings."""
    df = data.copy()
    all_cols = df.columns.tolist()

    new_features = {}
    new_feature_names = []
    if len(all_cols) > 1:
        for i in range(2, len(all_cols) + 1):
            for j, comb in enumerate(combinations(all_cols, i)):
                feature_name = f"all_interact_len{i}_num{j}"
                Q = df[comb[0]].astype("str")
                for x in range(1, i):
                    Q = Q + "_" + df[comb[x]].astype("str")
                new_features[feature_name] = Q
                new_feature_names.append(feature_name)

    if new_features:
        new_df = pd.DataFrame(new_features)
        df = pd.concat([df, new_df], axis=1)

    st.session_state.label_name_adv_clf = new_feature_names
    return df


def train_and_evaluate(model_name, model_instance, X_train, y_train, X_test, y_test):
    """
    Trains a model and returns its accuracy, predictions, and the FITTED model object.
    """
    fitted_model = None
    # Special handling for Deep Neural Network (DNN)
    if model_name.startswith("DNN"):
        dnn_model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu")
        ])
        is_binary = len(np.unique(y_train)) == 2
        if is_binary:
            dnn_model.add(keras.layers.Dense(1, activation="sigmoid"))
            dnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        else:
            num_classes = len(np.unique(y_train))
            dnn_model.add(keras.layers.Dense(num_classes, activation="softmax"))
            dnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        dnn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        y_probs = dnn_model.predict(X_test, verbose=0)
        predictions = (y_probs > 0.5).astype("int32").flatten() if is_binary else np.argmax(y_probs, axis=1)
        fitted_model = dnn_model

    # Standard scikit-learn models
    else:
        if isinstance(model_instance, CatBoostClassifier):
            model_instance.fit(X_train, y_train, verbose=0)
        else:
            model_instance.fit(X_train, y_train)
        predictions = model_instance.predict(X_test)
        fitted_model = model_instance

    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions, fitted_model


def select_top_models(flat_results, corr_matrices_dict, n_top=15, threshold=0.9):
    """ Selects top N models based on accuracy and diversity. """
    sorted_models = sorted(flat_results, key=lambda x: x['accuracy'], reverse=True)
    top_models = []
    if not sorted_models: return []

    top_models.append(sorted_models[0])

    for candidate in sorted_models[1:]:
        # Changed the hardcoded limit to the n_top parameter
        if len(top_models) >= n_top:
            break

        is_diverse_enough = True
        candidate_name = candidate['model_id']
        corr_matrix_to_use = corr_matrices_dict[candidate['dataset_index']]

        for selected_model in top_models:
            if selected_model['dataset_index'] == candidate['dataset_index']:
                selected_model_name = selected_model['model_id']
                if candidate_name in corr_matrix_to_use.index and selected_model_name in corr_matrix_to_use.columns:
                    correlation = corr_matrix_to_use.loc[candidate_name, selected_model_name]
                    if correlation > threshold:
                        is_diverse_enough = False
                        break
        if is_diverse_enough:
            top_models.append(candidate)

    return top_models


def get_tuner_config(model_type_name):
    """Returns a base model instance and its hyperparameter grid for tuning."""
    name = model_type_name.lower()

    if "sgd" in name:
        model = SGDClassifier(loss="log_loss", random_state=42, n_jobs=-1)
        params = {'alpha': uniform(0.0001, 0.1)}
    elif "logistic" in name:
        model = LogisticRegression(solver='liblinear', random_state=42, n_jobs=-1)
        params = {'C': uniform(0.1, 10), 'penalty': ['l1', 'l2']}
    elif "knn" in name:
        model = KNeighborsClassifier(n_jobs=-1)
        params = {'n_neighbors': randint(3, 15), 'weights': ['uniform', 'distance']}
    elif "svc" in name:
        model = SVC(probability=True, random_state=42)
        params = {'C': uniform(0.1, 10), 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly']}
    elif "tree" in name and "extra" not in name and "forest" not in name:
        model = DecisionTreeClassifier(random_state=42)
        params = {'max_depth': randint(3, 20), 'min_samples_split': randint(2, 10)}
    elif "forest" in name:
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        params = {'n_estimators': randint(100, 500), 'max_depth': randint(5, 30), 'min_samples_split': randint(2, 10)}
    elif "extra" in name:
        model = ExtraTreesClassifier(random_state=42, n_jobs=-1)
        params = {'n_estimators': randint(100, 500), 'max_depth': randint(5, 30), 'min_samples_split': randint(2, 10)}
    elif "ada" in name:
        model = AdaBoostClassifier(random_state=42)
        params = {'n_estimators': randint(50, 300), 'learning_rate': uniform(0.01, 1.0)}
    elif "gradient" in name or "gdb" in name:
        model = GradientBoostingClassifier(random_state=42)
        params = {'n_estimators': randint(100, 400), 'learning_rate': uniform(0.01, 0.3), 'max_depth': randint(3, 10)}
    elif "xgb" in name:
        model = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        params = {'n_estimators': randint(100, 400), 'learning_rate': uniform(0.01, 0.3), 'max_depth': randint(3, 10)}
    elif "lgbm" in name:
        model = LGBMClassifier(random_state=42, n_jobs=-1)
        params = {'n_estimators': randint(100, 400), 'learning_rate': uniform(0.01, 0.3), 'num_leaves': randint(20, 50)}
    elif "cat" in name:
        model = CatBoostClassifier(verbose=0, random_state=42)
        params = {}  # CatBoost is often best with defaults; no tuning here
    else:  # Default for models like Naive Bayes
        model = GaussianNB()
        params = {}

    return model, params


# --- Streamlit App UI and Workflow ---

st.markdown("<h1>🤖 AnalytiBot Advanced Classification</h1>", unsafe_allow_html=True)

with st.expander("🤔 What is this & How Does It Work?", expanded=False):
    st.markdown("""
    <p>Welcome to the <b>Advanced Classification Engine</b>! This tool automates the process of building, comparing, and ensembling multiple machine learning models to find the best solution for your classification task.</p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🏎️ The Process")
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Preprocessing & Training**", icon="📊")
    c1.write("The engine creates multiple feature sets and trains diverse models on them.")
    c2.info("**2. Model Ensembling**", icon="🧠")
    c2.write("Top models are compiled into powerful voting and stacking ensembles.")
    c3.info("**3. Super Model Prediction**", icon="🎯")
    c3.write("The best-performing ensembles are selected to form a final 'super model' for prediction.")

# --- Step 1: File Upload ---
if st.session_state.df_uploaded_adv_clf is None:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🏁 Step 1: Upload Your Data")
    st.write(
        "Upload your dataset in **CSV or Excel** format. For best results, ensure your data is clean and well-structured.")
    st.warning("Please remove any ID columns or other irrelevant features beforehand.", icon="⚠️")

    uploaded_file = st.file_uploader("Drag and drop your file or click to browse", type=['csv', 'xlsx', 'xls'],
                                     key="adv_clf_uploader")

    if uploaded_file is not None:
        try:
            with st.spinner('Reading your file...'):
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df_uploaded_adv_clf = df
            st.session_state.df_processed_adv_clf = df.copy()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: Could not read file. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.df_uploaded_adv_clf is not None:
    # --- Step 2: Data Preview & Column Selection ---
    if st.session_state.df_uploaded_adv_clf is not None:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("🔍 Step 2: Preview & Select Target")
        st.dataframe(st.session_state.df_processed_adv_clf.head())

        if st.button("🔄 Upload New File", key="adv_clf_new_upload_11"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        with st.expander("📊 Show Full Data Information"):
            lp.information(st.session_state.df_uploaded_adv_clf)



        if st.session_state.pointer_adv_clf ==0:
            col1, col2 = st.columns(2)
            with col1:
                removable_cols = st.multiselect("Optional: Choose columns to remove",
                                                list(st.session_state.df_uploaded_adv_clf.columns))
            with col2:
                target_col = st.selectbox("Select Target Column",
                                          ["--Choose--"] + list(st.session_state.df_uploaded_adv_clf.columns))

            cutter=st.slider("Select Small Chunk of data",0.0,0.5,1.0,0.05)
            cutter=int(st.session_state.df_uploaded_adv_clf.shape[0]*cutter)
            c1, c2 = st.columns([1, 1])
            if c1.button("Apply and Continue", key="ad_clf_apply1", type="primary"):
                if target_col == "--Choose--" or cutter==0.0:
                    st.error("You must select a target column or Batch size.")
                else:
                    st.session_state.df_uploaded_adv_clf = st.session_state.df_uploaded_adv_clf.head(cutter)
                    df = st.session_state.df_uploaded_adv_clf
                    st.session_state.df_target_adv_clf = df[target_col]
                    cols_to_drop = [target_col] + removable_cols
                    st.session_state.df_uploaded_adv_clf = df.drop(columns=cols_to_drop)

                    st.success(f"Target set to '{target_col}'. {len(removable_cols)} other column(s) removed.")
                    st.session_state.pointer_adv_clf = 1
                    time.sleep(1)
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 3: Start Engine ---
    if st.session_state.pointer_adv_clf == 1:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("🚀 Ready to Start the Engine?")
        if st.button("▶️ Start Engine", type="primary", key="adv_clf_start_engine"):
            st.session_state.pointer_adv_clf = 2
            st.session_state.split_adv_clf = "active"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Processing Pipeline ---
    if st.session_state.pointer_adv_clf == 2:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)

        # Data Splitting
        if st.session_state.split_adv_clf == "active":
            with st.spinner('Step 1: Splitting data into training and testing sets...'):
                splitter(st.session_state.df_uploaded_adv_clf)

        if st.session_state.split_data_viewer_adv_clf == "active":
            st.header("Step 1: Data Splitting")
            st.subheader("📊 Data Split Overview")
            train_rows, train_cols = st.session_state.X_train_adv_clf.shape
            test_rows, test_cols = st.session_state.X_test_adv_clf.shape
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Training Samples (Rows)", value=train_rows)
                st.info(f"Number of Features: {train_cols}", icon="⚙️")
            with col2:
                st.metric(label="Testing Samples (Rows)", value=test_rows)
                st.info(f"Number of Features: {test_cols}", icon="⚙️")
            st.session_state.miss_adv_clf = "active"
            st.divider()

        # Missing Value Imputation
        if st.session_state.miss_adv_clf == "active":
            st.header("Step 2: Cleaning Data")
            with st.spinner('Imputing missing values and removing duplicates...'):
                st.session_state.imputer_adv_clf = SimpleImputer(strategy="median")
                st.session_state.n_name1_adv_clf = list(
                    st.session_state.X_train_adv_clf.select_dtypes(include=[np.number]).columns)
                st.session_state.c_name1_adv_clf = [col for col in st.session_state.X_train_adv_clf.columns if
                                                    col not in st.session_state.n_name1_adv_clf]

                st.session_state.imputer_adv_clf.fit(st.session_state.X_train_adv_clf[st.session_state.n_name1_adv_clf])
                st.session_state.X_train_adv_clf = filler(st.session_state.X_train_adv_clf)
                st.session_state.X_test_adv_clf = filler(st.session_state.X_test_adv_clf)

                # Align indices after dropping duplicates
                st.session_state.X_train_adv_clf.drop_duplicates(inplace=True)
                st.session_state.y_train_adv_clf = st.session_state.y_train_adv_clf.loc[
                    st.session_state.X_train_adv_clf.index]
                st.session_state.X_test_adv_clf.drop_duplicates(inplace=True)
                st.session_state.y_test_adv_clf = st.session_state.y_test_adv_clf.loc[
                    st.session_state.X_test_adv_clf.index]

            st.subheader("🧹 Missing Values & Duplicates Handled")
            train_rows, test_rows = st.session_state.X_train_adv_clf.shape[0], st.session_state.X_test_adv_clf.shape[0]
            col1, col2 = st.columns(2)
            col1.metric(label="Training Samples (Rows) After Cleaning", value=train_rows)
            col2.metric(label="Testing Samples (Rows) After Cleaning", value=test_rows)
            st.session_state.feature_adv_clf = "active"
            st.divider()

        # Feature Engineering
        if st.session_state.feature_adv_clf == "active":
            st.header("Step 3: Feature Engineering")
            with st.spinner('Creating new feature sets...'):
                data_train = st.session_state.X_train_adv_clf.copy()
                data_test = st.session_state.X_test_adv_clf.copy()

                # Set 1: Baseline
                st.session_state.X_train1_adv_clf = data_train
                st.session_state.X_test1_adv_clf = data_test
                # Set 2: Num/Cat Interactions
                st.session_state.X_train2_adv_clf = feature_maker2(data_train)
                st.session_state.X_test2_adv_clf = feature_maker2(data_test)
                # Set 3: All Column Interactions
                st.session_state.X_train3_adv_clf = feature_maker3(data_train)
                st.session_state.X_test3_adv_clf = feature_maker3(data_test)

            st.subheader("🏹 Generated Feature Sets")
            tab1, tab2, tab3 = st.tabs(["Set 1: Baseline", "Set 2: Interactions", "Set 3: All Interactions"])
            with tab1:
                st.info("Baseline dataset after cleaning, with no new features added.")
                st.metric("Number of Features", st.session_state.X_train1_adv_clf.shape[1])
                st.dataframe(st.session_state.X_train1_adv_clf.head())
            with tab2:
                st.info(
                    "Includes new features from interactions between numerical columns and between categorical columns.")
                st.metric("Number of Features", st.session_state.X_train2_adv_clf.shape[1])
                st.dataframe(st.session_state.X_train2_adv_clf.head())
            with tab3:
                st.info("Includes new features created by combining all columns into new categorical features.")
                st.metric("Number of Features", st.session_state.X_train3_adv_clf.shape[1])
                st.dataframe(st.session_state.X_train3_adv_clf.head())

            st.session_state.encoding_adv_clf = "active"
            st.divider()

        # Encoding and Scaling
        if st.session_state.encoding_adv_clf == "active":
            st.header("Step 4: Encoding & Scaling")
            with st.spinner('Applying transformations to all feature sets... This may take a moment.'):
                # --- Process Feature Set 1 ---
                num_cat1 = st.session_state.X_train1_adv_clf.select_dtypes(include=np.number).columns.tolist()
                alp_cat1 = st.session_state.X_train1_adv_clf.select_dtypes(exclude=np.number).columns.tolist()
                pipeline1 = ColumnTransformer([("Scale", StandardScaler(), num_cat1),
                                               ("OneHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                                alp_cat1)], remainder="passthrough")
                X_train1_transformed = pipeline1.fit_transform(st.session_state.X_train1_adv_clf)
                X_test1_transformed = pipeline1.transform(st.session_state.X_test1_adv_clf)
                cols1 = num_cat1 + pipeline1.named_transformers_['OneHot'].get_feature_names_out(alp_cat1).tolist()
                st.session_state.X_train1_adv_clf = pd.DataFrame(X_train1_transformed, columns=cols1,
                                                                 index=st.session_state.X_train_adv_clf.index)
                st.session_state.X_test1_adv_clf = pd.DataFrame(X_test1_transformed, columns=cols1,
                                                                index=st.session_state.X_test_adv_clf.index)

                # --- Process Feature Set 2 ---
                before2=st.session_state.X_train2_adv_clf.shape[1]
                num_cat2 = st.session_state.X_train2_adv_clf.select_dtypes(include=np.number).columns.tolist()
                alp_cat2 = st.session_state.X_train2_adv_clf.select_dtypes(exclude=np.number).columns.tolist()
                pipeline2 = ColumnTransformer([("Scale", StandardScaler(), num_cat2),
                                               ("OneHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                                alp_cat2)], remainder="passthrough")
                X_train2_transformed = pipeline2.fit_transform(st.session_state.X_train2_adv_clf)
                X_test2_transformed = pipeline2.transform(st.session_state.X_test2_adv_clf)
                cols2 = num_cat2 + pipeline2.named_transformers_['OneHot'].get_feature_names_out(alp_cat2).tolist()
                st.session_state.X_train2_adv_clf = pd.DataFrame(X_train2_transformed, columns=cols2,
                                                                 index=st.session_state.X_train_adv_clf.index)
                st.session_state.X_test2_adv_clf = pd.DataFrame(X_test2_transformed, columns=cols2,
                                                                index=st.session_state.X_test_adv_clf.index)

                # --- Process Feature Set 3 ---
                before3 = st.session_state.X_train3_adv_clf.shape[1]
                X_train_prep3 = st.session_state.X_train3_adv_clf.copy()
                X_test_prep3 = st.session_state.X_test3_adv_clf.copy()
                y_train = st.session_state.y_train_adv_clf

                # Target Encoding with Stratified K-Fold
                target_enc_cols = st.session_state.label_name_adv_clf
                folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, val_idx in folds.split(X_train_prep3, y_train):
                    X_train_fold, X_val_fold = X_train_prep3.iloc[train_idx], X_train_prep3.iloc[val_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    encoder = TargetEncoder(cols=target_enc_cols)
                    encoder.fit(X_train_fold, y_train_fold)
                    X_train_prep3.iloc[val_idx, :] = encoder.transform(X_val_fold)

                final_encoder = TargetEncoder(cols=target_enc_cols)
                final_encoder.fit(X_train_prep3, y_train)
                X_test_prep3 = final_encoder.transform(X_test_prep3)

                # Final Scaling and OHE
                orig_alp_cat = st.session_state.c_name1_adv_clf
                cols_to_scale = st.session_state.n_name1_adv_clf + target_enc_cols
                pipeline3 = ColumnTransformer([("Scale", StandardScaler(), cols_to_scale),
                                               ("OneHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                                orig_alp_cat)], remainder="passthrough")
                X_train3_transformed = pipeline3.fit_transform(X_train_prep3)
                X_test3_transformed = pipeline3.transform(X_test_prep3)
                cols3 = cols_to_scale + pipeline3.named_transformers_['OneHot'].get_feature_names_out(
                    orig_alp_cat).tolist()
                st.session_state.X_train3_adv_clf = pd.DataFrame(X_train3_transformed, columns=cols3,
                                                                 index=st.session_state.X_train_adv_clf.index)
                st.session_state.X_test3_adv_clf = pd.DataFrame(X_test3_transformed, columns=cols3,
                                                                index=st.session_state.X_test_adv_clf.index)

        # Display Encoding Results
            encoder=LabelEncoder()
            st.session_state.y_train_adv_clf=encoder.fit_transform(st.session_state.y_train_adv_clf)
            st.session_state.y_test_adv_clf=encoder.transform(st.session_state.y_test_adv_clf)
            st.subheader("🧊 Final Transformed Feature Sets")

            tab1, tab2, tab3 = st.tabs(["Processed Set 1", "Processed Set 2", "Processed Set 3"])

            with tab1:
                st.info("Standard Scaling on numerical features and One-Hot Encoding on categorical features.")
                c1, c2 = st.columns(2)
                c1.metric("Features Before", st.session_state.X_train_adv_clf.shape[1])
                c2.metric("Features After", st.session_state.X_train1_adv_clf.shape[1],
                          delta=st.session_state.X_train1_adv_clf.shape[1] - st.session_state.X_train_adv_clf.shape[1])
                st.write("**Preview of Transformed Data:**")
                st.dataframe(st.session_state.X_train1_adv_clf.head())

            with tab2:
                st.info("Same processing as Set 1, applied after creating interaction features.")
                c1, c2 = st.columns(2)
                c1.metric("Features Before",
                          before2)  # After feature creation but before encoding
                c2.metric("Features After", st.session_state.X_train2_adv_clf.shape[1],
                          delta=st.session_state.X_train2_adv_clf.shape[1] - before2)
                st.write("**Preview of Transformed Data:**")
                st.dataframe(st.session_state.X_train2_adv_clf.head())

            with tab3:
                st.info("Target Encoding on new interaction features, plus standard processing on original features.")
                c1, c2 = st.columns(2)
                c1.metric("Features Before", before3)
                c2.metric("Features After", st.session_state.X_train3_adv_clf.shape[1],
                          delta=st.session_state.X_train3_adv_clf.shape[1] - before3)
                st.write("**Preview of Transformed Data:**")
                st.dataframe(st.session_state.X_train3_adv_clf.head())

            mapping_df = pd.DataFrame({
                'Original Class (Old Value)': encoder.classes_,
                'Encoded Label (New Value)': range(len(encoder.classes_))
            })

            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(label="Total Unique Classes", value=len(encoder.classes_))
                st.write("**Encoding Map:**")
                st.dataframe(mapping_df, hide_index=True)

            with col2:
                st.write("**Encoded Training Labels Preview:**")
                # Display the transformed numpy array as a DataFrame
                preview_df = pd.DataFrame(st.session_state.y_train_adv_clf, columns=['Encoded Target'])
                st.dataframe(preview_df.head())

            st.success("All preprocessing steps are complete! The next step will be model training.")
            st.session_state.normal_adv_clf="active"
            st.divider()

        if st.session_state.normal_adv_clf=="active":
            st.header("Step 5: Model Training")

            st.session_state.normal_result_adv_clf = []
            st.session_state.normal_corr_adv_clf = {}
            st.session_state.trained_models_adv_clf = {}

            with st.spinner("Initializing Training..."):
                models_to_train = {
                    "SGD": SGDClassifier(loss="log_loss", n_jobs=-1, random_state=42),
                    "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=42),
                    "KNN": KNeighborsClassifier(n_jobs=-1), "SVC": SVC(probability=True, random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
                    "Extra Trees": ExtraTreesClassifier(n_jobs=-1, random_state=42),
                    "AdaBoost": AdaBoostClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42), "Gaussian NB": GaussianNB(),
                    "MLP": MLPClassifier(max_iter=500, random_state=42),
                    "XGBoost": XGBClassifier(n_jobs=-1, use_label_encoder=False, eval_metric='logloss',
                                             random_state=42),
                    "CatBoost": CatBoostClassifier(random_state=42),
                    "LightGBM": LGBMClassifier(n_jobs=-1, random_state=42), "DNN": None
                }
                datasets = {
                    1: (st.session_state.X_train1_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test1_adv_clf, st.session_state.y_test_adv_clf),
                    2: (st.session_state.X_train2_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test2_adv_clf, st.session_state.y_test_adv_clf),
                    3: (st.session_state.X_train3_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test3_adv_clf, st.session_state.y_test_adv_clf)
                }

                progress_text, progress_bar, status_placeholder = st.empty(), st.progress(0), st.empty()
                total_runs, run_count = len(models_to_train) * len(datasets), 0

                all_model_results_flat, all_predictions = [], {1: {}, 2: {}, 3: {}}
                accuracy_results_dict = {model_name: [] for model_name in models_to_train}
                trained_models_dict = {}

            # --- Training Loop ---
            for model_name, model_instance in models_to_train.items():
                for i in range(1, 4):
                    run_count += 1
                    X_train, y_train, X_test, y_test = datasets[i]
                    model_id = f"{model_name.replace(' ', '_')}_{i}"
                    with status_placeholder.container(): st.info(
                        f"**Training:** `{model_id}` ({run_count}/{total_runs})")
                    acc, preds, fitted_model = train_and_evaluate(model_id, model_instance, X_train, y_train, X_test,
                                                                  y_test)

                    all_predictions[i][model_id] = preds
                    accuracy_results_dict[model_name].append(acc)
                    trained_models_dict[model_id] = fitted_model
                    all_model_results_flat.append(
                        {'model_id': model_id, 'model_type': model_name, 'accuracy': acc, 'dataset_index': i})
                    progress_bar.progress(run_count / total_runs)

            progress_text.success("🎉 All models trained successfully!")
            status_placeholder.empty()

            with st.spinner("Saving results and selecting top models..."):
                st.session_state.trained_models_adv_clf = trained_models_dict
                for i in range(1, 4):
                    y_test = datasets[i][3]
                    pred_df = pd.DataFrame(all_predictions[i])
                    error_df = (pred_df.apply(lambda col: col != y_test)).astype(np.int8)
                    st.session_state.normal_corr_adv_clf[i] = error_df.corr()
                for model_name, acc_list in accuracy_results_dict.items():
                    st.session_state.normal_result_adv_clf.append([model_name, acc_list])

                # Select Top 15 Models
                st.session_state.top_15_models = select_top_models(
                    all_model_results_flat,
                    st.session_state.normal_corr_adv_clf,
                    n_top=15  # Explicitly set to 15
                )

            st.success("Analysis complete! View the results below.")


    # --- Display Results Section ---

            st.header("📊 Training Results & Analysis")

            tab1, tab2, tab3 = st.tabs(["**Performance Dashboard**", "**Error Correlation**", "🏆 **Top 15 Models**"])

            with tab1:
                st.subheader("Model Accuracy Scores")
                st.info("All models performance data.")
                display_df = pd.DataFrame.from_records(st.session_state.normal_result_adv_clf,
                                                       columns=["Model", "Scores"]).set_index("Model")
                display_df[["Set 1", "Set 2", "Set 3"]] = pd.DataFrame(display_df.Scores.tolist(),
                                                                       index=display_df.index)
                st.dataframe(
                    display_df[["Set 1", "Set 2", "Set 3"]].style.background_gradient(cmap='Greens').format("{:.4f}"),
                    use_container_width=True)

            with tab2:
                st.subheader("Error Correlation Heatmaps")
                st.info(
                    "This show the similarity between the diffrent models.")
                for i, corr_matrix in st.session_state.normal_corr_adv_clf.items():
                    with st.expander(f"Heatmap for Data Set {i}", expanded=(i == 1)):
                        fig, ax = plt.subplots(figsize=(12, 10))
                        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
                        ax.set_title(f'Error Correlation on Data Set {i}', fontsize=16)
                        st.pyplot(fig)

            with tab3:
                st.subheader("✨ The Champions: Top 15 Models")
                st.info("Top 15 Model on the basis of accuracy and dissimilarity.")

                if 'top_15_models' in st.session_state and st.session_state.top_15_models:
                    for i, model in enumerate(st.session_state.top_15_models):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        col1.metric(label=f"#{i + 1}: {model['model_type']}", value=f"{model['accuracy']:.4f}")
                        col2.write(f"**Model ID:** `{model['model_id']}`")
                        col3.write(f"**On Set:** {model['dataset_index']}")
                        st.divider()
                else:
                    st.warning("Could not select top models. Please check the training results.")

        if 'top_15_models' in st.session_state and st.session_state.top_15_models:
            st.header("✨ Hyperparameter Tuning & Final Selection")
            st.session_state.tuning_summary = []
            st.session_state.final_best_models = []

            top_models_to_tune = st.session_state.top_15_models

            # --- UI Placeholders ---
            progress_text = st.empty()
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            with st.spinner("Initializing Tuning Process..."):
                datasets = {
                    1: (st.session_state.X_train1_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test1_adv_clf, st.session_state.y_test_adv_clf),
                    2: (st.session_state.X_train2_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test2_adv_clf, st.session_state.y_test_adv_clf),
                    3: (st.session_state.X_train3_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test3_adv_clf, st.session_state.y_test_adv_clf)
                }
                num_models_to_tune = len(top_models_to_tune)

            # --- Main Tuning Loop ---
            for i, model_info in enumerate(top_models_to_tune):
                model_id = model_info['model_id']
                model_type = model_info['model_type']
                original_accuracy = model_info['accuracy']
                dataset_index = model_info['dataset_index']

                progress_bar.progress((i + 1) / num_models_to_tune)
                progress_text.text(f"Processing Model {i + 1}/{num_models_to_tune}")
                status_placeholder.info(f"**Now Tuning:** `{model_id}`")

                # --- Special Case: Skip DNN Tuning ---
                if model_type == "DNN":
                    status_placeholder.warning(f"`{model_id}` (DNN) is skipped from tuning. Keeping the original.")
                    final_model = st.session_state.trained_models_adv_clf[model_id]
                    st.session_state.tuning_summary.append({
                        "model_id": model_id, "status": "Skipped (DNN)", "original_accuracy": original_accuracy,
                        "new_accuracy": original_accuracy, "best_params": "N/A"
                    })
                    st.session_state.final_best_models.append({'info': model_info, 'model': final_model})
                    continue

                # Get data and tuner config
                X_train, y_train, X_test, y_test = datasets[dataset_index]
                base_model, params = get_tuner_config(model_type)

                # --- Perform Tuning if params are available ---
                if params:
                    name = model_type.lower()
                    if "svc" in name or "forest" in name or "extra" in name or "gradient" in name or "gdb" in name or "ada" in name:
                        tuner = RandomizedSearchCV(estimator=base_model, param_distributions=params,
                                                   n_iter=15, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
                    else:
                        tuner = RandomizedSearchCV(estimator=base_model, param_distributions=params,
                                                   n_iter=25, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)

                    tuner.fit(X_train, y_train)
                    tuned_accuracy = tuner.score(X_test, y_test)

                    # Compare and decide whether to keep the new model
                    if tuned_accuracy > original_accuracy:
                        status_placeholder.success(
                            f"**Improved!** `{model_id}`: {original_accuracy:.4f} -> {tuned_accuracy:.4f}")
                        final_model = tuner.best_estimator_
                        model_info['accuracy'] = tuned_accuracy
                        st.session_state.tuning_summary.append({
                            "model_id": model_id, "status": "Improved", "original_accuracy": original_accuracy,
                            "new_accuracy": tuned_accuracy, "best_params": tuner.best_params_
                        })
                    else:
                        status_placeholder.warning(
                            f"**No Improvement.** `{model_id}`: Keeping original model ({original_accuracy:.4f}).")
                        final_model = st.session_state.trained_models_adv_clf[model_id]
                        st.session_state.tuning_summary.append({
                            "model_id": model_id, "status": "Not Improved", "original_accuracy": original_accuracy,
                            "new_accuracy": tuned_accuracy, "best_params": "N/A"
                        })
                else:  # For models like CatBoost or Naive Bayes with no tuning grid
                    status_placeholder.warning(f"`{model_id}` has no tuning grid. Keeping original model.")
                    final_model = st.session_state.trained_models_adv_clf[model_id]
                    st.session_state.tuning_summary.append({
                        "model_id": model_id, "status": "No Tuning Grid", "original_accuracy": original_accuracy,
                        "new_accuracy": original_accuracy, "best_params": "N/A"
                    })

                st.session_state.final_best_models.append({'info': model_info, 'model': final_model})

            progress_text.success("✅ All tuning steps are complete!")
            status_placeholder.empty()

            st.subheader("📊 Hyperparameter Tuning Results")
            st.info(
                "Here is the information about the Tuned model and original model.")

            for result in st.session_state.tuning_summary:
                with st.expander(f"**{result['model_id']}** - Status: {result['status']}"):
                    cols = st.columns(3)
                    original_acc = result['original_accuracy']
                    new_acc = result['new_accuracy']

                    cols[0].metric("Original Accuracy", f"{original_acc:.4f}")

                    if result['status'] == "Improved":
                        cols[1].metric("Tuned Accuracy", f"{new_acc:.4f}", delta=f"{new_acc - original_acc:.4f}")
                        cols[2].write("**Best Parameters:**")
                        cols[2].json(result['best_params'])
                    elif result['status'] == "Not Improved":
                        cols[1].metric("Tuning Attempt", f"{new_acc:.4f}", delta=f"{new_acc - original_acc:.4f}")
                        cols[2].warning("Original model was better and has been kept.")
                    else:  # Skipped or No Grid
                        cols[1].metric("Tuned Accuracy", "N/A")
                        cols[2].info("This model was not tuned.")

            st.session_state.voting_adv_clf="active"

        if st.session_state.voting_adv_clf == "active":
            st.header("🤹🏻 Voting Classifier")
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            with st.spinner("Ensemble building in progress..."):
                final_models_info = [m['info'] for m in st.session_state.final_best_models]
                models_by_set = {
                    i: sorted(
                        [m for m in final_models_info if m['dataset_index'] == i],
                        key=lambda x: x['accuracy'], reverse=True
                    ) for i in range(1, 4)
                }

                datasets = {
                    1: (st.session_state.X_train1_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test1_adv_clf, st.session_state.y_test_adv_clf),
                    2: (st.session_state.X_train2_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test2_adv_clf, st.session_state.y_test_adv_clf),
                    3: (st.session_state.X_train3_adv_clf, st.session_state.y_train_adv_clf,
                        st.session_state.X_test3_adv_clf, st.session_state.y_test_adv_clf)
                }

                st.session_state.voting_results_adv_clf = []

                voting_tasks = []
                for i in range(1, 4):
                    models_for_set = models_by_set[i]
                    if not models_for_set:
                        continue

                    num_to_take = len(models_for_set) // 2
                    if num_to_take >= 2:
                        top_half_models_info = models_for_set[:num_to_take]
                        estimators = [
                            (m['model_id'], st.session_state.trained_models_adv_clf[m['model_id']])
                            for m in top_half_models_info
                            if not m["model_id"].startswith("MLP") and not m["model_id"].startswith("DNN")
                        ]
                        if len(estimators) > 1:
                            voting_tasks.append({
                                "type": "half",
                                "set_index": i,
                                "models": top_half_models_info,
                                "estimators": estimators
                            })

                    estimators_all = [
                        (m['model_id'], st.session_state.trained_models_adv_clf[m['model_id']])
                        for m in models_for_set
                        if not m["model_id"].startswith("MLP") and not m["model_id"].startswith("DNN")
                    ]
                    if len(estimators_all) > 1:
                        voting_tasks.append({
                            "type": "all",
                            "set_index": i,
                            "models": models_for_set,
                            "estimators": estimators_all
                        })

                total_tasks = len(voting_tasks)
                task_count = 0


                def safe_progress(bar, count, total):
                    bar.progress(min(count / total, 1.0))


                for task in voting_tasks:
                    task_count += 1
                    safe_progress(progress_bar, task_count, total_tasks)
                    i = task["set_index"]
                    X_train, y_train, X_test, y_test = datasets[i]

                    status_placeholder.info(
                        f"Training Voter for Set {i} ({'Top Half' if task['type'] == 'half' else 'All'} Models)...")
                    voter = VotingClassifier(estimators=task["estimators"], voting='soft')
                    voter.fit(X_train, y_train)
                    accuracy = voter.score(X_test, y_test)
                    best_single_acc = task["models"][0]['accuracy']
                    st.session_state.voting_results_adv_clf.append({
                        "name": f"Voter | Set {i} ({'Top Half' if task['type'] == 'half' else 'All'} Models)",
                        "accuracy": accuracy,
                        "improvement": accuracy - best_single_acc,
                        "best_single_model_acc": best_single_acc,
                        "components": [m['model_id'] for m in task["models"]]
                    })

            st.subheader("🏁 Voting Ensemble Results")
            for result in st.session_state.voting_results_adv_clf:
                with st.expander(f"**{result['name']}** - Final Accuracy: {result['accuracy']:.4f}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Ensemble Accuracy", f"{result['accuracy']:.4f}",
                                  delta=f"{result['improvement']:.4f} vs. Best Single Model")
                        st.write(f"Best Single Model Accuracy: `{result['best_single_model_acc']:.4f}`")
                    with col2:
                        st.write("**Models Used in Voting:**")
                        st.dataframe(pd.DataFrame(result['components'], columns=["Model ID"]), use_container_width=True)

            st.session_state.stacking_adv_clf = "active"

        # ---------- STACKING SECTION ----------
        if st.session_state.stacking_adv_clf == "active":
            st.header("👑 Stacking Ensemble: The Super Models")
            st.info("We use predictions from selected base models as features to train a meta-model.")
            with st.spinner("Building Stacking Ensemble..."):
                datasets = {
                    1: (st.session_state.X_train1_adv_clf, st.session_state.X_test1_adv_clf),
                    2: (st.session_state.X_train2_adv_clf, st.session_state.X_test2_adv_clf),
                    3: (st.session_state.X_train3_adv_clf, st.session_state.X_test3_adv_clf)
                }

                y_train = st.session_state.y_train_adv_clf
                y_test = st.session_state.y_test_adv_clf

                meta_features_train_list = []
                meta_features_test_list = []
                final_models = st.session_state.final_best_models
                progress_bar_stack = st.progress(0, text="Generating meta-features from base models...")

                for i, model_dict in enumerate(final_models):
                    model_obj = model_dict['model']
                    model_info = model_dict['info']
                    dataset_idx = model_info['dataset_index']
                    model_name = model_info['model_type']

                    # ❌ Skip MLP and DNN as base models
                    if model_name in ["MLP", "DNN"]:
                        continue

                    X_train, X_test = datasets[dataset_idx]
                    X_train = X_train.reindex(columns=model_obj.feature_names_in_, fill_value=0)
                    X_test = X_test.reindex(columns=model_obj.feature_names_in_, fill_value=0)

                    train_preds = model_obj.predict_proba(X_train)
                    test_preds = model_obj.predict_proba(X_test)

                    meta_features_train_list.append(train_preds)
                    meta_features_test_list.append(test_preds)

                    progress_bar_stack.progress((i + 1) / len(final_models),
                                                text=f"Generated from: {model_info['model_id']}")

                meta_X_train = np.hstack(meta_features_train_list)
                meta_X_test = np.hstack(meta_features_test_list)

                st.session_state.stacking_results_adv_clf = []


                def create_meta_dnn(input_shape, num_classes):
                    model = keras.Sequential([
                        keras.layers.Input(shape=(input_shape,)),
                        keras.layers.Dense(32, activation='relu'),
                        keras.layers.Dense(16, activation='relu')
                    ])
                    if num_classes <= 2:
                        model.add(keras.layers.Dense(1, activation='sigmoid'))
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    else:
                        model.add(keras.layers.Dense(num_classes, activation='softmax'))
                        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    return model


                num_classes = len(np.unique(y_train))
                is_binary = num_classes <= 2

                # a) XGBoost Meta-Learner
                progress_bar_stack.progress(0.2, text="Training XGBoost Meta-Model...")
                meta_xgb = XGBClassifier(n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42)
                meta_xgb.fit(meta_X_train, y_train)
                proba_xgb = meta_xgb.predict_proba(meta_X_test)
                preds_xgb = np.argmax(proba_xgb, axis=1)
                acc_xgb = accuracy_score(y_test, preds_xgb)
                st.session_state.stacking_results_adv_clf.append({'name': 'Stacker (XGB Meta)', 'accuracy': acc_xgb})

                # b) Logistic Regression Meta-Learner
                progress_bar_stack.progress(0.4, text="Training Logistic Regression Meta-Model...")
                meta_lr = LogisticRegression(n_jobs=-1, random_state=42)
                meta_lr.fit(meta_X_train, y_train)
                proba_lr = meta_lr.predict_proba(meta_X_test)
                preds_lr = np.argmax(proba_lr, axis=1)
                acc_lr = accuracy_score(y_test, preds_lr)
                st.session_state.stacking_results_adv_clf.append(
                    {'name': 'Stacker (Logistic Meta)', 'accuracy': acc_lr})

                # c) DNN Meta-Learner
                progress_bar_stack.progress(0.6, text="Training DNN Meta-Model...")
                meta_dnn = create_meta_dnn(meta_X_train.shape[1], num_classes)
                meta_dnn.fit(meta_X_train, y_train, epochs=30, batch_size=32, verbose=0)
                proba_dnn = meta_dnn.predict(meta_X_test, verbose=0)
                preds_dnn = (proba_dnn > 0.5).astype("int32").flatten() if is_binary else np.argmax(proba_dnn, axis=1)
                acc_dnn = accuracy_score(y_test, preds_dnn)
                st.session_state.stacking_results_adv_clf.append({'name': 'Stacker (DNN Meta)', 'accuracy': acc_dnn})

                # Weighted Combos
                progress_bar_stack.progress(0.8, text="Calculating weighted meta-ensembles...")
                combos = [
                    ("XGB 0.6 + DNN 0.4", 0.6 * proba_xgb + 0.4 * proba_dnn),
                    ("XGB 0.6 + LR 0.4", 0.6 * proba_xgb + 0.4 * proba_lr),
                    ("DNN 0.6 + LR 0.4", 0.6 * proba_dnn + 0.4 * proba_lr),
                ]
                for name, proba in combos:
                    preds = np.argmax(proba, axis=1)
                    acc = accuracy_score(y_test, preds)
                    st.session_state.stacking_results_adv_clf.append({'name': f'Stacker ({name})', 'accuracy': acc})

                progress_bar_stack.progress(1.0, text="Stacking complete!")
                time.sleep(1)
                progress_bar_stack.empty()

            st.subheader("📊 Stacking Ensemble Performance")
            df = pd.DataFrame(st.session_state.stacking_results_adv_clf).sort_values('accuracy', ascending=False)
            st.dataframe(df.style.background_gradient(cmap='viridis').format({'accuracy': '{:.4f}'}),
                         use_container_width=True)

            # --- 5. Final Top 3 Models Overall ---
            st.header("🎉 Overall Analysis Complete: The Final Podium")
            st.balloons()

            with st.spinner("Compiling all results to find the ultimate champions..."):
                all_results = []

                # Get tuned model results
                for res in st.session_state.tuning_summary:
                    all_results.append({
                        'name': res['model_id'],
                        'accuracy': res['new_accuracy'] if res['status'] == 'Improved' else res['original_accuracy'],
                        'type': 'Tuned Single Model'
                    })

                # Get voting model results
                for res in st.session_state.voting_results_adv_clf:
                    all_results.append({
                        'name': res['name'],
                        'accuracy': res['accuracy'],
                        'type': 'Voting Ensemble'
                    })

                # Get stacking model results
                for res in st.session_state.stacking_results_adv_clf:
                    all_results.append({
                        'name': res['name'],
                        'accuracy': res['accuracy'],
                        'type': 'Stacking Ensemble'
                    })

                overall_df = pd.DataFrame(all_results).sort_values('accuracy', ascending=False).reset_index(drop=True)
                top_3 = overall_df.head(3)

            st.subheader("🥇🥈🥉 Top 3 Performing Models")

            cols = st.columns(3)
            emojis = ["🥇 Gold", "🥈 Silver", "🥉 Bronze"]

            for i, row in top_3.iterrows():
                with cols[i]:
                    st.markdown(f"<h3 style='text-align: center; color: #FFD700;'>{emojis[i]}</h3>",
                                unsafe_allow_html=True)
                    st.markdown(f"<div class='step-container' style='text-align: center; border-color: #FFD700;'>"
                                f"<h4>{row['name']}</h4>"
                                f"<p>Type: <b>{row['type']}</b></p>"
                                f"<h2 style='color: #33FFC7;'>{row['accuracy']:.4f}</h2>"
                                f"<p>Accuracy</p>"
                                f"</div>", unsafe_allow_html=True)
