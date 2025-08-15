import time
import streamlit as st
import pandas as pd
import numpy as np
from Advanced import load_plot_adv_clf as lp  # Assuming this module has general data info functions
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from itertools import combinations
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, VotingRegressor, StackingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint


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
    """Initializes all necessary session state variables for regression."""
    # Suffix changed to _adv_reg to avoid conflicts
    if 'df_uploaded_adv_reg' not in st.session_state: st.session_state.df_uploaded_adv_reg = None
    if 'df_processed_adv_reg' not in st.session_state: st.session_state.df_processed_adv_reg = None
    if 'df_target_adv_reg' not in st.session_state: st.session_state.df_target_adv_reg = None
    if 'X_train_adv_reg' not in st.session_state: st.session_state.X_train_adv_reg = None
    if 'X_test_adv_reg' not in st.session_state: st.session_state.X_test_adv_reg = None
    if 'y_train_adv_reg' not in st.session_state: st.session_state.y_train_adv_reg = None
    if 'y_test_adv_reg' not in st.session_state: st.session_state.y_test_adv_reg = None
    if 'X_train1_adv_reg' not in st.session_state: st.session_state.X_train1_adv_reg = None
    if 'X_train2_adv_reg' not in st.session_state: st.session_state.X_train2_adv_reg = None
    if 'X_train3_adv_reg' not in st.session_state: st.session_state.X_train3_adv_reg = None
    if 'X_test1_adv_reg' not in st.session_state: st.session_state.X_test1_adv_reg = None
    if 'X_test2_adv_reg' not in st.session_state: st.session_state.X_test2_adv_reg = None
    if 'X_test3_adv_reg' not in st.session_state: st.session_state.X_test3_adv_reg = None
    if "n_name1_adv_reg" not in st.session_state: st.session_state.n_name1_adv_reg = None
    if "c_name1_adv_reg" not in st.session_state: st.session_state.c_name1_adv_reg = None
    if "label_name_adv_reg" not in st.session_state: st.session_state.label_name_adv_reg = None
    if "imputer_adv_reg" not in st.session_state: st.session_state.imputer_adv_reg = None
    if 'pointer_adv_reg' not in st.session_state: st.session_state.pointer_adv_reg = 0
    flags = ["split_adv_reg", "split_data_viewer_adv_reg", "miss_adv_reg", "feature_adv_reg", "encoding_adv_reg",
             "normal_adv_reg", "normal_model_adv_reg", "normal_result_adv_reg", "normal_corr_adv_reg",
             "tuned_adv_reg", "tuned_model_adv_reg", "voting_adv_reg", "stacking_adv_reg"]
    for flag in flags:
        if flag not in st.session_state: st.session_state[flag] = "sleep"


initialize_state()


def splitter(data):
    """Splits data into training and testing sets."""
    X = data
    y = st.session_state.df_target_adv_reg
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.session_state.X_train_adv_reg = X_train
    st.session_state.X_test_adv_reg = X_test
    st.session_state.y_train_adv_reg = y_train
    st.session_state.y_test_adv_reg = y_test
    st.session_state.split_adv_reg = "sleep"
    st.session_state.split_data_viewer_adv_reg = "active"
    st.rerun()


def filler(data):
    """Fills missing values using the fitted imputer for numerical and mode for categorical."""
    num_cat = st.session_state.n_name1_adv_reg
    alp_cat = st.session_state.c_name1_adv_reg
    data_filled = data.copy()
    if num_cat:
        imputed_numerical = st.session_state.imputer_adv_reg.transform(data_filled[num_cat])
        data_filled[num_cat] = imputed_numerical
    for val in alp_cat:
        mode_val = st.session_state.X_train_adv_reg[val].mode()[0]
        data_filled[val].fillna(mode_val, inplace=True)
    return data_filled


def feature_maker2(data):
    """Creates interaction features between numerical columns and between categorical columns."""
    df = data.copy()
    num_cat = list(df.select_dtypes(include=[np.number]).columns)
    alp_cat = [col for col in df.columns if col not in num_cat]
    new_features = {}
    if len(num_cat) > 1:
        for i in range(2, len(num_cat) + 1):
            for j, comb in enumerate(combinations(num_cat, i)):
                Q = df[comb[0]].astype(float) + 1
                for x in range(1, i):
                    Q *= (df[comb[x]].astype(float) + 1)
                new_features[f"num_interact_len{i}_num{j}"] = Q
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
    st.session_state.label_name_adv_reg = new_feature_names
    return df


def train_and_evaluate_reg(model_name, model_instance, X_train, y_train, X_test, y_test):
    """
    Trains a regression model and returns its R-squared score, predictions, and the FITTED model object.
    """
    fitted_model = None
    if model_name.startswith("DNN"):
        dnn_model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1)  # Single output neuron, linear activation for regression
        ])
        dnn_model.compile(optimizer="adam", loss="mean_squared_error")
        dnn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        predictions = dnn_model.predict(X_test, verbose=0).flatten()
        fitted_model = dnn_model
    else:
        if isinstance(model_instance, CatBoostRegressor):
            model_instance.fit(X_train, y_train, verbose=0)
        else:
            model_instance.fit(X_train, y_train)
        predictions = model_instance.predict(X_test)
        fitted_model = model_instance

    score = r2_score(y_test, predictions)
    return score, predictions, fitted_model


def select_top_models(flat_results, corr_matrices_dict, n_top=15, threshold=0.9):
    """ Selects top N models based on R-squared and diversity. """
    sorted_models = sorted(flat_results, key=lambda x: x['r2_score'], reverse=True)
    top_models = []
    if not sorted_models: return []
    top_models.append(sorted_models[0])
    for candidate in sorted_models[1:]:
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


def get_tuner_config_reg(model_type_name):
    """Returns a base regressor model and its hyperparameter grid for tuning."""
    name = model_type_name.lower()
    if "sgd" in name:
        model = SGDRegressor(random_state=42)
        params = {'alpha': uniform(0.0001, 0.1), 'penalty': ['l1', 'l2', 'elasticnet']}
    elif "linear" in name:
        model = LinearRegression(n_jobs=-1)
        params = {}
    elif "knn" in name:
        model = KNeighborsRegressor(n_jobs=-1)
        params = {'n_neighbors': randint(3, 15), 'weights': ['uniform', 'distance']}
    elif "svr" in name:
        model = SVR()
        params = {'C': uniform(0.1, 10), 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly']}
    elif "tree" in name and "extra" not in name and "forest" not in name:
        model = DecisionTreeRegressor(random_state=42)
        params = {'max_depth': randint(3, 20), 'min_samples_split': randint(2, 10)}
    elif "forest" in name:
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        params = {'n_estimators': randint(100, 500), 'max_depth': randint(5, 30), 'min_samples_split': randint(2, 10)}
    elif "extra" in name:
        model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
        params = {'n_estimators': randint(100, 500), 'max_depth': randint(5, 30), 'min_samples_split': randint(2, 10)}
    elif "ada" in name:
        model = AdaBoostRegressor(random_state=42)
        params = {'n_estimators': randint(50, 300), 'learning_rate': uniform(0.01, 1.0)}
    elif "gradient" in name or "gdb" in name:
        model = GradientBoostingRegressor(random_state=42)
        params = {'n_estimators': randint(100, 400), 'learning_rate': uniform(0.01, 0.3), 'max_depth': randint(3, 10)}
    elif "xgb" in name:
        model = XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
        params = {'n_estimators': randint(100, 400), 'learning_rate': uniform(0.01, 0.3), 'max_depth': randint(3, 10)}
    elif "lgbm" in name:
        model = LGBMRegressor(random_state=42, n_jobs=-1)
        params = {'n_estimators': randint(100, 400), 'learning_rate': uniform(0.01, 0.3), 'num_leaves': randint(20, 50)}
    elif "cat" in name:
        model = CatBoostRegressor(verbose=0, random_state=42)
        params = {}
    else:  # Default
        model = LinearRegression()
        params = {}
    return model, params


# --- Streamlit App UI and Workflow ---
st.markdown("<h1>⚙️ AnalytiBot Advanced Regression</h1>", unsafe_allow_html=True)

with st.expander("🤔 What is this & How Does It Work?", expanded=False):
    st.markdown("""
    <p>Welcome to the <b>Advanced Regression Engine</b>! This tool automates the process of building, comparing, and ensembling multiple machine learning models to find the best solution for your regression task.</p>
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
if st.session_state.df_uploaded_adv_reg is None:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🏁 Step 1: Upload Your Data")
    st.write(
        "Upload your dataset in **CSV or Excel** format. For best results, ensure your data is clean and the target variable is numerical.")
    st.warning("Please remove any ID columns or other irrelevant features beforehand.", icon="⚠️")
    uploaded_file = st.file_uploader("Drag and drop your file or click to browse", type=['csv', 'xlsx', 'xls'],
                                     key="adv_reg_uploader")
    if uploaded_file is not None:
        try:
            with st.spinner('Reading your file...'):
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df_uploaded_adv_reg = df
            st.session_state.df_processed_adv_reg = df.copy()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: Could not read file. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.df_uploaded_adv_reg is not None:
    # --- Step 2: Data Preview & Column Selection ---
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🔍 Step 2: Preview & Select Target")
    st.dataframe(st.session_state.df_processed_adv_reg.head())
    if st.button("🔄 Upload New File", key="adv_reg_new_upload_11"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    with st.expander("📊 Show Full Data Information"):
        lp.information(st.session_state.df_uploaded_adv_reg)

    if st.session_state.pointer_adv_reg == 0:
        col1, col2 = st.columns(2)
        with col1:
            removable_cols = st.multiselect("Optional: Choose columns to remove",
                                            list(st.session_state.df_uploaded_adv_reg.columns))
        with col2:
            target_col = st.selectbox("Select Target Column",
                                      ["--Choose--"] + list(st.session_state.df_uploaded_adv_reg.columns))
        cutter = st.slider("Select Small Chunk of data", 0.0, 1.0, 1.0, 0.05)
        if cutter < 1.0:
            cutter_val = int(st.session_state.df_uploaded_adv_reg.shape[0] * cutter)
        else:
            cutter_val = st.session_state.df_uploaded_adv_reg.shape[0]

        c1, c2 = st.columns([1, 1])
        if c1.button("Apply and Continue", key="ad_reg_apply1", type="primary"):
            if target_col == "--Choose--":
                st.error("You must select a target column.")
            elif not pd.api.types.is_numeric_dtype(st.session_state.df_uploaded_adv_reg[target_col]):
                st.error("Target column must be numerical for regression.")
            else:
                st.session_state.df_uploaded_adv_reg = st.session_state.df_uploaded_adv_reg.head(cutter_val)
                df = st.session_state.df_uploaded_adv_reg
                mean_of_target=df[target_col].mean()
                st.session_state.df_target_adv_reg = df[target_col].fillna(mean_of_target)
                cols_to_drop = [target_col] + removable_cols
                st.session_state.df_uploaded_adv_reg = df.drop(columns=cols_to_drop)
                st.success(f"Target set to '{target_col}'. {len(removable_cols)} other column(s) removed.")
                st.session_state.pointer_adv_reg = 1
                time.sleep(1)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 3: Start Engine ---
    if st.session_state.pointer_adv_reg == 1:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("🚀 Ready to Start the Engine?")
        if st.button("▶️ Start Engine", type="primary", key="adv_reg_start_engine"):
            st.session_state.pointer_adv_reg = 2
            st.session_state.split_adv_reg = "active"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Processing Pipeline ---
    if st.session_state.pointer_adv_reg == 2:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        if st.session_state.split_adv_reg == "active":
            with st.spinner('Step 1: Splitting data into training and testing sets...'):
                splitter(st.session_state.df_uploaded_adv_reg)

        if st.session_state.split_data_viewer_adv_reg == "active":
            st.header("Step 1: Data Splitting")
            st.subheader("📊 Data Split Overview")
            train_rows, train_cols = st.session_state.X_train_adv_reg.shape
            test_rows, test_cols = st.session_state.X_test_adv_reg.shape
            col1, col2 = st.columns(2)
            col1.metric(label="Training Samples (Rows)", value=train_rows)
            col1.info(f"Number of Features: {train_cols}", icon="⚙️")
            col2.metric(label="Testing Samples (Rows)", value=test_rows)
            col2.info(f"Number of Features: {test_cols}", icon="⚙️")
            st.session_state.miss_adv_reg = "active"
            st.divider()

        if st.session_state.miss_adv_reg == "active":
            st.header("Step 2: Cleaning Data")
            with st.spinner('Imputing missing values and removing duplicates...'):
                st.session_state.imputer_adv_reg = SimpleImputer(strategy="median")
                st.session_state.n_name1_adv_reg = list(
                    st.session_state.X_train_adv_reg.select_dtypes(include=[np.number]).columns)
                st.session_state.c_name1_adv_reg = [col for col in st.session_state.X_train_adv_reg.columns if
                                                    col not in st.session_state.n_name1_adv_reg]
                st.session_state.imputer_adv_reg.fit(st.session_state.X_train_adv_reg[st.session_state.n_name1_adv_reg])
                st.session_state.X_train_adv_reg = filler(st.session_state.X_train_adv_reg)
                st.session_state.X_test_adv_reg = filler(st.session_state.X_test_adv_reg)
                st.session_state.X_train_adv_reg.drop_duplicates(inplace=True)
                st.session_state.y_train_adv_reg = st.session_state.y_train_adv_reg.loc[
                    st.session_state.X_train_adv_reg.index]
                st.session_state.X_test_adv_reg.drop_duplicates(inplace=True)
                st.session_state.y_test_adv_reg = st.session_state.y_test_adv_reg.loc[
                    st.session_state.X_test_adv_reg.index]
            st.subheader("🧹 Missing Values & Duplicates Handled")
            train_rows, test_rows = st.session_state.X_train_adv_reg.shape[0], st.session_state.X_test_adv_reg.shape[0]
            col1, col2 = st.columns(2)
            col1.metric(label="Training Samples (Rows) After Cleaning", value=train_rows)
            col2.metric(label="Testing Samples (Rows) After Cleaning", value=test_rows)
            st.session_state.feature_adv_reg = "active"
            st.divider()

        if st.session_state.feature_adv_reg == "active":
            st.header("Step 3: Feature Engineering")
            with st.spinner('Creating new feature sets...'):
                data_train, data_test = st.session_state.X_train_adv_reg.copy(), st.session_state.X_test_adv_reg.copy()
                st.session_state.X_train1_adv_reg, st.session_state.X_test1_adv_reg = data_train, data_test
                st.session_state.X_train2_adv_reg, st.session_state.X_test2_adv_reg = feature_maker2(
                    data_train), feature_maker2(data_test)
                st.session_state.X_train3_adv_reg, st.session_state.X_test3_adv_reg = feature_maker3(
                    data_train), feature_maker3(data_test)
            st.subheader("🏹 Generated Feature Sets")
            tab1, tab2, tab3 = st.tabs(["Set 1: Baseline", "Set 2: Interactions", "Set 3: All Interactions"])
            with tab1: st.metric("Number of Features", st.session_state.X_train1_adv_reg.shape[1]); st.dataframe(
                st.session_state.X_train1_adv_reg.head())
            with tab2: st.metric("Number of Features", st.session_state.X_train2_adv_reg.shape[1]); st.dataframe(
                st.session_state.X_train2_adv_reg.head())
            with tab3: st.metric("Number of Features", st.session_state.X_train3_adv_reg.shape[1]); st.dataframe(
                st.session_state.X_train3_adv_reg.head())
            st.session_state.encoding_adv_reg = "active"
            st.divider()

        if st.session_state.encoding_adv_reg == "active":
            st.header("Step 4: Encoding & Scaling")
            with st.spinner('Applying transformations to all feature sets...'):
                # Process Set 1
                num_cat1 = st.session_state.X_train1_adv_reg.select_dtypes(include=np.number).columns.tolist()
                alp_cat1 = st.session_state.X_train1_adv_reg.select_dtypes(exclude=np.number).columns.tolist()
                pipeline1 = ColumnTransformer([("Scale", StandardScaler(), num_cat1),
                                               ("OneHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                                alp_cat1)], remainder="passthrough")
                X_train1_transformed = pipeline1.fit_transform(st.session_state.X_train1_adv_reg)
                X_test1_transformed = pipeline1.transform(st.session_state.X_test1_adv_reg)
                cols1 = num_cat1 + pipeline1.named_transformers_['OneHot'].get_feature_names_out(alp_cat1).tolist()
                st.session_state.X_train1_adv_reg = pd.DataFrame(X_train1_transformed, columns=cols1,
                                                                 index=st.session_state.X_train_adv_reg.index)
                st.session_state.X_test1_adv_reg = pd.DataFrame(X_test1_transformed, columns=cols1,
                                                                index=st.session_state.X_test_adv_reg.index)

                # Process Set 2
                before2 = st.session_state.X_train2_adv_reg.shape[1]
                num_cat2 = st.session_state.X_train2_adv_reg.select_dtypes(include=np.number).columns.tolist()
                alp_cat2 = st.session_state.X_train2_adv_reg.select_dtypes(exclude=np.number).columns.tolist()
                pipeline2 = ColumnTransformer([("Scale", StandardScaler(), num_cat2),
                                               ("OneHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                                alp_cat2)], remainder="passthrough")
                X_train2_transformed = pipeline2.fit_transform(st.session_state.X_train2_adv_reg)
                X_test2_transformed = pipeline2.transform(st.session_state.X_test2_adv_reg)
                cols2 = num_cat2 + pipeline2.named_transformers_['OneHot'].get_feature_names_out(alp_cat2).tolist()
                st.session_state.X_train2_adv_reg = pd.DataFrame(X_train2_transformed, columns=cols2,
                                                                 index=st.session_state.X_train_adv_reg.index)
                st.session_state.X_test2_adv_reg = pd.DataFrame(X_test2_transformed, columns=cols2,
                                                                index=st.session_state.X_test_adv_reg.index)

                # Process Set 3
                before3 = st.session_state.X_train3_adv_reg.shape[1]
                X_train_prep3, X_test_prep3 = st.session_state.X_train3_adv_reg.copy(), st.session_state.X_test3_adv_reg.copy()
                y_train = st.session_state.y_train_adv_reg
                target_enc_cols = st.session_state.label_name_adv_reg
                folds = KFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, val_idx in folds.split(X_train_prep3, y_train):
                    X_train_fold, X_val_fold = X_train_prep3.iloc[train_idx], X_train_prep3.iloc[val_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    encoder = TargetEncoder(cols=target_enc_cols)
                    encoder.fit(X_train_fold, y_train_fold)
                    X_train_prep3.iloc[val_idx, :] = encoder.transform(X_val_fold)
                final_encoder = TargetEncoder(cols=target_enc_cols)
                final_encoder.fit(X_train_prep3, y_train)
                X_test_prep3 = final_encoder.transform(X_test_prep3)
                orig_alp_cat = st.session_state.c_name1_adv_reg
                cols_to_scale = st.session_state.n_name1_adv_reg + target_enc_cols
                pipeline3 = ColumnTransformer([("Scale", StandardScaler(), cols_to_scale),
                                               ("OneHot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                                orig_alp_cat)], remainder="passthrough")
                X_train3_transformed = pipeline3.fit_transform(X_train_prep3)
                X_test3_transformed = pipeline3.transform(X_test_prep3)
                cols3 = cols_to_scale + pipeline3.named_transformers_['OneHot'].get_feature_names_out(
                    orig_alp_cat).tolist()
                st.session_state.X_train3_adv_reg = pd.DataFrame(X_train3_transformed, columns=cols3,
                                                                 index=st.session_state.X_train_adv_reg.index)
                st.session_state.X_test3_adv_reg = pd.DataFrame(X_test3_transformed, columns=cols3,
                                                                index=st.session_state.X_test_adv_reg.index)

            st.subheader("🧊 Final Transformed Feature Sets")
            # Display results in tabs...
            st.success("All preprocessing steps are complete! The next step will be model training.")
            st.session_state.normal_adv_reg = "active"
            st.divider()

        if st.session_state.normal_adv_reg == "active":
            st.header("Step 5: Model Training")
            st.session_state.normal_result_adv_reg, st.session_state.normal_corr_adv_reg, st.session_state.trained_models_adv_reg = [], {}, {}
            with st.spinner("Initializing Training..."):
                models_to_train = {
                    "SGD": SGDRegressor(random_state=42), "Linear Regression": LinearRegression(n_jobs=-1),
                    "KNN": KNeighborsRegressor(n_jobs=-1), "SVR": SVR(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(n_jobs=-1, random_state=42),
                    "Extra Trees": ExtraTreesRegressor(n_jobs=-1, random_state=42),
                    "AdaBoost": AdaBoostRegressor(random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "MLP": MLPRegressor(max_iter=500, random_state=42),
                    "XGBoost": XGBRegressor(n_jobs=-1, eval_metric='rmse', random_state=42),
                    "CatBoost": CatBoostRegressor(random_state=42),
                    "LightGBM": LGBMRegressor(n_jobs=-1, random_state=42), "DNN": None
                }
                datasets = {
                    1: (st.session_state.X_train1_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test1_adv_reg, st.session_state.y_test_adv_reg),
                    2: (st.session_state.X_train2_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test2_adv_reg, st.session_state.y_test_adv_reg),
                    3: (st.session_state.X_train3_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test3_adv_reg, st.session_state.y_test_adv_reg)
                }
                progress_text, progress_bar, status_placeholder = st.empty(), st.progress(0), st.empty()
                total_runs, run_count = len(models_to_train) * len(datasets), 0
                all_model_results_flat, all_predictions = [], {1: {}, 2: {}, 3: {}}
                r2_results_dict, trained_models_dict = {model_name: [] for model_name in models_to_train}, {}

            for model_name, model_instance in models_to_train.items():
                for i in range(1, 4):
                    run_count += 1
                    X_train, y_train, X_test, y_test = datasets[i]
                    model_id = f"{model_name.replace(' ', '_')}_{i}"
                    with status_placeholder.container(): st.info(
                        f"**Training:** `{model_id}` ({run_count}/{total_runs})")
                    score, preds, fitted_model = train_and_evaluate_reg(model_id, model_instance, X_train, y_train,
                                                                        X_test, y_test)
                    all_predictions[i][model_id] = preds
                    r2_results_dict[model_name].append(score)
                    trained_models_dict[model_id] = fitted_model
                    all_model_results_flat.append(
                        {'model_id': model_id, 'model_type': model_name, 'r2_score': score, 'dataset_index': i})
                    progress_bar.progress(run_count / total_runs)

            progress_text.success("🎉 All models trained successfully!")
            status_placeholder.empty()
            with st.spinner("Saving results and selecting top models..."):
                st.session_state.trained_models_adv_reg = trained_models_dict
                for i in range(1, 4):
                    y_test = datasets[i][3]
                    pred_df = pd.DataFrame(all_predictions[i])
                    residual_df = pred_df.apply(lambda col: y_test - col)
                    st.session_state.normal_corr_adv_reg[i] = residual_df.corr()
                for model_name, score_list in r2_results_dict.items():
                    st.session_state.normal_result_adv_reg.append([model_name, score_list])
                st.session_state.top_15_models = select_top_models(all_model_results_flat,
                                                                   st.session_state.normal_corr_adv_reg, n_top=15)
            st.success("Analysis complete! View the results below.")

            st.header("📊 Training Results & Analysis")
            tab1, tab2, tab3 = st.tabs(["**Performance Dashboard**", "**Residual Correlation**", "🏆 **Top 15 Models**"])
            with tab1:
                st.subheader("Model R-squared (R²) Scores")
                display_df = pd.DataFrame.from_records(st.session_state.normal_result_adv_reg,
                                                       columns=["Model", "Scores"]).set_index("Model")
                display_df[["Set 1", "Set 2", "Set 3"]] = pd.DataFrame(display_df.Scores.tolist(),
                                                                       index=display_df.index)
                st.dataframe(
                    display_df[["Set 1", "Set 2", "Set 3"]].style.background_gradient(cmap='Greens').format("{:.4f}"),
                    use_container_width=True)
            with tab2:
                st.subheader("Residual Correlation Heatmaps")
                st.info(
                    "This shows the similarity in the errors made by different models. Low correlation is good for ensembling.")
                for i, corr_matrix in st.session_state.normal_corr_adv_reg.items():
                    with st.expander(f"Heatmap for Data Set {i}", expanded=(i == 1)):
                        fig, ax = plt.subplots(figsize=(12, 10));
                        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax, vmin=-1, vmax=1);
                        st.pyplot(fig)
            with tab3:
                st.subheader("✨ The Champions: Top 15 Models")
                if 'top_15_models' in st.session_state and st.session_state.top_15_models:
                    for i, model in enumerate(st.session_state.top_15_models):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        col1.metric(label=f"#{i + 1}: {model['model_type']}", value=f"{model['r2_score']:.4f}")
                        col2.write(f"**Model ID:** `{model['model_id']}`");
                        col3.write(f"**On Set:** {model['dataset_index']}");
                        st.divider()
                else:
                    st.warning("Could not select top models.")

        if 'top_15_models' in st.session_state and st.session_state.top_15_models:
            st.header("✨ Hyperparameter Tuning & Final Selection")
            st.session_state.tuning_summary, st.session_state.final_best_models = [], []
            top_models_to_tune = st.session_state.top_15_models
            progress_text, progress_bar, status_placeholder = st.empty(), st.progress(0), st.empty()
            with st.spinner("Initializing Tuning Process..."):
                datasets = {
                    1: (st.session_state.X_train1_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test1_adv_reg, st.session_state.y_test_adv_reg),
                    2: (st.session_state.X_train2_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test2_adv_reg, st.session_state.y_test_adv_reg),
                    3: (st.session_state.X_train3_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test3_adv_reg, st.session_state.y_test_adv_reg)
                }
                num_models_to_tune = len(top_models_to_tune)
            for i, model_info in enumerate(top_models_to_tune):
                model_id, model_type, original_r2, dataset_index = model_info['model_id'], model_info['model_type'], \
                model_info['r2_score'], model_info['dataset_index']
                progress_bar.progress((i + 1) / num_models_to_tune);
                status_placeholder.info(f"**Now Tuning:** `{model_id}`")
                if model_type == "DNN":
                    final_model = st.session_state.trained_models_adv_reg[model_id]
                    st.session_state.tuning_summary.append(
                        {"model_id": model_id, "status": "Skipped (DNN)", "original_r2": original_r2,
                         "new_r2": original_r2})
                    st.session_state.final_best_models.append({'info': model_info, 'model': final_model})
                    continue
                X_train, y_train, X_test, y_test = datasets[dataset_index]
                base_model, params = get_tuner_config_reg(model_type)
                if params:
                    tuner = RandomizedSearchCV(estimator=base_model, param_distributions=params, n_iter=20, cv=3,
                                               scoring='r2', n_jobs=-1, random_state=42)
                    tuner.fit(X_train, y_train)
                    tuned_r2 = tuner.score(X_test, y_test)
                    if tuned_r2 > original_r2:
                        final_model = tuner.best_estimator_
                        model_info['r2_score'] = tuned_r2
                        st.session_state.tuning_summary.append(
                            {"model_id": model_id, "status": "Improved", "original_r2": original_r2, "new_r2": tuned_r2,
                             "best_params": tuner.best_params_})
                    else:
                        final_model = st.session_state.trained_models_adv_reg[model_id]
                        st.session_state.tuning_summary.append(
                            {"model_id": model_id, "status": "Not Improved", "original_r2": original_r2,
                             "new_r2": tuned_r2})
                else:
                    final_model = st.session_state.trained_models_adv_reg[model_id]
                    st.session_state.tuning_summary.append(
                        {"model_id": model_id, "status": "No Tuning Grid", "original_r2": original_r2,
                         "new_r2": original_r2})
                st.session_state.final_best_models.append({'info': model_info, 'model': final_model})
            progress_text.success("✅ All tuning steps are complete!");
            status_placeholder.empty()

            st.subheader("📊 Hyperparameter Tuning Results")
            st.info(
                "Here is the information about the Tuned model and original model.")

            for result in st.session_state.tuning_summary:
                with st.expander(f"**{result['model_id']}** - Status: {result['status']}"):
                    cols = st.columns(3)
                    # hanged '_r2'
                    original_r2_val = result['original_r2']
                    new_r2_val = result['new_r2']

                    cols[0].metric("Original R² Score", f"{original_r2_val:.4f}")

                    if result['status'] == "Improved":
                        #  Update metric labels and deltas for R-squared
                        cols[1].metric("Tuned R² Score", f"{new_r2_val:.4f}",
                                       delta=f"{new_r2_val - original_r2_val:.4f}")
                        cols[2].write("**Best Parameters:**")
                        cols[2].json(result['best_params'])
                    elif result['status'] == "Not Improved":
                        # pdate metric labels and deltas for R-squared
                        cols[1].metric("Tuning Attempt", f"{new_r2_val:.4f}",
                                       delta=f"{new_r2_val - original_r2_val:.4f}")
                        cols[2].warning("Original model was better and has been kept.")
                    else:
                        cols[1].metric("Tuned R² Score", "N/A")
                        cols[2].info("This model was not tuned.")

            st.session_state.voting_adv_reg = "active"

        if st.session_state.voting_adv_reg == "active":
            st.header("🤹🏻 Voting Regressor")
            with st.spinner("Building Voting Ensembles..."):
                final_models_info = [m['info'] for m in st.session_state.final_best_models]
                models_by_set = {
                    i: sorted([m for m in final_models_info if m['dataset_index'] == i], key=lambda x: x['r2_score'],
                              reverse=True) for i in range(1, 4)}
                datasets = {
                    1: (st.session_state.X_train1_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test1_adv_reg, st.session_state.y_test_adv_reg),
                    2: (st.session_state.X_train2_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test2_adv_reg, st.session_state.y_test_adv_reg),
                    3: (st.session_state.X_train3_adv_reg, st.session_state.y_train_adv_reg,
                        st.session_state.X_test3_adv_reg, st.session_state.y_test_adv_reg)
                }
                st.session_state.voting_results_adv_reg = []
                for i in range(1, 4):
                    models_for_set = models_by_set[i]
                    if len(models_for_set) < 2: continue
                    estimators = [(m['model_id'], st.session_state.trained_models_adv_reg[m['model_id']]) for m in
                                  models_for_set if
                                  not m["model_id"].startswith("MLP") and not m["model_id"].startswith("DNN")]
                    if len(estimators) > 1:
                        X_train, y_train, X_test, y_test = datasets[i]
                        voter = VotingRegressor(estimators=estimators)
                        voter.fit(X_train, y_train)
                        score = voter.score(X_test, y_test)
                        st.session_state.voting_results_adv_reg.append(
                            {"name": f"Voter | Set {i} (All Models)", "r2_score": score,
                             "improvement": score - models_for_set[0]['r2_score'],
                             "best_single_model_r2": models_for_set[0]['r2_score'],
                             "components": [m['model_id'] for m in models_for_set]})
            st.subheader("🏁 Voting Ensemble Results")
            for result in st.session_state.voting_results_adv_reg:
                with st.expander(f"**{result['name']}** - Final R² Score: {result['r2_score']:.4f}", expanded=True):
                    col1, col2 = st.columns([1, 2]);
                    col1.metric("Ensemble R² Score", f"{result['r2_score']:.4f}",
                                delta=f"{result['improvement']:.4f} vs. Best Single Model");
                    col2.write("**Models Used:**");
                    col2.dataframe(pd.DataFrame(result['components'], columns=["Model ID"]))
            st.session_state.stacking_adv_reg = "active"

        if st.session_state.stacking_adv_reg == "active":
            st.header("👑 Stacking Ensemble: The Super Models")
            with st.spinner("Building Stacking Ensemble..."):
                datasets = {
                    1: (st.session_state.X_train1_adv_reg, st.session_state.X_test1_adv_reg),
                    2: (st.session_state.X_train2_adv_reg, st.session_state.X_test2_adv_reg),
                    3: (st.session_state.X_train3_adv_reg, st.session_state.X_test3_adv_reg)
                }
                y_train, y_test = st.session_state.y_train_adv_reg, st.session_state.y_test_adv_reg
                meta_features_train_list, meta_features_test_list = [], []
                final_models = st.session_state.final_best_models
                progress_bar_stack = st.progress(0, text="Generating meta-features...")

                for i, model_dict in enumerate(final_models):
                    model_obj, model_info = model_dict['model'], model_dict['info']
                    dataset_idx, model_name = model_info['dataset_index'], model_info['model_type']
                    if model_name in ["MLP", "DNN"]: continue
                    X_train, X_test = datasets[dataset_idx]

                    # Handle different attribute names for feature names
                    if hasattr(model_obj, 'feature_names_in_'):
                        # For scikit-learn models
                        feature_names = model_obj.feature_names_in_
                    else:
                        # For CatBoost models
                        feature_names = model_obj.feature_names_

                    X_train_aligned = X_train.reindex(columns=feature_names, fill_value=0)
                    X_test_aligned = X_test.reindex(columns=feature_names, fill_value=0)

                    # Generate out-of-fold predictions for training data to prevent leakage
                    train_preds = cross_val_predict(model_obj, X_train_aligned, y_train, cv=5, n_jobs=-1)
                    test_preds = model_obj.predict(X_test_aligned)
                    meta_features_train_list.append(train_preds.reshape(-1, 1))
                    meta_features_test_list.append(test_preds.reshape(-1, 1))
                    progress_bar_stack.progress((i + 1) / len(final_models),
                                                text=f"Generated from: {model_info['model_id']}")

                meta_X_train = np.hstack(meta_features_train_list)
                meta_X_test = np.hstack(meta_features_test_list)
                st.session_state.stacking_results_adv_reg = []

                # Meta-learners
                meta_xgb = XGBRegressor(n_jobs=-1, eval_metric='rmse', random_state=42)
                meta_xgb.fit(meta_X_train, y_train)
                preds_xgb = meta_xgb.predict(meta_X_test)
                st.session_state.stacking_results_adv_reg.append(
                    {'name': 'Stacker (XGB Meta)', 'r2_score': r2_score(y_test, preds_xgb)})

                meta_lr = LinearRegression(n_jobs=-1)
                meta_lr.fit(meta_X_train, y_train)
                preds_lr = meta_lr.predict(meta_X_test)
                st.session_state.stacking_results_adv_reg.append(
                    {'name': 'Stacker (Linear Meta)', 'r2_score': r2_score(y_test, preds_lr)})
                progress_bar_stack.empty()

            st.subheader("📊 Stacking Ensemble Performance")
            df = pd.DataFrame(st.session_state.stacking_results_adv_reg).sort_values('r2_score', ascending=False)
            st.dataframe(df.style.background_gradient(cmap='viridis').format({'r2_score': '{:.4f}'}),
                         use_container_width=True)

            # --- Final Top 3 Models Overall ---
            st.header("🎉 Overall Analysis Complete: The Final Podium")
            st.balloons()
            with st.spinner("Compiling all results..."):
                all_results = []
                for res in st.session_state.tuning_summary:
                    all_results.append({'name': res['model_id'],
                                        'r2_score': res['new_r2'] if res['status'] == 'Improved' else res[
                                            'original_r2'], 'type': 'Tuned Single Model'})
                for res in st.session_state.voting_results_adv_reg:
                    all_results.append({'name': res['name'], 'r2_score': res['r2_score'], 'type': 'Voting Ensemble'})
                for res in st.session_state.stacking_results_adv_reg:
                    all_results.append({'name': res['name'], 'r2_score': res['r2_score'], 'type': 'Stacking Ensemble'})
                overall_df = pd.DataFrame(all_results).sort_values('r2_score', ascending=False).reset_index(drop=True)
                top_3 = overall_df.head(3)

            st.subheader("🥇🥈🥉 Top 3 Performing Models")
            cols, emojis = st.columns(3), ["🥇 Gold", "🥈 Silver", "🥉 Bronze"]
            for i, row in top_3.iterrows():
                with cols[i]:
                    st.markdown(f"<h3 style='text-align: center; color: #FFD700;'>{emojis[i]}</h3>",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='step-container' style='text-align: center; border-color: #FFD700;'><h4>{row['name']}</h4><p>Type: <b>{row['type']}</b></p><h2 style='color: #33FFC7;'>{row['r2_score']:.4f}</h2><p>R-squared Score</p></div>",
                        unsafe_allow_html=True)