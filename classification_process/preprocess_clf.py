import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler
import time


def reset_preprocessing_state():
    """Clears all session state variables related to the preprocessing workflow."""
    keys_to_delete = [
        'preprocess_stage', 'df_cleaned', 'fill_step_skipped',
        'new_feature_definitions', 'X_train_split', 'X_test_split',
        'y_train_split', 'y_test_split', 'X_train_filled', 'X_test_filled',
        'X_train_featured', 'X_test_featured', 'X_train_processed',
        'X_test_processed', 'y_train_processed', 'y_test_processed'
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    # Re-initialize to the starting stage
    initialize_state_preprocess()


# ======================================================================================
# 1. INITIALIZATION: Central state management for the entire workflow
# ======================================================================================

def initialize_state_preprocess():
    """Initializes all session state variables for the preprocessing workflow."""
    keys_to_initialize = {
        'preprocess_stage': 'split', 'df_cleaned': None, 'fill_step_skipped': False,
        'new_feature_definitions': [],
        'X_train_split': None, 'X_test_split': None, 'y_train_split': None, 'y_test_split': None,
        'X_train_filled': None, 'X_test_filled': None,
        'X_train_featured': None, 'X_test_featured': None,
        'X_train_processed': None, 'X_test_processed': None, 'y_train_processed': None, 'y_test_processed': None
    }
    for key, default_value in keys_to_initialize.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ======================================================================================
# 2. HELPER FUNCTIONS FOR EACH UI TASK
# ======================================================================================

def display_split_ui(df):
    """Shows the UI for splitting the data, including a column dropper."""
    st.write("First, you can remove unwanted columns. Then, split the data.")
    with st.expander("Remove Unwanted Columns (Optional)"):
        cols_to_drop = st.multiselect("Select columns to remove (e.g., ID, name columns):", df.columns.tolist(),
                                      key='drop_cols')
        if st.button("Apply Column Removal", key="remove_columns_btn"):
            st.session_state.df_cleaned = df.drop(columns=cols_to_drop)
            st.success(f"Removed {len(cols_to_drop)} columns.")
            time.sleep(1)
            st.rerun()
    st.divider()
    active_df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else df
    if st.session_state.df_cleaned is not None:
        st.info(f"Proceeding with {active_df.shape[1]} columns.")
    with st.expander("Perform Data Splitting", expanded=True):
        method = st.selectbox("Choose splitting method", ["--Choose--", "Standard Split", "Positional Split"])
        if method == "Standard Split":
            _split_standard(active_df)
        elif method == "Positional Split":
            _split_positional(active_df)


def _split_standard(df):
    """UI and logic for a standard scikit-learn train-test split."""
    st.write("Using `train_test_split` with optional stratification.")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key='std_split_size')
    with col2:
        target = st.selectbox("Target Column", df.columns, key='std_split_target')
    with col3:
        stratify_options = ["None"] + df.select_dtypes(include=['object', 'category']).columns.tolist()
        stratify_feature = st.selectbox("Stratify By (optional)", stratify_options, key='std_split_stratify')
    if st.button("Perform Standard Split", key="perform_std_split"):
        stratify_data = None
        if stratify_feature != "None":
            if df[stratify_feature].notna().all():
                stratify_data = df[stratify_feature]
            else:
                st.error(f"Column '{stratify_feature}' has missing values and cannot be used for stratification.")
                return
        with st.spinner("Splitting..."):
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_data,
                                                                random_state=42)
            st.session_state.update(
                {'X_train_split': X_train, 'X_test_split': X_test, 'y_train_split': y_train, 'y_test_split': y_test})
            st.success("Splitting complete!")
            time.sleep(1)
            st.rerun()


def _split_positional(df):
    """UI and logic for a simple positional split."""
    st.write("Splitting data based on its order (e.g., for time series).")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key='pos_split_size')
    with col2:
        target = st.selectbox("Target Column", df.columns, key='pos_split_target')
    if st.button("Perform Positional Split", key="perform_pos_split"):
        slicer = int(df.shape[0] * (1 - test_size))
        X = df.drop(target, axis=1)
        y = df[target]
        st.session_state.update(
            {'X_train_split': X.iloc[:slicer], 'X_test_split': X.iloc[slicer:], 'y_train_split': y.iloc[:slicer],
             'y_test_split': y.iloc[slicer:]})
        st.success("Splitting complete!")
        time.sleep(1)
        st.rerun()


def display_fill_ui():
    """Shows the UI for filling missing values."""
    st.write("Choose a method to fill any missing values found in your training and test sets.")
    filler_choice = st.radio("Select Filling Method:", ["Auto Filler", "Manual Filler"], horizontal=True, index=None,
                             key="fill_choice")

    if filler_choice == "Auto Filler":
        st.info("Using Median for numerical and Mode for categorical features.")
        if st.button("Apply Auto Filler", key="apply_auto_filler"):
            with st.spinner("Applying Auto Filler..."):
                X_train_filled = st.session_state.X_train_split.copy()
                X_test_filled = st.session_state.X_test_split.copy()
                numerical_features = X_train_filled.select_dtypes(include=np.number).columns
                categorical_features = X_train_filled.select_dtypes(exclude=np.number).columns

                if len(numerical_features) > 0:
                    imputer_num = SimpleImputer(strategy='median')
                    X_train_filled[numerical_features] = imputer_num.fit_transform(X_train_filled[numerical_features])
                    X_test_filled[numerical_features] = imputer_num.transform(X_test_filled[numerical_features])
                if len(categorical_features) > 0:
                    imputer_cat = SimpleImputer(strategy='most_frequent')
                    X_train_filled[categorical_features] = imputer_cat.fit_transform(
                        X_train_filled[categorical_features])
                    X_test_filled[categorical_features] = imputer_cat.transform(X_test_filled[categorical_features])
                st.session_state.update({'X_train_filled': X_train_filled, 'X_test_filled': X_test_filled})
                st.success("Auto filling complete!")

                st.session_state.X_train_filled.drop_duplicates(inplace=True)
                st.session_state.X_test_filled.drop_duplicates(inplace=True)
                st.session_state.y_train_split = st.session_state.y_train_split.loc[st.session_state.X_train_filled.index]
                st.session_state.y_test_split = st.session_state.y_test_split.loc[st.session_state.X_test_filled.index]
                time.sleep(1)
                st.rerun()

    elif filler_choice == "Manual Filler":
        st.info("Choose your strategies below. The operation will apply to all relevant columns.")
        col1, col2 = st.columns(2)
        with col1:
            num_strategy = st.selectbox("Numerical Strategy", ["mean", "median", "most_frequent"], key="num_strategy")
        with col2:
            cat_strategy = st.selectbox("Categorical Strategy", ["most_frequent"], key="cat_strategy")
        if st.button("Apply Manual Filler", key="apply_manual_filler"):
            with st.spinner("Applying Manual Filler..."):
                X_train_filled = st.session_state.X_train_split.copy()
                X_test_filled = st.session_state.X_test_split.copy()
                numerical_features = X_train_filled.select_dtypes(include=np.number).columns
                categorical_features = X_train_filled.select_dtypes(exclude=np.number).columns
                if len(numerical_features) > 0:
                    imputer_num = SimpleImputer(strategy=num_strategy)
                    X_train_filled[numerical_features] = imputer_num.fit_transform(X_train_filled[numerical_features])
                    X_test_filled[numerical_features] = imputer_num.transform(X_test_filled[numerical_features])
                if len(categorical_features) > 0:
                    imputer_cat = SimpleImputer(strategy=cat_strategy)
                    X_train_filled[categorical_features] = imputer_cat.fit_transform(
                        X_train_filled[categorical_features])
                    X_test_filled[categorical_features] = imputer_cat.transform(X_test_filled[categorical_features])
                st.session_state.update({'X_train_filled': X_train_filled, 'X_test_filled': X_test_filled})
                st.success("Manual filling complete!")

                st.session_state.X_train_filled.drop_duplicates(inplace=True)
                st.session_state.X_test_filled.drop_duplicates(inplace=True)
                st.session_state.y_train_split = st.session_state.y_train_split.loc[st.session_state.X_train_filled.index]
                st.session_state.y_test_split = st.session_state.y_test_split.loc[st.session_state.X_test_filled.index]
                time.sleep(1)
                st.rerun()


def display_feature_ui():
    """Shows the UI for creating new features. Its ONLY job is to let the user define them."""
    st.write("Create new features from existing ones.")

    if 'X_train_featured' in st.session_state and st.session_state.X_train_featured is not None:
        X_df = st.session_state.X_train_featured
    elif 'X_train_filled' in st.session_state and st.session_state.X_train_filled is not None:
        X_df = st.session_state.X_train_filled
    else:
        X_df = st.session_state.X_train_split

    if X_df is None:
        st.warning("Data not yet available for feature engineering.")
        return

    feature_type = st.radio(
        "What type of feature do you want to create?",
        ["Numerical-to-Numerical", "Categorical-to-Categorical"],
        horizontal=True, index=None, key="feature_type"
    )

    if feature_type == "Numerical-to-Numerical":
        num_cols = X_df.select_dtypes(include=np.number).columns.tolist()
        cols_to_combine = st.multiselect("Select 2 to 4 numerical features to combine", num_cols, max_selections=4,
                                         key="num_feat_combine")
        if len(cols_to_combine) >= 2:
            operands = []
            st.write("#### Define the operation:")
            ui_columns = st.columns(len(cols_to_combine) * 2 - 1)
            for i, col_name in enumerate(cols_to_combine):
                ui_columns[i * 2].write(f"<p style='text-align: center; margin-top: 2rem;'><b>`{col_name}`</b></p>",
                                        unsafe_allow_html=True)
                if i < len(cols_to_combine) - 1:
                    with ui_columns[i * 2 + 1]:
                        op = st.selectbox(f"Op {i + 1}", ["+", "-", "*", "/"], key=f"operand_{i}",
                                          label_visibility="collapsed")
                        operands.append(op)
            if st.button("Add Numerical Feature Definition", key="add_num_feat_def"):
                definition = {'type': 'numerical', 'cols': cols_to_combine, 'ops': operands}
                st.session_state.new_feature_definitions.append(definition)
                st.rerun()

    elif feature_type == "Categorical-to-Categorical":
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        cols_to_combine = st.multiselect("Select 2 to 4 categorical features to combine", cat_cols, max_selections=4,
                                         key="cat_feat_combine")
        if st.button("Add Categorical Feature Definition", key="add_cat_feat_def"):
            if 2 <= len(cols_to_combine) <= 4:
                definition = {'type': 'categorical', 'cols': cols_to_combine}
                st.session_state.new_feature_definitions.append(definition)
                st.rerun()
            else:
                st.warning("Please select between 2 and 4 features.")

    st.divider()
    if st.session_state.new_feature_definitions:
        st.write("#### Defined New Features (to be applied):")
        for i, definition in enumerate(st.session_state.new_feature_definitions):
            if definition['type'] == 'numerical':
                expr_parts = [f"`{c}`" for c in definition['cols']]
                expr = expr_parts[0]
                for i, op in enumerate(definition['ops']):
                    expr += f" {op} {expr_parts[i + 1]}"
                st.write(f"- **Numerical Feature {i + 1}**: `{expr}`")
            else:
                st.write(f"- **Categorical Feature {i + 1}**: Combining `{', '.join(definition['cols'])}`")
        if st.button("Clear All Definitions", key="clear_feat_defs"):
            st.session_state.new_feature_definitions = []
            st.rerun()


# ======================================================================================
# 3. MAIN WORKFLOW CONTROLLER
# ======================================================================================

def run_preprocessing_workflow(df):
    """
    The main function to call from your parent page.
    It controls the entire multi-step workflow using a 'stage' variable.
    """
    initialize_state_preprocess()
    st.subheader(f"Step: {st.session_state.preprocess_stage.replace('_', ' ').capitalize()}")

    # --- STAGE 1: DATA SPLITTING ---
    if st.session_state.preprocess_stage == 'split':
        if st.session_state.X_train_split is None:
            display_split_ui(df)
        else:
            if ((st.session_state.y_train_split.isna().sum() > 0) or (st.session_state.y_test_split.isna().sum() > 0)):
                st.info("**Smart Feature:** Target miss value fill automatically")
                y_fill_mode=st.session_state.y_train_split.mode()[0]
                st.session_state.y_train_split.fillna(y_fill_mode, inplace=True)
                st.session_state.y_test_split.fillna(y_fill_mode, inplace=True)
            st.success("✅ Step 1 complete: Data has been split.")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Train Set Shape:**", st.session_state.X_train_split.shape)
            with col2:
                st.write("**Test Set Shape:**", st.session_state.X_test_split.shape)
            if st.button("➡️ Continue to Next Step", key="continue_to_fill"):
                if st.session_state.X_train_split.isna().sum().sum() > 0:
                    st.session_state.preprocess_stage = 'fill'
                else:
                    st.session_state.preprocess_stage = 'feature'
                    st.session_state.fill_step_skipped = True
                st.rerun()

    # --- STAGE 2: FILLING MISSING VALUES ---
    elif st.session_state.preprocess_stage == 'fill':
        if st.session_state.X_train_filled is None:
            display_fill_ui()
        else:
            st.success("✅ Step 2 complete: Missing values have been filled.")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➡️ Continue to Next Step", key="continue_to_feature"):
                # VALIDATION: Ensure data has been filled.
                if st.session_state.X_train_filled is None:
                    st.error("❌ Please apply a filling method before continuing.")
                else:
                    st.session_state.preprocess_stage = 'feature'
                    st.rerun()
        with col2:
            if st.button("⬅️ Go back to Split", key="back_to_split"):
                st.session_state.preprocess_stage = 'split'
                # Clear all subsequent states
                for key in ['X_train_split', 'X_test_split', 'y_train_split', 'y_test_split', 'df_cleaned',
                            'X_train_filled', 'X_test_filled']:
                    st.session_state[key] = None
                st.rerun()

    # --- STAGE 3: FEATURE ENGINEERING ---
    elif st.session_state.preprocess_stage == 'feature':
        if st.session_state.fill_step_skipped:
            st.info("Step 2 (Filling Missing Values) was skipped as no missing values were found.")
            st.session_state.X_train_split.drop_duplicates(inplace=True)
            st.session_state.X_test_split.drop_duplicates(inplace=True)
            st.session_state.y_train_split=st.session_state.y_train_split.loc[st.session_state.X_train_split.index]
            st.session_state.y_test_split=st.session_state.y_test_split.loc[st.session_state.X_test_split.index]
        st.info("Duplicates removed automatically")
        display_feature_ui()

        base_train_df = st.session_state.X_train_filled if st.session_state.X_train_filled is not None else st.session_state.X_train_split
        base_test_df = st.session_state.X_test_filled if st.session_state.X_test_filled is not None else st.session_state.X_test_split

        if st.session_state.new_feature_definitions:
            if st.button("Apply All New Features", key="apply_features_main"):
                with st.spinner("Generating features..."):
                    X_train_temp, X_test_temp = base_train_df.copy(), base_test_df.copy()
                    for definition in st.session_state.new_feature_definitions:
                        if definition['type'] == 'numerical':
                            cols, ops = definition['cols'], definition['ops']
                            full_expression = f"`{cols[0]}`"
                            for i, op in enumerate(ops):
                                full_expression += f" {op} (`{cols[i + 1]}` + 1e-6)" if op == '/' else f" {op} `{cols[i + 1]}`"
                            name_from_ops = '_'.join(ops).replace('*', 'mul').replace('/', 'div').replace('+',
                                                                                                          'add').replace(
                                '-', 'sub')
                            new_col_name = f"{'_'.join(cols)}_{name_from_ops}"
                            X_train_temp[new_col_name] = X_train_temp.eval(full_expression, engine='python')
                            X_test_temp[new_col_name] = X_test_temp.eval(full_expression, engine='python')
                        else:
                            cols = definition['cols']
                            new_col_name = '_'.join(map(str, cols))
                            X_train_temp[new_col_name] = X_train_temp[cols].astype(str).agg('_'.join, axis=1)
                            X_test_temp[new_col_name] = X_test_temp[cols].astype(str).agg('_'.join, axis=1)
                    st.session_state.update({'X_train_featured': X_train_temp, 'X_test_featured': X_test_temp,
                                             'new_feature_definitions': []})
                    st.success("New features created!")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➡️ Continue to Encoding", key="continue_to_encoding"):
                if st.session_state.X_train_featured is None:
                    st.session_state.X_train_featured = base_train_df
                    st.session_state.X_test_featured = base_test_df
                st.session_state.preprocess_stage = 'encode'
                st.rerun()
        with col2:
            if st.button("⬅️ Go back to Previous Step", key="back_to_fill_or_split"):
                st.session_state.preprocess_stage = 'fill' if not st.session_state.fill_step_skipped else 'split'
                for key in ['X_train_filled', 'X_test_filled', 'X_train_featured', 'X_test_featured',
                            'new_feature_definitions']:
                    st.session_state[key] = None
                st.rerun()

    # --- STAGE 4: ENCODING & SCALING ---
    elif st.session_state.preprocess_stage == 'encode':
        st.write("Encode categorical features and/or scale numerical data.")
        X_df_current = st.session_state.X_train_featured.copy()
        cat_cols = X_df_current.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X_df_current.select_dtypes(include=np.number).columns.tolist()

        ohe_cols, ord_cols, encoding_method = [], [], None
        if cat_cols:
            with st.expander("Encoding Options", expanded=True):
                encoding_method = st.radio("Select Encoding Method for Features:",
                                           ["One-Hot All", "Ordinal All", "Manual Selection"], index=None,
                                           key="encoding_method")
                if encoding_method == "Manual Selection":
                    ohe_cols = st.multiselect("Select features to One-Hot Encode:", cat_cols, key="ohe_cols")
                    ord_cols = st.multiselect("Select features to Ordinal Encode:",
                                              [c for c in cat_cols if c not in ohe_cols], key="ord_cols")
                elif encoding_method == "One-Hot All":
                    ohe_cols = cat_cols
                elif encoding_method == "Ordinal All":
                    ord_cols = cat_cols
        else:
            st.info("ℹ️ No categorical features found to encode.")

        st.divider()

        scale_data = False
        if num_cols:
            scale_data = st.checkbox("Scale numerical features (StandardScaler)", key="scale_checkbox",
                                     help="Highly recommended for distance-based models.")
        else:
            st.info("ℹ️ No numerical features found to scale.")

        st.divider()

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Apply and Finalize Data", key="finalize_data_button"):
                if cat_cols and not encoding_method:
                    st.error("❌ Please select an encoding method for your categorical features before finalizing.")
                else:
                    with st.spinner("Processing data..."):
                        X_train_final, X_test_final = st.session_state.X_train_featured.copy(), st.session_state.X_test_featured.copy()
                        y_train_final, y_test_final = st.session_state.y_train_split.copy(), st.session_state.y_test_split.copy()

                        if y_train_final.dtype == 'object' or y_train_final.dtype.name == 'category':
                            le = LabelEncoder()
                            y_train_final = le.fit_transform(y_train_final)
                            y_test_final = le.transform(y_test_final)
                            st.toast("Target variable (y) has been Label-Encoded.", icon="✅")

                        if scale_data and num_cols:
                            scaler = StandardScaler()
                            X_train_final[num_cols] = scaler.fit_transform(X_train_final[num_cols])
                            X_test_final[num_cols] = scaler.transform(X_test_final[num_cols])
                            st.toast("Numerical features have been scaled.", icon="✅")

                        if ohe_cols:
                            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
                            ohe_train_data = ohe.fit_transform(X_train_final[ohe_cols])
                            ohe_test_data = ohe.transform(X_test_final[ohe_cols])
                            ohe_train_df = pd.DataFrame(ohe_train_data, index=X_train_final.index,
                                                        columns=ohe.get_feature_names_out(ohe_cols))
                            ohe_test_df = pd.DataFrame(ohe_test_data, index=X_test_final.index,
                                                       columns=ohe.get_feature_names_out(ohe_cols))
                            X_train_final = X_train_final.drop(columns=ohe_cols).join(ohe_train_df)
                            X_test_final = X_test_final.drop(columns=ohe_cols).join(ohe_test_df)
                            st.toast("Categorical features have been One-Hot Encoded.", icon="✅")
                        if ord_cols:
                            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                            X_train_final[ord_cols] = oe.fit_transform(X_train_final[ord_cols])
                            X_test_final[ord_cols] = oe.transform(X_test_final[ord_cols])
                            st.toast("Categorical features have been Ordinal Encoded.", icon="✅")

                        st.session_state.update({
                            'X_train_processed': X_train_final, 'X_test_processed': X_test_final,
                            'y_train_processed': y_train_final, 'y_test_processed': y_test_final,
                            'preprocess_stage': 'done'
                        })
                        st.rerun()
        with col2:
            if st.button("⬅️ Go back to Feature Engineering", key="back_to_feature"):
                st.session_state.preprocess_stage = 'feature'
                st.rerun()

    # --- STAGE 5: DONE ---
    elif st.session_state.preprocess_stage == 'done':
        st.info("Removing duplicates")
        st.balloons()
        st.success("🎉 Preprocessing Workflow Complete!")
        st.write("Your data is now fully preprocessed and ready for model training.")
        return (
            st.session_state.X_train_processed,
            st.session_state.X_test_processed,
            st.session_state.y_train_processed,
            st.session_state.y_test_processed
        )

    # Return None during intermediate steps
    return None, None, None, None