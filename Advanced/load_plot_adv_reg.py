import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def line_plot(df):

    st.write("You can select any two features for Lineplotting")


    num_cat = ["Choices"] + list(df.select_dtypes(include=[np.number]).columns)
    alp_cat = ["Choices"] + [col for col in df.columns if
                             df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])]


    if len(num_cat) > 2:
        col1, col2 = st.columns(2)
        with col1:
            selected_x = st.selectbox("Select X coordinate", num_cat, key='line_x')
        with col2:
            selected_y = st.selectbox("Select Y coordinate", num_cat, key='line_y')

        huee, colorss = None, None
        box = False

        if len(alp_cat) > 1:
            box = st.checkbox("Tick to activate hue", key='line_hue_check')
            if box:
                palette_list = ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Dark2', 'colorblind', 'viridis',
                                'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'rocket', 'mako',
                                'coolwarm', 'bwr', 'seismic', 'RdBu', 'Spectral', 'vlag']
                col1_hue, col2_hue = st.columns(2)
                with col1_hue:
                    st.write("🏹 See chart multiple categories in one click")
                    huee = st.selectbox("Select One", options=alp_cat, key='line_hue_select')
                with col2_hue:
                    st.write("❄️ Change color theme of chart")
                    colorss = st.selectbox("Select Pallete", options=palette_list, key='line_palette_select')
        else:
            st.info("ℹ️ Not enough categorical features available to use the 'hue' option.")


        if st.button("Generate Line Plot", key='line_submit'):
            if selected_x != "Choices" and selected_y != "Choices":
                fig, ax = plt.subplots()

                if box and huee is not None and huee != "Choices":
                    sns.lineplot(x=selected_x, y=selected_y, hue=huee, data=df, palette=colorss, ax=ax)
                else:
                    sns.lineplot(x=selected_x, y=selected_y, data=df, color='red', ax=ax)
                plt.title(f"{selected_x} vs {selected_y}")
                plt.xticks(rotation=90)
                st.pyplot(fig)
            else:
                st.error("❌ Please select both X and Y coordinates.")
    else:
        st.warning("⚠️ Not enough numerical features to for plotting a line chart.")


def scatter_plot(df):

    st.write("You can select any two features for scatterplotting")


    num_cat = ["Choices"] + list(df.select_dtypes(include=[np.number]).columns)
    alp_cat = ["Choices"] + [col for col in df.columns if
                             df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])]


    if len(num_cat) > 2:
        col1, col2 = st.columns(2)
        with col1:
            selected_x = st.selectbox("Select X coordinate", num_cat, key='scatter_x')
        with col2:
            selected_y = st.selectbox("Select Y coordinate", num_cat, key='scatter_y')

        huee, colorss = None, None
        box = False


        if len(alp_cat) > 1:
            box = st.checkbox("Tick to activate hue", key='scatter_hue_check')
            if box:
                palette_list = ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Dark2', 'colorblind', 'viridis',
                                'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'rocket', 'mako',
                                'coolwarm', 'bwr', 'seismic', 'RdBu', 'Spectral', 'vlag']
                col1_hue, col2_hue = st.columns(2)
                with col1_hue:
                    st.write("🏹 See chart multiple categories in one click")
                    huee = st.selectbox("Select One", options=alp_cat, key='scatter_hue_select')
                with col2_hue:
                    st.write("❄️ Change color theme of chart")
                    colorss = st.selectbox("Select Pallete", options=palette_list, key='scatter_palette_select')
        else:
            st.info("ℹ️ Not enough categorical features available to use the 'hue' option.")


        if st.button("Generate Scatter Plot", key='scatter_submit'):
            if selected_x != "Choices" and selected_y != "Choices":
                fig, ax = plt.subplots()

                if box and huee is not None and huee != "Choices":
                    sns.scatterplot(x=selected_x, y=selected_y, hue=huee, data=df, palette=colorss, ax=ax)
                else:
                    sns.scatterplot(x=selected_x, y=selected_y, data=df, color='green', ax=ax)
                plt.title(f"{selected_x} vs {selected_y}")
                plt.xticks(rotation=90)
                st.pyplot(fig)
            else:
                st.error("❌ Please select both X and Y coordinates.")
    else:
        st.warning("⚠️ Not enough numerical features for plotting a scatter chart.")



def histogram_plot(df):

    st.write("You can select any one numerical feature for a histogram.")


    num_cat = ["Choices"] + list(df.select_dtypes(include=[np.number]).columns)
    alp_cat = ["Choices"] + [col for col in df.columns if
                             df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])]


    if len(num_cat) > 1:
        selected = st.selectbox("Select a feature", num_cat, key='hist_x')

        box = False
        huee, colorss = None, None


        if len(alp_cat) > 1:
            box = st.checkbox("Tick to activate hue", key='hist_hue_check')
            if box:
                palette_list = ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Dark2', 'colorblind', 'viridis',
                                'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'rocket', 'mako',
                                'coolwarm', 'bwr', 'seismic', 'RdBu', 'Spectral', 'vlag']
                col1, col2 = st.columns(2)
                with col1:
                    st.write("🏹 Split the distribution by a category.")
                    huee = st.selectbox("Select Hue", options=alp_cat, key='hist_hue_select')
                with col2:
                    st.write("❄️ Change the color theme.")
                    colorss = st.selectbox("Select Palette", options=palette_list, key='hist_palette_select')
        else:
            st.info("ℹ️ Not enough categorical features available to use the 'hue' option.")


        if st.button("Generate Histogram", key='hist_submit'):
            if selected != "Choices":
                fig, ax = plt.subplots()

                if box and huee is not None and huee != "Choices":
                    sns.histplot(x=selected, hue=huee, data=df, palette=colorss, element="step", ax=ax)
                else:
                    sns.histplot(x=selected, data=df, color="orange", kde=True, ax=ax)
                plt.title(f"Distribution of: {selected}")
                plt.xticks(rotation=90)
                st.pyplot(fig)
            else:
                st.error("❌ Please select a feature.")
    else:
        st.warning("⚠️ Not enough numerical features to plot a histogram.")


def barchart_plot(df):

    st.write("You can select one categorical and one numerical feature.")


    num_cols = ["Choices"] + list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = ["Choices"] + [col for col in df.columns if
                              df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])]


    if len(num_cols) > 1 and len(cat_cols) > 1:
        col1, col2 = st.columns(2)
        with col1:
            selected_x = st.selectbox("Select X coordinate (Categorical)", cat_cols, key='bar_x')
        with col2:
            selected_y = st.selectbox("Select Y coordinate (Numerical)", num_cols, key='bar_y')

        box = False
        huee, colorss = None, None

        if len(cat_cols) > 2:
            box = st.checkbox("Tick to activate hue", key='bar_hue_check')
            if box:
                palette_list = ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Dark2', 'colorblind', 'viridis',
                                'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'rocket', 'mako',
                                'coolwarm', 'bwr', 'seismic', 'RdBu', 'Spectral', 'vlag']
                c1, c2 = st.columns(2)
                with c1:
                    st.write("🏹 Split bars by another category.")

                    hue_options = [c for c in cat_cols if c != selected_x]
                    huee = st.selectbox("Select Hue", options=hue_options, key='bar_hue_select')
                with c2:
                    st.write("❄️ Change the color theme.")
                    colorss = st.selectbox("Select Palette", options=palette_list, key='bar_palette_select')
        else:
            st.info("ℹ️ You need at least two categorical features to use the 'hue' option.")


        if st.button("Generate Bar Plot", key='bar_submit'):
            if selected_x != "Choices" and selected_y != "Choices":
                fig, ax = plt.subplots()
                if box and huee is not None and huee != "Choices":
                    sns.barplot(x=selected_x, y=selected_y, hue=huee, data=df, palette=colorss, ax=ax)
                else:
                    sns.barplot(x=selected_x, y=selected_y, data=df, color="purple", ax=ax)
                plt.title(f"Bar Plot of {selected_x} vs {selected_y}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.error("❌ Please select both X and Y coordinates.")
    else:
        st.warning("⚠️ A bar chart requires at least one numerical and one categorical feature.")


def piechart_plot(df):

    st.write("You can select any one categorical feature for a Pie Chart")

    cat_cols = ["Choices"] + df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(cat_cols) > 1:
        selected_col = st.selectbox("Select a feature", cat_cols, key='pie_select')

        palette_list = ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Dark2', 'colorblind', 'viridis',
                        'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'rocket',
                        'mako', 'coolwarm', 'bwr', 'seismic', 'RdBu', 'Spectral', 'vlag']
        selected_palette = st.selectbox("Select a color palette", options=palette_list, key='pie_palette_select')

        if st.button("Generate Pie Chart", key='pie_submit'):
            if selected_col != "Choices":
                value_counts = df[selected_col].value_counts()
                fig, ax = plt.subplots()
                ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90,
                       colors=sns.color_palette(selected_palette, n_colors=len(value_counts)))
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                plt.title(f"Proportion of: {selected_col}")
                st.pyplot(fig)
            else:
                st.error("❌ Please select a feature.")
    else:
        st.warning("⚠️ A pie chart requires at least one categorical feature.")


def countchart_plot(df):

    st.write("You can select a categorical feature to see its count.")

    num_cat = df.select_dtypes(include=[np.number]).columns
    cat_cols = ["Choices"] + [col for col in df.columns if col not in num_cat]

    if len(cat_cols) > 1:
        selected = st.selectbox("Select a feature to count", cat_cols, key='count_x')

        box = False
        huee, colorss = None, None

        if len(cat_cols) > 2:
            box = st.checkbox("Tick to activate hue", key='count_hue_check')
            if box:
                palette_list = ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Dark2', 'colorblind', 'viridis',
                                'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'rocket', 'mako',
                                'coolwarm', 'bwr', 'seismic', 'RdBu', 'Spectral', 'vlag']
                col1, col2 = st.columns(2)
                with col1:
                    st.write("🏹 Split the count by another category.")

                    hue_options = [c for c in cat_cols if c != selected]
                    huee = st.selectbox("Select Hue", options=hue_options, key='count_hue_select')
                with col2:
                    st.write("❄️ Change the color theme.")
                    colorss = st.selectbox("Select Palette", options=palette_list, key='count_palette_select')
        else:
            st.info("ℹ️ You need at least two categorical features to use the 'hue' option.")

        if st.button("Generate Count Plot", key='count_submit'):
            if selected != "Choices":
                fig, ax = plt.subplots()
                if box and huee is not None and huee != "Choices":
                    sns.countplot(x=selected, hue=huee, data=df, palette=colorss, ax=ax)
                else:
                    sns.countplot(x=selected, data=df, palette=[sns.color_palette("Set2")[0]], ax=ax)
                plt.title(f"Count of: {selected}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.error("❌ Please select a feature.")
    else:
        st.warning("⚠️ A count plot requires at least one categorical feature.")


def correlation_plot(df):

    st.write("↘️ Get ready to make a correlation matrix for numerical features.")


    num_cols = df.select_dtypes(include=[np.number]).columns

    if len(num_cols) >= 2:
        if st.button("Generate Correlation Heatmap", key='corr_submit'):
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8)) # Adjust figure size for better readability
            sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=.25, fmt=".2f", ax=ax)
            plt.title("Correlation Matrix")
            st.pyplot(fig)
    else:
        st.warning("⚠️ You need at least two numerical columns to create a correlation matrix.")


def kde_plot(df):

    st.write("You can select any one numerical feature for a KDE plot.")

    num_cat = ["Choices"] + list(df.select_dtypes(include=[np.number]).columns)
    alp_cat = ["Choices"] + [col for col in df.columns if col not in num_cat]

    if len(num_cat) > 1:
        selected_x = st.selectbox("Select a feature", num_cat, key='kde_x')

        box = False
        huee, colorss = None, None

        if len(alp_cat) > 1:
            box = st.checkbox("Tick to activate hue", key='kde_hue_check')
            if box:
                palette_list = ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Dark2', 'colorblind', 'viridis',
                                'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'rocket',
                                'mako',
                                'coolwarm', 'bwr', 'seismic', 'RdBu', 'Spectral', 'vlag']
                col1, col2 = st.columns(2)
                with col1:
                    st.write("🏹 Split the density plot by a category.")
                    huee = st.selectbox("Select Hue", options=alp_cat, key='kde_hue_select')
                with col2:
                    st.write("❄️ Change the color theme of the chart.")
                    colorss = st.selectbox("Select Palette", options=palette_list, key='kde_palette_select')
        else:
            st.info("ℹ️ Not enough categorical features available to use the 'hue' option.")

        if st.button("Generate KDE Plot", key='kde_submit'):
            if selected_x != "Choices":
                fig, ax = plt.subplots()

                if box and huee is not None and huee != "Choices":
                    sns.kdeplot(x=selected_x, hue=huee, data=df, palette=colorss, ax=ax, fill=True)
                else:
                    sns.kdeplot(x=selected_x, data=df, fill=True, color='green', ax=ax)
                plt.title(f"KDE Plot of {selected_x}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.error("❌ Please select a feature.")
    else:
        st.warning("⚠️ A KDE plot requires at least one numerical feature.")


def box_plot(df):

    st.write("You can select one categorical and one numerical feature or just a single numerical feature.")

    num_cols = ["Choices"] + list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = ["Choices"] + [col for col in df.columns if col not in num_cols]

    if len(num_cols) > 1:
        selected_x = "Choices"
        col1, col2 = st.columns(2)

        if len(cat_cols) > 1:
            with col1:
                selected_x = st.selectbox("Select X coordinate (Categorical)", cat_cols, key='box_x')

        with col2:
            selected_y = st.selectbox("Select Y coordinate (Numerical)", num_cols, key='box_y')


        box = False
        huee, colorss = None, None
        if len(cat_cols) > 2:
            box = st.checkbox("Tick to activate hue", key='box_hue_check')
            if box:
                palette_list = ['tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Pastel1', 'Dark2', 'colorblind', 'viridis',
                                'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'rocket', 'mako',
                                'coolwarm', 'bwr', 'seismic', 'RdBu', 'Spectral', 'vlag']
                c1, c2 = st.columns(2)
                with c1:
                    st.write("🏹 Split boxes by another category.")

                    hue_options = [c for c in cat_cols if c != selected_x]
                    huee = st.selectbox("Select Hue", options=hue_options, key='box_hue_select')
                with c2:
                    st.write("❄️ Change the color theme.")
                    colorss = st.selectbox("Select Palette", options=palette_list, key='box_palette_select')
        else:
            st.info("ℹ️ You need at least two categorical features to use the 'hue' option.")

        if st.button("Generate Box Plot", key='box_submit'):
            fig, ax = plt.subplots()

            if selected_x != "Choices" and selected_y != "Choices":
                if box and huee is not None and huee != "Choices":
                    sns.boxplot(x=selected_x, y=selected_y, hue=huee, data=df, palette=colorss, ax=ax)
                else:
                    sns.boxplot(x=selected_x, y=selected_y, data=df, palette="Set2", ax=ax)
                plt.title(f"Box Plot of {selected_y} by {selected_x}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            elif selected_x == "Choices" and selected_y != "Choices":
                if box and huee is not None and huee != "Choices":

                    sns.boxplot(y=selected_y, x=huee, data=df, palette=colorss, ax=ax)
                    plt.title(f"Box Plot of {selected_y} grouped by {huee}")
                else:
                    sns.boxplot(y=selected_y, data=df, color='purple', ax=ax)
                    plt.title(f"Box Plot of {selected_y}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.error("❌ Please select at least a Y coordinate.")
    else:
        st.warning("⚠️ A box plot requires at least one numerical feature.")


def information(df):
    num_features = df.shape[1]
    num_rows = df.shape[0]
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()

    st.header("Data Summary")
    st.write("↘️ Here is a short overview of your data and its internal structure")
    with st.expander("📊 View Detailed Analysis", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("No. of Features", num_features)
        col2.metric("No. of Rows", num_rows)
        col3.metric("Total Missing Values", missing_values)
        col4, col5, col6 = st.columns(3)
        col4.metric("Duplicate Rows", duplicate_rows)
        col5.metric("Categorical Features", len(categorical_features))
        col6.metric("Numerical Features", len(numerical_features))

    st.subheader("💁🏻 Data Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("➡️ Numerical columns detail:")
        with st.expander("Show Numerical Features"):
            if(len(numerical_features) > 0):
                for i, col in enumerate(numerical_features):
                    st.caption(f"{i + 1} -- {col}")
            else:
                st.caption("No numerical features available.")
    with col2:
        st.write("➡️ Categorical columns detail:")
        with st.expander("Show Categorical Features"):
            if(len(categorical_features) > 0):
                for i, col in enumerate(categorical_features):
                    st.caption(f"{i + 1} -- {col}")
            else:
                st.caption("No categorical features available.")

    st.write("➡️ Check missing values per column:")
    with st.expander("Perform Missing Value Analysis"):
        if(df.isna().sum().sum() > 0):
            column_list = df.columns.tolist()
            new_columns_list = ["--Choose a feature--"]
            for i in column_list:
                if(df[i].isnull().sum() > 0):
                    new_columns_list.append(i)
            selected_column = st.selectbox("Select a feature", options=new_columns_list, key='missing_value_select')
            if selected_column != "--Choose a feature--":
                missing_count = int(df[selected_column].isnull().sum())
                st.metric(f"Missing Values in '{selected_column}'", missing_count)
        else:
            st.caption("No missing values in dataset.")

    st.subheader("📊 Data Visualization")
    with st.expander("Enter Visualization Mode"):
        st.write("📈 Visualize different charts to study your dataset in depth.")

        tab_list = ["Line", "Scatter", "Histogram", "Bar", "Pie", "Count", "Correlation", "KDE", "Box"]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(tab_list)

        with tab1:
            st.header("Line Chart")
            line_plot(df)
        with tab2:
            st.header("Scatter Plot")
            scatter_plot(df)
        with tab3:
            st.header("Histogram")
            histogram_plot(df)
        with tab4:
            st.header("Bar Chart")
            barchart_plot(df)
        with tab5:
            st.header("Pie Chart")
            piechart_plot(df)
        with tab6:
            st.header("Count Chart")
            countchart_plot(df)
        with tab7:
            st.header("Correlation Chart")
            correlation_plot(df)
        with tab8:
            st.header("KDE Chart")
            kde_plot(df)
        with tab9:
            st.header("Box Chart")
            box_plot(df)

