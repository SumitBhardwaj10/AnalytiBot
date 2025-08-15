import streamlit as st

# ----------------- PAGE SETUP -----------------
# Define each page of the Streamlit application using st.Page.

# Define the "About" page, which will serve as the default landing page.
about_page = st.Page(
    page="pagess/aboutmodel.py",
    title="About Model",
    icon="🐎",
    default=True
)

# Define the "Classification" page.
classification_page = st.Page(
    page="pagess/Classification.py",
    title="Classification Model",
    icon="🦑",
)

# Define the "Regression" page.
regression_page = st.Page(
    page="pagess/Regression.py",
    title="Regression Model",
    icon="🐦",
)

# Define the "Advanced Classification" page.
auto_classification_page = st.Page(
    page="Advanced/Adv_Classification.py",
    title="Adv Clf Model",
    icon="💪🏼"
)

# Define the "Advanced Regression" page.
auto_regression_page = st.Page(
    page="Advanced/Adv_Regression.py",
    title="Adv Reg Model",
    icon="💪🏼"
)

# ----------------- NAVIGATION SETUP -----------------
# Configure the multi-page navigation sidebar with organized sections.
pg = st.navigation({
    "Info": [about_page],
    "Models": [classification_page, regression_page],
    "Advanced Zone": [auto_classification_page, auto_regression_page]
})

# ----------------- SIDEBAR CUSTOMIZATION -----------------
# Set a custom logo at the top of the sidebar.
st.logo("assets/Sidebar.png")

# Add a custom text footer to the sidebar.
st.sidebar.text("Analysis with 💗 Sumit")

# ----------------- APP EXECUTION -----------------
# Execute the navigation logic to render the selected page.
pg.run()