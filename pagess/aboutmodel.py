import streamlit as st
import base64
import time
from forms.feedback import feedback_function

# ----------------- HELPER FUNCTIONS -----------------

# This helper function encodes image files into base64 strings.
# This allows embedding images directly into HTML/CSS, preventing reloads on navigation.
@st.cache_data
def get_base64_of_bin_file(bin_file):
    """
    Reads a binary file and returns its base64 encoded string.
    The @st.cache_data decorator ensures this function only runs once per file,
    and the result is stored in cache for subsequent reruns to improve performance.
    """
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Asset file not found: {bin_file}. Please ensure it is in the correct path.")
        return ""

# ----------------- PAGE CONFIGURATION & STYLING -----------------

# Configure the page to use a wide layout for better content organization.
st.set_page_config(layout="wide")

# Inject custom CSS for advanced styling of the application.
# This includes a video background, custom fonts, button styles, and card designs.
st.markdown(f"""
<style>
    /* --- Video Background --- */
    .stApp {{
        background: #1E1E1E; /* Fallback color */
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}

    #bg-video {{
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        z-index: -2;
    }}
    #bg-video video {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        margin: auto;
        min-width: 50%;
        min-height: 50%;
        filter: brightness(0.3); /* Darken the video to make text readable */
    }}
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5); /* Dark overlay */
        z-index: -1;
    }}

    /* --- General Styling --- */
    .main-container {{
        background: rgba(30, 30, 30, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeIn 1s ease-in-out;
    }}
    h1, h2, p, label {{
        color: #FFFFFF !important;
    }}
    h2 {{
        border-bottom: 3px solid #00C49A;
        padding-bottom: 10px;
        margin-top: 40px;
        text-align: center;
    }}

    /* --- Button Styling --- */
    div[data-testid="stButton"] > button {{
        border-radius: 10px;
        border: 2px solid #00C49A;
        background-color: transparent;
        color: #00C49A;
        transition: all 0.3s ease;
        padding: 10px 25px;
        font-weight: bold;
    }}
    div[data-testid="stButton"] > button:hover {{
        background-color: #00C49A;
        color: #1E1E1E;
        transform: scale(1.05);
    }}

    /* --- Feature Card Styling --- */
    .feature-card {{
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 100%;
    }}
    .feature-card:hover {{
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.1);
    }}
    .feature-card img {{
        height: 60px;
        margin-bottom: 1rem;
    }}
    .feature-card p {{
        font-size: 1.1rem;
        font-weight: bold;
        color: #FFFFFF !important;
    }}
    """, unsafe_allow_html=True)

# ----------------- DIALOG DEFINITIONS -----------------
# The following functions define the content for modal dialogs (pop-ups).

# Defines the dialog content for the "Overview" section.
@st.dialog("Overview")
def show_contact_form():
    st.header("✨ Key Features")
    features = {
        "Data Manipulation": "Customize preprocessing steps.",
        "Visualization": "Generate multiple types of plots.",
        "Feature Making": "Engineer new features from your data.",
        "Model Selection": "Choose from a wide array of algorithms.",
        "Model Tuning": "Optimize models with hyperparameter tuning.",
        "Model Evaluation": "Assess performance with various metrics.",
        "Stacking": "Build powerful multi-layer stacked models.",
        "Advance Modeling": "Access sophisticated ensemble techniques.",
    }
    for feature, desc in features.items():
        st.write(f"**{feature}:** {desc}")

# Defines the dialog content for the "Models Information" section.
@st.dialog("Models Information")
def models_used():
    st.header("🤖 Models Available in AnalytiBot")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Classification")
        st.write("• Logistic Regression", "• K-Nearest Neighbors", "• Support Vector Machine (SVC)", "• Decision Tree", "• Random Forest", "• Extra Trees", "• AdaBoost", "• Gradient Boosting", "• Bagging Classifier", "• Gaussian Naive Bayes", "• MLP Classifier (Neural Network)", "• XGBoost Classifier", "• CatBoost Classifier", "• LightGBM Classifier")
    with col2:
        st.subheader("Regression")
        st.write("• Linear Regression", "• Ridge Regression", "• Support Vector Regressor (SVR)", "• Decision Tree Regressor", "• Random Forest Regressor", "• Extra Trees Regressor", "• AdaBoost Regressor", "• Gradient Boosting Regressor", "• Bagging Regressor", "• MLP Regressor (Neural Network)", "• XGBoost Regressor", "• CatBoost Regressor", "• LightGBM Regressor")

# Defines the dialog content for the "Library Information" section.
@st.dialog("Library Information")
def Library_used():
    st.header("📚 Core Libraries Used")
    st.write("AnalytiBot is built on the shoulders of giants. We leverage the best of the open-source Python ecosystem:")
    st.write("• **Streamlit:** For creating the interactive web application.", "• **Pandas:** For data manipulation, cleaning, and analysis.", "• **NumPy:** For numerical operations and array processing.", "• **Scikit-learn:** For a wide range of machine learning models, metrics, and utilities.", "• **Matplotlib & Seaborn:** For powerful and beautiful data visualizations.", "• **XGBoost, CatBoost, LightGBM:** For high-performance gradient boosting models.", "• **SciPy:** For scientific and technical computing.", "• **Vecstack:** For implementing stacking ensembles.")

# Defines the dialog for the feedback form, which calls an external function.
@st.dialog("Feedback Form")
def feedback_took():
    feedback_function()

# ----------------- MAIN PAGE CONTENT -----------------

# Header section with the main title and subtitle.
with st.container():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 4rem; font-weight: bold;'>AnalytiBot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.5rem; max-width: 800px; margin: auto;'>An interactive no-code platform for building complete machine learning pipelines, from data to deployment.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Centered "Get Started" button that triggers the overview dialog.
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Get Started & Explore Features"):
            show_contact_form()

# Main content container with a styled, blurred background.
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # --- About AnalytiBot Section ---
    st.markdown("<h2>🔎 About AnalytiBot</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.1rem; max-width: 900px; margin: auto;'>
    Welcome to the engine room of our application! AnalytiBot leverages the power of <b>machine learning</b> to solve challenging problems. 
    It's more than just a tool—it's your personal data scientist.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Features Section with dynamically generated cards ---
    st.subheader("What Can You Do?")
    feature_data = {
        "Data Manipulation": "assets/icon_data.png", "Visualization": "assets/icon_viz.png",
        "Feature Making": "assets/icon_feature.png", "Model Selection": "assets/icon_model.png",
        "Model Tuning": "assets/icon_tune.png", "Model Evaluation": "assets/icon_eval.png",
        "Stacking": "assets/icon_stack.png", "Advanced Modeling": "assets/icon_adv.png",
    }
    # Loop to create a 4-column layout for feature cards.
    cols = st.columns(4)
    i = 0
    for feature, icon_path in feature_data.items():
        with cols[i % 4]:
            st.markdown(f"""
            <div class="feature-card">
                <img src="data:image/png;base64,{get_base64_of_bin_file(icon_path)}">
                <p>{feature}</p>
            </div>
            """, unsafe_allow_html=True)
        i += 1
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Technology & Methods Section ---
    st.markdown("<h2>🔬 Technology & Methods</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large", vertical_alignment="center")
    with col1:
        st.markdown("""
        <p style='font-size: 1.1rem;'>
        Our engine is built in Python, leveraging a suite of powerful libraries like <b>Pandas, NumPy,</b> and <b>Scikit-learn</b> for robust analysis and modeling. For cutting-edge performance, we integrate <b>XGBoost, CatBoost,</b> and <b>LightGBM</b>. All visualizations are powered by <b>Matplotlib</b> and <b>Seaborn</b>.
        </p>
        <p style='font-size: 1.1rem;'>
        We employ advanced techniques like multi-layer <b>Stacking</b> and <b>Voting</b> ensembles to achieve optimal, diverse, and highly accurate results.
        </p>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Explore Models"): models_used()
        if c2.button("Explore Libraries"): Library_used()
    with col2:
        st.image("assets/library_logos.png")

    # --- Future Outlook Section ---
    st.markdown("<h2>📈 The Future is AI-Driven</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.1rem; max-width: 900px; margin: auto;'>
    The machine learning industry is expanding at an unprecedented rate. Staying ahead requires powerful, accessible tools. AnalytiBot is designed to put state-of-the-art ML capabilities directly into your hands. The world of machine learning (ML) is in a state of perpetual, rapid evolution. As we move through 2025, the trends shaping the industry are less about foundational discoveries and more about the sophisticated application, integration, and industrialization of AI. The dominant force continues to be Generative AI, which has matured beyond simple text and image generation. This trend is pushing the boundaries of creativity and engineering, making AI a collaborative partner rather than just a tool.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Limitations Section ---
    st.markdown("<h2>⚠️ Limitations & Considerations</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large", vertical_alignment="center")
    with col1:
        st.info("🔹 Due to multiple instruction system may crash.", icon="⚠️")
        st.info("🔹 You must do everything precisely to get best results.", icon="👾")
        st.info("🔹 Multiple output is Not yet available.", icon="❄️")
    with col2:
        st.info("🔹 Training multiple advanced models can be time-consuming.", icon="⏳")
        st.info("🔹 The system requires flat, tabular data (no nested lists/JSON).", icon="📋")
        st.info("🔹 Image data must be pre-converted into pixel features before upload.", icon="🖼️")

    # --- About The Creator Section ---
    st.markdown("<h2>👨‍💻 About The Creator</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2], gap="large", vertical_alignment="center")
    with col1:
        st.image("assets/profile_pic.png")
    with col2:
        st.markdown("""
        <p style='font-size: 1.1rem;'>
        I'm <b>Sumit Bhardwaj</b>, a Computer Engineering student at <b>J.C. Bose University of Science and Technology</b>. This project was a challenging and rewarding journey into the world of applied ML.
        </p>
        <p style='font-size: 1.1rem;'>
        Your feedback is invaluable for motivating new ideas and future expansions. Please don't hesitate to share your thoughts!
        </p>
        """, unsafe_allow_html=True)
        # Button to trigger the feedback form dialog.
        if st.button("💌 Provide Feedback"):
            feedback_took()

    st.markdown('</div>', unsafe_allow_html=True) # Closes the main-container div