# AnalytiML: An Interactive & Automated Machine Learning Platform

An intuitive web application built with Streamlit that empowers users to perform end-to-end machine learning tasks, from data exploration to model deployment, with both interactive and fully automated workflows.

### ➡️ **[Live Demo: analytibot.streamlit.app](https://analytibot.streamlit.app)**

---

## ✨ Features

This project is divided into two powerful modes to cater to both hands-on data scientists and users looking for rapid results.

### 🛠️ Interactive ML Workbench

For users who want full control over the machine learning pipeline.

* 📤 **Upload & Explore:** Upload your dataset (e.g., CSV) and instantly view data profiles, statistics, and visualizations.
* 📊 **Data Visualization:** Generate various plots like histograms, box plots, and correlation heatmaps to understand data distributions and relationships.
* 🔧 **Data Manipulation & Feature Engineering:** Clean your data and engineer new features to improve model performance.
* 🧠 **Train a Variety of Models:** Train classic and advanced ML models, including:
    * XGBoost
    * LightGBM
    * Deep Neural Networks (DNN)
    * Random Forest
    * ...and many more from the Scikit-learn library.
* 📈 **Performance Analysis & Comparison:** Evaluate models using key metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) and visually compare their performance side-by-side.
* ⚙️ **Advanced Customization:**
    * **Hyperparameter Tuning:** Fine-tune your models to find the optimal set of parameters.
    * **Ensemble Methods:** Combine model predictions using **Bagging**, **Voting**, and **Stacking** to build more robust and accurate meta-models.

### 🤖 Automated ML Pipeline

For users who need quick, high-quality results without manual intervention.

* 🚀 **One-Click AutoML:** Simply upload your dataset and let the application handle the entire workflow automatically.
* 🔍 **Automated Analysis:** The app automatically performs data preprocessing, feature engineering, model training, and hyperparameter tuning across a suite of algorithms.
* 🏆 **Smart Ensembling:** It identifies the top-performing base models and automatically creates powerful **Voting** and **Stacking** ensembles.
* 🥇 **Top Model Recommendations:** At the end of the process, the app presents the **top 3 best-performing models** for your specific dataset, complete with their performance metrics.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-learn, XGBoost, LightGBM, TensorFlow/Keras
* **Data Visualization:** Matplotlib, Seaborn, Plotly

---

## 🚀 Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sumitbhardwaj10/AnalytiBot.git
    cd AnalytiBot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

Once the dependencies are installed, you can run the Streamlit app with the following command:

```bash
streamlit run app.py
