import re
import streamlit as st
import requests
# --- PUT THIS AT THE BOTTOM OF YOUR SCRIPT ---
# You can remove the other imports for re, requests, etc., at the bottom
# since they are already imported at the top.
WEBHOOK_URL="https://eotsu39ewtmgztt.m.pipedream.net"
def is_valid_mail(email):
    # Corrected regex pattern (no extra ']')
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(email_pattern, email) is not None


def feedback_function():
    st.write("Please provide your feedback below:")
    with st.form("FeedbackForm", clear_on_submit=True):
        name = st.text_input("Enter your name")
        age = st.text_input("Enter your age")
        email = st.text_input("Enter your email")
        message = st.text_area("Enter your message")  # Using text_area is better for messages
        submit = st.form_submit_button("Submit")

        if submit:
            # Combined check for all empty fields
            if not name or not age or not email or not message:
                st.error("Please fill out all fields.", icon="❗")
                st.stop()  # This stops the execution if any field is empty

            # Check for valid email format
            if not is_valid_mail(email):
                st.error("Please enter a valid email.", icon="📩")
                st.stop()  # This stops the execution if the email is invalid

            # This code is ONLY reached if all checks above pass
            data = {"name": name, "age": age, "email": email, "message": message}

            try:
                response = requests.post(WEBHOOK_URL, json=data)
                response.raise_for_status()  # This will raise an error for bad responses
                st.success("Thank you for your feedback!")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to submit: {e}", icon="⚠️")