import streamlit as st
from joblib import load

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
model = load("model.pkl")
vectorizer = load("vectorizer.pkl")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📌 About")

st.sidebar.info(
    """
    **Spam Email Classifier**

    This application uses a Machine Learning model
    to classify an email as **Spam** or **Not Spam**.

    **Technologies Used**
    - Python
    - Streamlit
    - Scikit-learn
    - Joblib
    """
)

# -------------------------------
# Main Heading
# -------------------------------
st.title("📧 Spam Email Classifier")

st.write(
    "Enter the email message below and click **Predict** "
    "to check whether it is **Spam** or **Not Spam**."
)

st.markdown("---")

# -------------------------------
# Email Input
# -------------------------------
email = st.text_area(
    "📩 Email Message",
    height=220,
    placeholder="""Example:

Congratulations!

You have won a free iPhone.
Click the link below to claim your prize.
"""
)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔍 Predict", use_container_width=True):

    if email.strip() == "":
        st.warning("⚠ Please enter an email message before predicting.")

    else:

        email_vector = vectorizer.transform([email])

        prediction = model.predict(email_vector)

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("🚨 This Email is **SPAM**")
            st.write(
                "This message appears to contain characteristics commonly found in spam emails."
            )

        else:
            st.success("✅ This Email is **NOT SPAM**")
            st.write(
                "This message appears to be a legitimate email."
            )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")

st.caption(
    "Developed as a Mini Project using Machine Learning and Streamlit."
)