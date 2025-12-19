import streamlit as st
import joblib

# Page Configuration
st.set_page_config(page_title="Fake Job Detector", page_icon="🛡", layout="centered")

# Sidebar - Credits section
with st.sidebar:
    st.title("📌 Project Info")
    st.info("This AI model analyzes job postings to help users avoid potential employment fraud.")
    
    st.markdown("---")
    st.warning("⚠ Disclaimer: This tool is for educational purposes only.")

# Main Header
st.title("🛡 Fake Job Posting Detector")
st.markdown("Paste the job description below to check its safety and fraud risk.")

# Input Box
job_text = st.text_area("Paste Full Job Description Here:", height=200, placeholder="Example: Urgent hiring for Data Entry. High salary. No experience needed...")

if st.button("Analyze Job Post 🚀"):
    if job_text.strip() == "":
        st.error("Please enter some text first! ✍")
    else:
        try:
            # Loading the model files with your specific names
            model = joblib.load('logistic_regression_model_final.joblib')
            vectorizer = joblib.load('tfidf_vectorizer_final.joblib')
            
            # Prediction logic
            data = vectorizer.transform([job_text])
            prediction = model.predict(data)
            probability = model.predict_proba(data)
            
            st.markdown("---")
            st.subheader("📊 Analysis Result")
            
            # Result Section
            if prediction[0] == 1:
                st.error(f"### 🚩 Prediction: FRAUD DETECTED!")
                st.warning(f"Probability of being fake: {probability[0][1]*100:.2f}%")
            else:
                st.success(f"### ✅ Prediction: APPEARS TO BE REAL")
                st.info(f"Confidence Level: {probability[0][0]*100:.2f}%")
            
            # Safety Tips Section
            st.markdown("---")
            st.subheader("💡 Safety Checklist for You:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("✅ Verify the company on their official website.")
                st.write("✅ Check if the interview is at an official office.")
            with col2:
                st.write("❌ Never pay any 'Registration' or 'Security' fees.")
                st.write("❌ Do not share personal bank details over chat.")

        except Exception as e:
            st.error(f"System Error: {e}")
            st.info("Make sure 'logistic_regression_model_final.joblib' and 'tfidf_vectorizer_final.joblib' are in the same folder.")