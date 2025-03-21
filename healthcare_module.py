import streamlit as st
import pandas as pd
import numpy as np
import pyttsx3
import speech_recognition as sr
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
from reportlab.pdfgen import canvas
import random
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

def health_app():
    st.markdown("<h1 style='text-align: center; color: #e7feff;'>💊 Health Care Recommendation System 🩺</h1>", unsafe_allow_html=True)
    st.write("Select the symptoms and get your predicted disease with relevant precautions and recommendations.")

# Load datasets
description = pd.read_csv("datasets/description.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")
workout = pd.read_csv("datasets/workout_df.csv")
doctors = pd.read_csv("datasets/doctors.csv")
symptoms_df = pd.read_csv("datasets/symtoms_df.csv")
tips = pd.read_csv("datasets/health_tips.csv")

# Load ML Model
svc_model = pickle.load(open("models/svc.pkl", "rb"))
dataset = pd.read_csv("datasets/Training.csv")
all_symptoms = dataset.columns[:-1].tolist()

# Label datasets
le = LabelEncoder()
le.fit(dataset['prognosis'])

st.title("💊 Health Care Recommendation System")

# ---------------- BMI / Health Score Checker ------------------
st.subheader("📏 BMI & Health Score Checker")
height = st.number_input("Enter your height (in cm):", min_value=50.0, max_value=250.0, step=0.5)
weight = st.number_input("Enter your weight (in kg):", min_value=10.0, max_value=200.0, step=0.5)

if height > 0 and weight > 0:
    bmi = weight / ((height / 100) ** 2)
    st.markdown(f"**BMI: {bmi:.2f}**")
    if bmi < 18.5:
        st.warning("Underweight")
    elif 18.5 <= bmi < 25:
        st.success("Normal weight")
    elif 25 <= bmi < 30:
        st.info("Overweight")
    else:
        st.error("Obese")

# ---------------- Daily Tip ------------------
st.subheader("💡 Daily Health Tip")
tip = random.choice(tips['Tip'].tolist())
st.info(tip)

# ---------------- Voice Input ------------------
st.subheader("🎙 Voice Input (Optional)")
selected_symptoms = []

if st.button("🎤 Speak your symptoms"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak your symptoms (e.g., headache, cough)")
        try:
            audio = r.listen(source, timeout=5)
            voice_text = r.recognize_google(audio)
            voice_text_cleaned = voice_text.lower().replace(" ", "_")
            detected_symptoms = [symptom for symptom in all_symptoms if symptom.replace("_", "") in voice_text_cleaned.replace("_", "")]
            if detected_symptoms:
                st.success(f"Detected symptoms: {', '.join(sym.replace('_', ' ') for sym in detected_symptoms)}")
                selected_symptoms = detected_symptoms

                # Auto Predict after voice input
                st.subheader("🔍 Auto-Prediction Result")
                input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
                input_df = pd.DataFrame([input_vector], columns=all_symptoms)
                prediction_encoded = svc_model.predict(input_df)[0]
                predicted_disease = le.inverse_transform([prediction_encoded])[0]
                st.success(f"🩺 Predicted Disease: **{predicted_disease.replace('_', ' ')}**")

                # Description
                desc = description[description['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]['Description']
                if not desc.empty:
                    st.subheader("📃 Disease Description")
                    st.write(desc.values[0])

                # Precautions
                pre = precautions[precautions['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
                if not pre.empty:
                    st.subheader("🛡 Precautions")
                    for val in pre.iloc[0][1:]:
                        st.write(f"- {val}")

                # Medications
                med = medications[medications['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
                if not med.empty:
                    st.subheader("💊 Medications")
                    for val in med['Medication']:
                        st.write(f"- {val}")

                # Workout
                wo = workout[workout['disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
                if not wo.empty:
                    st.subheader("🏃 Workouts")
                    for val in wo['workout']:
                        st.write(f"- {val}")

                # Diet
                diet = diets[diets['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
                if not diet.empty:
                    st.subheader("🥗 Diet Suggestions")
                    for val in diet['Diet']:
                        st.write(f"- {val}")

                # Doctors
                doc = doctors[doctors['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
                if not doc.empty:
                    st.subheader("👨‍⚕️ Doctor Suggestions")
                    for i in range(len(doc)):
                        st.markdown(f"- **{doc.iloc[i]['Doctor_Name']}**, *{doc.iloc[i]['Specialization']}* ({doc.iloc[i]['Location']})")

                # Risk Meter
                st.subheader("📊 Disease Risk Level Meter")
                risk_level = random.choice(['Low', 'Moderate', 'High'])
                color_map = {"Low": "green", "Moderate": "orange", "High": "red"}
                fig, ax = plt.subplots(figsize=(5, 0.5))
                ax.barh(["Risk Level"], [1], color=color_map[risk_level])
                ax.set_xlim(0, 1)
                ax.set_title(f"Risk: {risk_level}", fontsize=14)
                ax.axis("off")
                st.pyplot(fig)

                # PDF Download using BytesIO
                buffer = BytesIO()
                c = canvas.Canvas(buffer)
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, 800, "Health Recommendation Report")
                c.setFont("Helvetica", 12)
                c.drawString(50, 770, f"Disease: {predicted_disease.replace('_', ' ')}")
                c.drawString(50, 750, f"Precautions: {', '.join(pre.iloc[0][1:].values)}")
                c.drawString(50, 730, f"Medications: {', '.join(med['Medication'].values)}")
                c.drawString(50, 710, f"Workout: {', '.join(wo['workout'].values)}")
                c.drawString(50, 690, f"Diet: {', '.join(diet['Diet'].values)}")
                c.save()
                buffer.seek(0)
                st.download_button(label="📥 Download Report as PDF", data=buffer, file_name="Health_Report.pdf", mime="application/pdf")
            else:
                st.warning("No symptoms detected.")
        except Exception as e:
            st.error(f"Could not recognize. Error: {e}")

# ---------------- Manual Selection ------------------
if not selected_symptoms:
    selected_symptoms = st.multiselect("Select Symptoms", [s.replace("_", " ") for s in all_symptoms])

# ---------------- Manual Prediction ------------------
# ---------------- Manual Prediction ------------------
if st.button("🔍 Predict Disease (Manual)"):
    selected_symptoms = [s.replace(" ", "_") for s in selected_symptoms]
    if not selected_symptoms:
        st.warning("Please select symptoms")
    else:
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
        input_df = pd.DataFrame([input_vector], columns=all_symptoms)
        prediction_encoded = svc_model.predict(input_df)[0]
        predicted_disease = le.inverse_transform([prediction_encoded])[0]
        st.success(f"🩺 Predicted Disease: **{predicted_disease.replace('_', ' ')}**")

        # Description
        desc = description[description['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]['Description']
        if not desc.empty:
            st.subheader("📃 Disease Description")
            st.write(desc.values[0])

        # Precautions
        pre = precautions[precautions['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
        if not pre.empty:
            st.subheader("🛡 Precautions")
            for val in pre.iloc[0][1:]:
                st.write(f"- {val}")

        # Medications
        med = medications[medications['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
        if not med.empty:
            st.subheader("💊 Medications")
            for val in med['Medication']:
                st.write(f"- {val}")

        # Workout
        wo = workout[workout['disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
        if not wo.empty:
            st.subheader("🏃 Workouts")
            for val in wo['workout']:
                st.write(f"- {val}")

        # Diet
        diet = diets[diets['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
        if not diet.empty:
            st.subheader("🥗 Diet Suggestions")
            for val in diet['Diet']:
                st.write(f"- {val}")

        # Doctors
        doc = doctors[doctors['Disease'].str.lower().str.replace("_", "") == predicted_disease.lower().replace("_", "")]
        if not doc.empty:
            st.subheader("👨‍⚕️ Doctor Suggestions")
            for i in range(len(doc)):
                st.markdown(f"- **{doc.iloc[i]['Doctor_Name']}**, *{doc.iloc[i]['Specialization']}* ({doc.iloc[i]['Location']})")

        # Risk Meter
        st.subheader("📊 Disease Risk Level Meter")
        risk_level = random.choice(['Low', 'Moderate', 'High'])
        color_map = {"Low": "green", "Moderate": "orange", "High": "red"}
        fig, ax = plt.subplots(figsize=(5, 0.5))
        ax.barh(["Risk Level"], [1], color=color_map[risk_level])
        ax.set_xlim(0, 1)
        ax.set_title(f"Risk: {risk_level}", fontsize=14)
        ax.axis("off")
        st.pyplot(fig)

        # PDF Download using BytesIO
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 800, "Health Recommendation Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 770, f"Disease: {predicted_disease.replace('_', ' ')}")
        c.drawString(50, 750, f"Precautions: {', '.join(pre.iloc[0][1:].values)}")
        c.drawString(50, 730, f"Medications: {', '.join(med['Medication'].values)}")
        c.drawString(50, 710, f"Workout: {', '.join(wo['workout'].values)}")
        c.drawString(50, 690, f"Diet: {', '.join(diet['Diet'].values)}")
        c.save()
        buffer.seek(0)
        st.download_button(label="📥 Download Report as PDF", data=buffer, file_name="Health_Report.pdf", mime="application/pdf")


        # [Add same blocks again as above if needed]

# Footer
st.markdown("---")
st.markdown("💙 Developed with care - Stay Healthy!")
