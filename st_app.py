
import streamlit as st
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('titanic_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# App interface
st.title('Titanic Survival Predictor')

# Input widgets
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    
with col2:
    fare = st.number_input('Fare', min_value=0, value=50)
    family_size = st.number_input('Family Size', min_value=0, max_value=10, value=0)

# Prediction logic
if st.button('Predict Survival'):
    input_data = pd.DataFrame([[pclass, sex, age, fare, family_size]],
                            columns=['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize'])
    
    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]
    
    st.subheader('Result')
    st.metric("Survival Probability", f"{probability:.1%}")
    st.write(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")


# print("\n=== Streamlit App Code ===")
# print("Save this as 'app.py' and run with: streamlit run app.py")
# print(streamlit_code)
