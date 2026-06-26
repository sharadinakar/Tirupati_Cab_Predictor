import streamlit as st
import pickle
import numpy as np
from model_blueprint import MyLinearRegression

# 1. UI heading very minimal
st.title("🚖 Tirupati Cab Price Predictor")
st.markdown("Enter the details below to predict the cab fare.")

# 2. సేవ్ అయిన మోడల్ ని లోడ్ చేయడం
# (Streamlit లో ప్రతిసారీ మోడల్ లోడ్ అవ్వకుండా cache వాడటం బెస్ట్ ప్రాక్టీస్)
@st.cache_resource
def load_model():
    with open('cab_price_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 3. ఒకదాని కింద ఒకటి వచ్చే ఇన్పుట్స్ (Vertical Layout)
# (నీ డేటాలో 3 ఫీచర్స్ ఉన్నాయని అనుకుని రాస్తున్నాను)
distance = st.number_input("Distance (in km)", min_value=0.0, value=5.0, step=0.5)
time = st.number_input("Time (in minutes)", min_value=0.0, value=15.0, step=1.0)
traffic = st.number_input("Traffic Level (1 = Low, 5 = High)", min_value=1, max_value=5, value=2, step=1)

# 4. మినిమల్ బటన్ (use_container_width=False వాడితే బటన్ చిన్నగా వస్తుంది)
if st.button("Predict Price", use_container_width=False):
    # మనం ఇచ్చిన ఇన్పుట్స్ ని మ్యాట్రిక్స్ లాగా మార్చడం
    features = np.array([[distance, time, traffic]])
    
    # ప్రిడిక్ట్ చేయడం
    price = model.predict(features)
    
    # రిజల్ట్ ని చూపించడం
    st.success(f"Estimated Fare: ₹{price[0]:.2f}")