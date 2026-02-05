# -----------------------------------------
# House Rent Prediction in India
# Using Scikit-Learn + Streamlit
# -----------------------------------------

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(page_title="House Rent Prediction", layout="wide")
st.title("🏠 House Rent Prediction in India")

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("House_Rent_Dataset.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------------------
# DATA PREPROCESSING
# -----------------------------------------

# Drop non-numeric / unnecessary columns
df = df.drop(columns=["Floor", "Posted On", "Point of Contact"], errors="ignore")

# Label Encoding categorical columns
label_encoders = {}
categorical_cols = [
    "Area Type",
    "Area Locality",
    "City",
    "Furnishing Status",
    "Tenant Preferred"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop("Rent", axis=1)
y = df["Rent"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

st.success(f"✅ Model trained successfully (R² Score: {accuracy:.2f})")

# -----------------------------------------
# USER INPUT
# -----------------------------------------
st.sidebar.header("Enter House Details")

bhk = st.sidebar.slider("BHK", 1, 6, 2)
size = st.sidebar.slider("Size (sq ft)", 300, 5000, 1000)
bathroom = st.sidebar.slider("Bathrooms", 1, 6, 2)

area_type = st.sidebar.selectbox(
    "Area Type", label_encoders["Area Type"].classes_
)

area_locality = st.sidebar.selectbox(
    "Area Locality", label_encoders["Area Locality"].classes_
)

city = st.sidebar.selectbox(
    "City", label_encoders["City"].classes_
)

furnishing = st.sidebar.selectbox(
    "Furnishing Status", label_encoders["Furnishing Status"].classes_
)

tenant = st.sidebar.selectbox(
    "Tenant Preferred", label_encoders["Tenant Preferred"].classes_
)

# Create input dataframe
input_data = pd.DataFrame([[
    bhk,
    size,
    label_encoders["Area Type"].transform([area_type])[0],
    label_encoders["Area Locality"].transform([area_locality])[0],
    label_encoders["City"].transform([city])[0],
    label_encoders["Furnishing Status"].transform([furnishing])[0],
    label_encoders["Tenant Preferred"].transform([tenant])[0],
    bathroom
]], columns=X.columns)

# -----------------------------------------
# PREDICTION
# -----------------------------------------
if st.sidebar.button("Predict Rent"):
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Estimated Monthly Rent")
    st.success(f"₹ {int(prediction):,}")

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("---")
st.caption("📊 House Rent Prediction in India using Scikit-Learn | Streamlit App")