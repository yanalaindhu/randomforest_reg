import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("insurance.csv")

# Encoding
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["region"] = df["region"].map({
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
})

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Streamlit UI
st.title("Insurance Cost Prediction")

age = st.number_input("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_map = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

region = region_map[region]

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, sex, bmi, children, smoker, region]],
        columns=[
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Insurance Cost: ₹ {prediction:.2f}")