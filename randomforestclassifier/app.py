import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Insurance Cost Prediction")

st.title("Insurance Cost Prediction using Random Forest Regressor")

uploaded_file = st.file_uploader(
    "Upload insurance.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Encoding categorical columns
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df["region"] = df["region"].map({
        "northeast": 0,
        "northwest": 1,
        "southeast": 2,
        "southwest": 3
    })

    X = df.drop("charges", axis=1)
    y = df["charges"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    st.subheader("Enter Customer Details")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
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

    if st.button("Predict Insurance Cost"):

        input_data = pd.DataFrame(
            [[
                age,
                sex,
                bmi,
                children,
                smoker,
                region
            ]],
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

        st.success(
            f"Predicted Insurance Cost: ₹ {prediction:,.2f}"
        )

else:
    st.info("Please upload insurance.csv file.")
