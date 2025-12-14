import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Iris Prediction App")
st.title("🌸 Iris Flower Prediction App")
st.write("Decision Tree Classifier")

# ---------------------------------
# DATASET (EMBEDDED IRIS DATA)
# ---------------------------------
data = {
    "SepalLengthCm": [
        5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
        5.4, 4.8, 4.8, 4.3, 5.8
    ],
    "SepalWidthCm": [
        3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
        3.7, 3.4, 3.0, 3.0, 4.0
    ],
    "PetalLengthCm": [
        1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
        1.5, 1.6, 1.4, 1.1, 1.2
    ],
    "PetalWidthCm": [
        0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
        0.2, 0.2, 0.1, 0.1, 0.2
    ],
    "Species": [
        "Iris-setosa", "Iris-setosa", "Iris-setosa", "Iris-setosa",
        "Iris-setosa", "Iris-setosa", "Iris-setosa", "Iris-setosa",
        "Iris-setosa", "Iris-setosa", "Iris-setosa", "Iris-setosa",
        "Iris-setosa", "Iris-setosa", "Iris-setosa"
    ]
}

df = pd.DataFrame(data)

# ---------------------------------
# SHOW DATASET
# ---------------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# ---------------------------------
# FEATURES & TARGET
# ---------------------------------
X = df.drop("Species", axis=1)
y = df["Species"]

# ---------------------------------
# TRAIN TEST SPLIT
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------
# MODEL TRAINING
# ---------------------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ---------------------------------
# MODEL EVALUATION
# ---------------------------------
predictions = model.predict(X_test)

st.subheader("📈 Model Performance")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, predictions))

st.write("Classification Report:")
st.text(classification_report(y_test, predictions))

# ---------------------------------
# USER INPUT
# ---------------------------------
st.subheader("🌼 Enter Flower Measurements")

sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# ---------------------------------
# PREDICTION
# ---------------------------------
if st.button("Predict Species"):
    input_df = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
    result = model.predict(input_df)[0]
    st.success(f"🌸 Predicted Species: **{result}**")
