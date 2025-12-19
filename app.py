import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
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
# LOAD IRIS DATASET
# ---------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
df["Species"] = df["Species"].map({
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
})

# ---------------------------------
# SHOW DATASET
# ---------------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# ---------------------------------
# FEATURES & TARGET
# ---------------------------------
X = df.iloc[:, :-1]
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
