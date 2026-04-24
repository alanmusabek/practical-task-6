import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(page_title="🌸 Iris Flower Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Classification")
st.markdown("Enter flower measurements to predict the species using our ML model.")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a **Random Forest** model trained on the Iris dataset.")
    st.write("**Features:**")
    st.write("- 📏 Sepal Length (cm)")
    st.write("- 📏 Sepal Width (cm)")
    st.write("- 📏 Petal Length (cm)")
    st.write("- 📏 Petal Width (cm)")
    
    st.markdown("---")
    st.write("🔗 [API Docs](http://localhost:8000/docs)")

# Input form
st.subheader("📝 Enter Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)

with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

# Prediction button
if st.button("🔮 Predict Species", type="primary"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    
    try:
        # Call FastAPI backend
        response = requests.post(
            "http://localhost:8000/predict",
            json={"features": features},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            pred_class = result["prediction"]
            proba = result["class_probabilities"]
            
            # Map prediction to class name
            class_names = {0: "🌸 Setosa", 1: "🌺 Versicolor", 2: "🌻 Virginica"}
            
            # Display result
            st.success(f"**Predicted Species:** {class_names.get(pred_class, 'Unknown')}")
            
            # Show probabilities
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": ["Setosa", "Versicolor", "Virginica"],
                "Probability": [proba["setosa"], proba["versicolor"], proba["virginica"]]
            })
            st.bar_chart(prob_df.set_index("Class"))
            
            # Show input summary
            with st.expander("Input Summary"):
                st.write(f"- Sepal: {sepal_length} × {sepal_width} cm")
                st.write(f"- Petal: {petal_length} × {petal_width} cm")
                
        else:
            st.error(f"API Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the backend is running on `http://localhost:8000`")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Example data section
with st.expander("Example Inputs"):
    st.write("**Setosa example:** [5.1, 3.5, 1.4, 0.2]")
    st.write("**Versicolor example:** [6.2, 2.9, 4.3, 1.3]")
    st.write("**Virginica example:** [7.3, 3.0, 6.3, 1.8]")