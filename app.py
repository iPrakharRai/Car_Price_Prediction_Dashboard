import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Load model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load data
car = pd.read_csv('Cleaned_Car_data.csv')

st.title("🚗 Car Price Prediction Dashboard")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", sorted(car['company'].unique()))
    year = st.selectbox("Year", sorted(car['year'].unique(), reverse=True))
    kms = st.number_input("Kilometers Driven", min_value=0)

with col2:
    name = st.selectbox("Model", car['name'].unique())
    fuel = st.selectbox("Fuel Type", car['fuel_type'].unique())

if st.button("Predict Price 💰"):
    input_df = pd.DataFrame(
        [[name, company, year, kms, fuel]],
        columns=['name','company','year','kms_driven','fuel_type']
    )

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")

# ---- Stylish Sidebar & Data Disclaimer ----
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="color: #007BFF; margin-bottom: 0;">🚙 Car Analysis</h2>
    <p style="color: #999999; font-size: 14px; margin-top: 5px;">Advanced ML Predictor</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📊 About the App")
st.sidebar.info(
    "This application uses a Machine Learning model to predict the estimated price of a car "
    "based on its specifications."
)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚠️ Data Disclaimer")
st.sidebar.warning(
    "The predictions provided by this dashboard are based on historical data. "
    "Actual market prices may vary due to external factors. "
    "Please use this tool as a reference rather than a definitive guide."
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #555;'><small>Built for premium insights</small></div>", unsafe_allow_html=True)

# ---- Custom CSS for Background and Footer ----
st.markdown("""
    <style>
    /* Gradient Background Effect */
    .stApp {
        background-color: #000000;
        background-image: 
            radial-gradient(circle at 20% 40%, rgba(0, 123, 255, 0.15) 0%, transparent 40%), 
            radial-gradient(circle at 80% 60%, rgba(0, 123, 255, 0.1) 0%, transparent 40%);
        background-attachment: fixed;
    }

    /* Custom Footer CSS */
    .custom-footer {
        background-color: transparent;
        color: #cccccc;
        padding: 40px 20px 20px 20px;
        margin-top: 50px;
        border-top: 1px solid #222222;
        font-family: sans-serif;
    }

    .footer-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        max-width: 1000px;
        margin: 0 auto;
    }

    .footer-col {
        flex: 1;
        min-width: 200px;
        margin-bottom: 20px;
    }

    .footer-col h4 {
        color: #ffffff;
        font-size: 16px;
        margin-bottom: 20px;
        font-weight: 600;
    }

    .footer-col ul {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }

    .footer-col ul li {
        margin-bottom: 12px;
    }

    .footer-col ul li a {
        color: #999999;
        text-decoration: none;
        font-size: 14px;
        transition: 0.3s;
    }

    .footer-col ul li a:hover {
        color: #007BFF;
    }

    .footer-bottom {
        text-align: center;
        padding-top: 20px;
        border-top: 1px solid #333333;
        color: #777777;
        font-size: 13px;
        margin-top: 20px;
    }
    </style>

    <div class="custom-footer">
        <div class="footer-container">
            <div class="footer-col">
                <h4>Get to Know Us</h4>
                <ul>
                    <li><a href="https://iprakharrai.github.io/prakhar_portfolio/#" target="_blank">About Prakhar Rai</a></li>
                    <li><a href="https://iprakharrai.github.io/prakhar_portfolio/#" target="_blank">Portfolio Overview</a></li>
                    <li><a href="https://colab.research.google.com/drive/1gzR_WMjSjkE0v-s69mmVUgO_H0UUStwo#scrollTo=TNc1t9QUCtGW" target="_blank">Data Insights</a></li>
                </ul>
            </div>
            <div class="footer-col">
                <h4>Connect with Me</h4>
                <ul>
                    <li><a href="https://www.linkedin.com/in/iprakharrai/" target="_blank">LinkedIn</a></li>
                    <li><a href="https://github.com/iPrakharRai" target="_blank">GitHub</a></li>
                    <li><a href="https://x.com/iPrakharRai" target="_blank">Twitter (X)</a></li>
                </ul>
            </div>
            <div class="footer-col">
                <h4>Let Us Help You</h4>
                <ul>
                    <li><a href="mailto:raiprakhar0123@gmail.com">Contact Support</a></li>
                    <li><span style="color: #999999; font-size: 14px; cursor: default;">Data Disclaimer (See Sidebar)</span></li>
                </ul>
            </div>
        </div>
        <div class="footer-bottom">
            © 2026 Prakhar Rai. All Rights Reserved. Data dashboard designed for premium insights.
        </div>
    </div>
""", unsafe_allow_html=True)