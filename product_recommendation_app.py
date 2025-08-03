#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#.\env\Scripts\Activate.ps1



import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

#setting page title
st.set_page_config(page_title="🛒 Shopper Spectrum", layout="wide")




# Load models and data
product_similarity_df = joblib.load(r"C:\Users\user\Music\PROJECT4\product_similarity.pkl")
product_names = joblib.load(r"C:\Users\user\Music\PROJECT4\product_names.pkl")
kmeans = joblib.load(r"C:\Users\user\Music\PROJECT4\kmeans_model.pkl")
scaler = joblib.load(r"C:\Users\user\Music\PROJECT4\scaler.pkl")


#product recommendations
def recommend_by_product_name(product_name, top_n=5):
    def get_code_by_name(name):
        for code, desc in product_names.items():
            if name.lower() in desc.lower():
                return code
        return None

    def recommend_products(stock_code, top_n):
        if stock_code not in product_similarity_df:
            return [f"❌ StockCode {stock_code} not found."]
        sim_scores = product_similarity_df[stock_code].sort_values(ascending=False).drop(stock_code)
        top_similar = sim_scores.head(top_n).index
        return [f"{code} - {product_names.get(code, 'Unknown Product')}" for code in top_similar]

    code = get_code_by_name(product_name)
    if code:
        return recommend_products(code, top_n)
    else:
        return [f"❌ Product name '{product_name}' not found."]


#importing menu options
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(" Navigation",["💼 Start Here", "🎯 Product Recommendation", "🔍 Customer Segmentation"],default_index=0)

if selected == "💼 Start Here":    

    st.markdown("<h1 style='text-align: center; color:#FF69B4;'>🛍️SHOPPER SPECTRUM</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Customer Segmentation & Product Recommendations</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #fff0f5;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                margin-top: 1.5rem;'>
        <h2 style='color: #d63384;'>👋 Welcome Shoppie!</h2>
        <p style='font-size: 1.1rem; line-height: 1.6;'>
            Unlock insights into customer behavior and make smarter product decisions with the
            <strong>Shopper Spectrum App</strong>.
        </p>
        <hr style='border: 1px solid #f5c2e7;'>
        <ul style='font-size: 1.05rem; line-height: 1.8; padding-left: 1.2rem;'>        
        <li>🎯 <strong>Product Recommendation</strong> – Discover products similar to what users love</li>
        <li>🔍 <strong>Customer Segmentation</strong> – Identify key segments based on RFM behavior</li>
        </ul>
        <p style='font-size: 1rem; margin-top: 1rem; color: #6c757d;'>
            Use the navigation menu on the left to get started.
        </p>        
    </div>""", unsafe_allow_html=True)
    
elif selected == "🎯 Product Recommendation":    
    st.markdown("<h1 style='text-align: center; color:#FF69B4;'>🛍️SHOPPER SPECTRUM</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Product Recommendations</h2>", unsafe_allow_html=True)
    
    
    product_desc_list = sorted(list(set(product_names.values())))
    selected_product = st.selectbox("Select a product:", product_desc_list)

    if st.button("🔍 Get Recommendations"):
        if selected_product:
            recs = recommend_by_product_name(selected_product)
            st.subheader("Top 5 Similar Products")
            for i, r in enumerate(recs, 1):
                st.success(f"{i}. {r}")
        else:
            st.warning("Please select a product.")
            
elif selected == "🔍 Customer Segmentation":
    
    st.markdown("<h1 style='text-align: center; color:#FF69B4;'>🛍️SHOPPER SPECTRUM</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Customer Segmentation</h2>", unsafe_allow_html=True)
    
    recency = st.number_input("🗓️Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("💰Monetary (total spend)", min_value=0.0, step=10.0)

    if st.button("🧠 Predict Cluster"):
        
        scaled_input = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(scaled_input)[0]

        if cluster == 0:
            label = "💎 High-Value Customer"
        elif cluster == 1:
            label = "🟢 Regular Customer"
        elif cluster == 2:
            label = "🟡 Occasional Customer"
        else:
            label = "🔴 At-Risk Customer"

        st.success(f"✅ Predicted Segment: **{label}** (Cluster #{cluster})")
    
# Footer

st.markdown("---")
st.caption("🛍️ Shopper Spectrum • E-Commerce Analytics • Streamlit Cloud Ready 🚀")
    
