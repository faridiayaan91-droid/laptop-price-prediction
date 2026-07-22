import streamlit as st
import pickle
import numpy as np


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="LaptopAI - Price Predictor",
    page_icon="💻",
    layout="wide"
)


# ---------------- LOAD MODEL ----------------

try:
    pipe = pickle.load(open('pipe.pkl','rb'))
    df = pickle.load(open('df.pkl','rb'))

except Exception as e:
    st.error("❌ Model files not found. Please check pipe.pkl and df.pkl")
    st.stop()


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}


/* Main title */

.main-title {
    font-size: 55px;
    font-weight: 800;
    text-align:center;
    color:#111827;
}


.subtitle {
    text-align:center;
    font-size:22px;
    color:#6b7280;
}


/* Cards */

.card {

    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 10px 30px rgba(0,0,0,0.08);

}


/* Button */

.stButton>button {

    width:100%;
    height:55px;
    border-radius:15px;
    font-size:20px;
    font-weight:bold;
    background:#2563eb;
    color:white;

}


.stButton>button:hover {

    background:#1d4ed8;

}


/* Result */

.result-card {

    background:linear-gradient(135deg,#2563eb,#7c3aed);
    padding:35px;
    border-radius:25px;
    color:white;
    text-align:center;

}


.price {

    font-size:45px;
    font-weight:900;

}


</style>

""", unsafe_allow_html=True)



# ---------------- HEADER ----------------


st.markdown(
"""
<div class="main-title">
💻 LaptopAI
</div>

<div class="subtitle">
AI Powered Laptop Price Prediction System
</div>

<br>

""",
unsafe_allow_html=True
)



# ---------------- SIDEBAR ----------------


st.sidebar.title("🚀 LaptopAI")

st.sidebar.write(
"""
### About Project

This AI system predicts laptop prices using Machine Learning.

Features used:

✔ Brand  
✔ RAM  
✔ CPU  
✔ GPU  
✔ Storage  
✔ Display  
✔ Operating System  


Model:
Machine Learning Regression
"""
)



# ---------------- INPUT SECTION ----------------


st.markdown("## 🔍 Enter Laptop Details")


col1,col2,col3 = st.columns(3)



with col1:

    company = st.selectbox(
        "Brand",
        sorted(df['Company'].unique())
    )


    type_name = st.selectbox(
        "Laptop Type",
        sorted(df['TypeName'].unique())
    )


    ram = st.selectbox(
        "RAM (GB)",
        [2,4,6,8,12,16,24,32,64]
    )



with col2:

    weight = st.number_input(
        "Weight (KG)",
        min_value=0.5,
        value=1.5
    )


    touchscreen = st.selectbox(
        "Touchscreen",
        ["NO","YES"]
    )


    ips = st.selectbox(
        "IPS Display",
        ["NO","YES"]
    )



with col3:

    screen_size = st.number_input(
        "Screen Size (inch)",
        min_value=10.0,
        value=15.6
    )


    resolution = st.selectbox(
        "Screen Resolution",
        [
        '1920x1080',
        '1366x768',
        '1600x900',
        '3840x2160',
        '3840x1800',
        '2880x1800',
        '2560x1600',
        '2560x1440',
        '2304x1440'
        ]
    )



col4,col5,col6 = st.columns(3)


with col4:

    cpu = st.selectbox(
        "CPU Brand",
        sorted(df['Cpu brand'].unique())
    )


    hdd = st.selectbox(
        "HDD Storage",
        [0,128,256,512,1024,2048]
    )


with col5:

    ssd = st.selectbox(
        "SSD Storage",
        [0,128,256,512,1024]
    )


    gpu = st.selectbox(
        "GPU Brand",
        sorted(df['Gpu brand'].unique())
    )



with col6:

    os = st.selectbox(
        "Operating System",
        sorted(df['os'].unique())
    )



st.write("")



# ---------------- PREDICTION ----------------


if st.button("🚀 Predict Laptop Price"):


    touchscreen = 1 if touchscreen=="YES" else 0

    ips = 1 if ips=="YES" else 0


    x_res = int(resolution.split('x')[0])

    y_res = int(resolution.split('x')[1])


    ppi = ((x_res**2 + y_res**2)**0.5)/screen_size



    query = np.array([

        company,
        type_name,
        ram,
        weight,
        touchscreen,
        ips,
        ppi,
        cpu,
        hdd,
        ssd,
        gpu,
        os

    ],dtype=object).reshape(1,12)



    prediction = np.exp(
        pipe.predict(query)[0]
    )



    st.markdown(

    f"""

    <div class="result-card">

    <h2>🎯 AI Prediction Result</h2>

    <div class="price">
    ₹ {int(prediction):,}
    </div>

    <h3>
    Estimated Laptop Market Price
    </h3>

    </div>

    """,

    unsafe_allow_html=True

    )



# ---------------- FOOTER ----------------


st.write("")

st.markdown(
"""
<center>

### Built with ❤️ using Machine Learning

Python | Scikit-Learn | Streamlit

</center>

""",
unsafe_allow_html=True
)