import streamlit as st
import joblib
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BHP Predictor", page_icon="🏠", layout="centered")

# ── Blue theme ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background: #f0f6ff; }
.block-container { max-width: 520px !important; padding-top: 2rem !important; }
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #E6F1FB !important;
    border: 1.5px solid #85B7EB !important;
    border-radius: 8px !important;
    color: #042C53 !important;
    font-size: 15px !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #0C447C !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button {
    background: #185FA5 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.65rem !important;
}
div[data-testid="stButton"] > button:hover { background: #0C447C !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model + encoder ───────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model   = joblib.load("RF_model.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, encoder

@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_df.csv")

model, encoder = load_assets()
df             = load_data()
locations      = sorted(df["location"].unique().tolist())

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#185FA5; border-radius:14px; padding:1.5rem;
            text-align:center; margin-bottom:1.5rem;">
  <h2 style="color:white; margin:0;">🏠 Bengaluru House Price Predictor</h2>
  <p style="color:#B5D4F4; margin:6px 0 0; font-size:.88rem;">
    Enter property details to get an instant price estimate
  </p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ─────────────────────────────────────────────────────────────────────
location = st.selectbox("Location", locations)
sqft     = st.number_input("Total area (sqft)", min_value=300, max_value=10000, value=1200, step=50)

c1, c2 = st.columns(2)
bhk  = c1.selectbox("BHK",       [1, 2, 3, 4, 5], index=1)
bath = c2.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Get price estimate"):
    encoded_loc = encoder.transform([location])[0]
    X = pd.DataFrame(
        [[sqft, bath, bhk, encoded_loc]],
        columns=["total_sqft", "bath", "bhk", "encoded_loc"]
    )
    price = model.predict(X)[0]
    low   = price * 0.88
    high  = price * 1.12
    pps   = price * 1e5 / sqft

    st.markdown(f"""
    <div style="background:#E6F1FB; border:1.5px solid #378ADD; border-radius:14px;
                padding:1.5rem; text-align:center; margin-top:1rem;">
      <p style="font-size:11px; font-weight:600; color:#185FA5;
                text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;">
        Estimated price
      </p>
      <p style="font-size:2.5rem; font-weight:700; color:#042C53; margin-bottom:4px;">
        ₹ {price:.2f} L
      </p>
      <p style="font-size:13px; color:#378ADD; margin-bottom:14px;">
        {bhk} BHK &nbsp;·&nbsp; {sqft:,} sqft &nbsp;·&nbsp; {bath} bath &nbsp;·&nbsp; {location}
      </p>
      <div style="display:flex; justify-content:center; gap:2rem;
                  border-top:1px solid #B5D4F4; padding-top:12px;">
        <div>
          <p style="font-size:11px; color:#185FA5; font-weight:600; margin:0;">LOW</p>
          <p style="font-size:1rem; color:#042C53; font-weight:600; margin:0;">₹ {low:.1f} L</p>
        </div>
        <div>
          <p style="font-size:11px; color:#185FA5; font-weight:600; margin:0;">PER SQFT</p>
          <p style="font-size:1rem; color:#042C53; font-weight:600; margin:0;">₹ {pps:,.0f}</p>
        </div>
        <div>
          <p style="font-size:11px; color:#185FA5; font-weight:600; margin:0;">HIGH</p>
          <p style="font-size:1rem; color:#042C53; font-weight:600; margin:0;">₹ {high:.1f} L</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
