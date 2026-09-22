import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Mastitis Risk Predictor (Demo)", layout="wide")

# ----------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# Encodes domain-informed relationships between behavioral/visual/production
# signals and SCC/EC, since real lab-collected SCC/EC data is not yet available.
# Once real data is collected, replace this function with a CSV loader.
# ----------------------------------------------------------------------------
def generate_synthetic_data(n_samples=4000, seed=42):
    rng = np.random.RandomState(seed)

    dirt_score = rng.uniform(0, 100, n_samples)                     # CCTV hygiene score
    bcs = np.clip(rng.normal(3.0, 0.7, n_samples), 1.0, 5.0)        # Body condition score
    parity = rng.randint(0, 9, n_samples)                            # Number of prior calvings
    age = np.clip(1.5 + parity * 0.9 + rng.normal(0, 0.6, n_samples), 1.5, 10.0)

    # Behavior split (accelerometer-derived activity classification)
    pct_lying = np.clip(rng.normal(48, 10, n_samples), 10, 85)
    pct_eating = np.clip(rng.normal(28, 7, n_samples), 5, 60)
    pct_walking = np.clip(100 - pct_lying - pct_eating + rng.normal(0, 3, n_samples), 2, 60)
    total = pct_lying + pct_eating + pct_walking
    pct_lying, pct_eating, pct_walking = (pct_lying / total * 100,
                                           pct_eating / total * 100,
                                           pct_walking / total * 100)

    # Milk yield: cows have a parity/age-adjusted "expected" yield; deficits below
    # that expectation are a mastitis signal (mastitis suppresses yield)
    expected_yield = 28 - 0.6 * parity - 0.3 * (age - 2)
    milk_yield = np.clip(expected_yield + rng.normal(0, 3, n_samples), 2, 45)

    # --- Build log(SCC) from domain rules ---
    log_scc = np.full(n_samples, np.log(65000))
    log_scc += (dirt_score - 40) * 0.009                              # dirt up -> SCC up
    log_scc += np.where(bcs < 2.5, (2.5 - bcs) * 0.55, 0)             # underconditioned -> SCC up
    log_scc += np.where(bcs > 3.5, (bcs - 3.5) * 0.45, 0)             # overconditioned -> SCC up (U-shape)
    log_scc += np.clip(pct_lying - 55, 0, None) * 0.030               # excess lying -> SCC up
    log_scc += np.clip(22 - pct_eating, 0, None) * 0.045              # low eating -> SCC up
    log_scc += np.clip(15 - pct_walking, 0, None) * 0.025             # low activity -> SCC up
    log_scc += parity * 0.09                                          # higher parity -> SCC up
    log_scc += (age - 3) * 0.025                                      # older -> SCC up (mild)

    yield_deficit = np.clip(expected_yield - milk_yield, 0, None)
    log_scc += yield_deficit * 0.10                                   # milk yield down -> SCC up

    log_scc += rng.normal(0, 0.32, n_samples)                         # biological noise
    scc = np.clip(np.exp(log_scc), 15000, 3_000_000)

    # EC correlates with SCC but has its own independent noise
    ec = np.clip(3.6 + (np.log10(scc) - 4.8) * 1.6 + rng.normal(0, 0.22, n_samples), 3.0, 10.0)

    return pd.DataFrame({
        "dirt_score": dirt_score, "bcs": bcs, "pct_lying": pct_lying,
        "pct_eating": pct_eating, "pct_walking": pct_walking,
        "milk_yield": milk_yield, "age": age, "parity": parity,
        "scc": scc, "ec": ec
    })

FEATURES = ["dirt_score", "bcs", "pct_lying", "pct_eating", "pct_walking",
            "milk_yield", "age", "parity"]

# ----------------------------------------------------------------------------
# 2. MODEL TRAINING (cached so it only runs once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def train_models():
    df = generate_synthetic_data(4000, seed=42)
    X = df[FEATURES]
    y_scc = np.log1p(df["scc"])
    y_ec = df["ec"]

    Xtr, Xte, ytr_s, yte_s, ytr_e, yte_e = train_test_split(
        X, y_scc, y_ec, test_size=0.2, random_state=42
    )

    model_scc = GradientBoostingRegressor(random_state=42).fit(Xtr, ytr_s)
    model_ec = GradientBoostingRegressor(random_state=42).fit(Xtr, ytr_e)

    pred_scc_test = np.expm1(model_scc.predict(Xte))
    metrics = {
        "scc_r2_log": r2_score(yte_s, model_scc.predict(Xte)),
        "scc_mae": mean_absolute_error(np.expm1(yte_s), pred_scc_test),
        "ec_r2": r2_score(yte_e, model_ec.predict(Xte)),
        "ec_mae": mean_absolute_error(yte_e, model_ec.predict(Xte)),
    }
    return model_scc, model_ec, df, metrics

model_scc, model_ec, train_df, metrics = train_models()

# ----------------------------------------------------------------------------
# 3. PRESETS
# ----------------------------------------------------------------------------
PRESETS = {
    "Healthy baseline": dict(dirt_score=20, bcs=3.0, pct_lying=45, pct_eating=30,
                              pct_walking=25, milk_yield=27, age=4, parity=2),
    "Subclinical example": dict(dirt_score=55, bcs=2.6, pct_lying=58, pct_eating=22,
                                  pct_walking=20, milk_yield=19, age=5, parity=4),
    "Clinical example": dict(dirt_score=85, bcs=2.0, pct_lying=70, pct_eating=12,
                               pct_walking=8, milk_yield=10, age=7, parity=6),
}

if "vals" not in st.session_state:
    st.session_state.vals = dict(PRESETS["Healthy baseline"])

def apply_preset(name):
    st.session_state.vals = dict(PRESETS[name])

# ----------------------------------------------------------------------------
# 4. SIDEBAR — INPUT CONTROLS
# ----------------------------------------------------------------------------
st.sidebar.title("🐄 Cow Input Parameters")

st.sidebar.markdown("**Quick presets**")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Healthy"):
    apply_preset("Healthy baseline")
if c2.button("Subclinical"):
    apply_preset("Subclinical example")
if c3.button("Clinical"):
    apply_preset("Clinical example")

st.sidebar.markdown("---")
v = st.session_state.vals

dirt_score = st.sidebar.slider("Dirt Score (CCTV hygiene, 0=clean, 100=very dirty)",
                                0, 100, int(v["dirt_score"]))
bcs = st.sidebar.slider("Body Condition Score (1=very thin, 5=obese)",
                         1.0, 5.0, float(v["bcs"]), step=0.1)

st.sidebar.markdown("**Behavior split (accelerometer-derived, %)**")
pct_lying = st.sidebar.slider("% Time Lying / Sleeping", 0, 100, int(v["pct_lying"]))
pct_eating = st.sidebar.slider("% Time Eating", 0, 100, int(v["pct_eating"]))
pct_walking = st.sidebar.slider("% Time Walking / Active", 0, 100, int(v["pct_walking"]))
behavior_total = pct_lying + pct_eating + pct_walking
if behavior_total == 0:
    behavior_total = 1
pct_lying_n = pct_lying / behavior_total * 100
pct_eating_n = pct_eating / behavior_total * 100
pct_walking_n = pct_walking / behavior_total * 100
st.sidebar.caption(f"Raw total: {behavior_total}% -> normalized to 100% automatically.")

st.sidebar.markdown("---")
milk_yield = st.sidebar.slider("Milk Yield (liters/day, farmer-reported)",
                                2.0, 45.0, float(v["milk_yield"]), step=0.5)
age = st.sidebar.slider("Age (years)", 1.5, 10.0, float(v["age"]), step=0.5)
parity = st.sidebar.slider("Parity (number of prior calvings)", 0, 8, int(v["parity"]))

# ----------------------------------------------------------------------------
# 5. PREDICTION
# ----------------------------------------------------------------------------
input_row = pd.DataFrame([{
    "dirt_score": dirt_score, "bcs": bcs, "pct_lying": pct_lying_n,
    "pct_eating": pct_eating_n, "pct_walking": pct_walking_n,
    "milk_yield": milk_yield, "age": age, "parity": parity
}])

pred_scc = float(np.expm1(model_scc.predict(input_row)[0]))
pred_scc = max(pred_scc, 0)
pred_ec = float(model_ec.predict(input_row)[0])

if pred_scc < 200_000:
    risk_label, risk_color = "Healthy", "#2ecc71"
elif pred_scc < 500_000:
    risk_label, risk_color = "Subclinical Mastitis Risk", "#f1c40f"
else:
    risk_label, risk_color = "Clinical Mastitis Risk", "#e74c3c"

# ----------------------------------------------------------------------------
# 6. MAIN PANEL
# ----------------------------------------------------------------------------
st.title("Mastitis Risk Prediction — Interactive Demo")
st.caption("Synthetic-data-trained regression model. Adjust the sidebar inputs to see "
           "how predicted SCC/EC and mastitis risk respond.")

col1, col2, col3 = st.columns(3)
col1.metric("Predicted SCC", f"{pred_scc:,.0f} cells/mL")
col2.metric("Predicted EC", f"{pred_ec:.2f} mS/cm")
col3.markdown(
    f"""<div style="background-color:{risk_color};padding:14px;border-radius:10px;
    text-align:center;color:white;font-weight:bold;font-size:20px;margin-top:8px;">
    {risk_label}</div>""",
    unsafe_allow_html=True
)

st.markdown("### SCC Risk Gauge")
gauge_max = 1_000_000
gauge_frac = min(pred_scc / gauge_max, 1.0)
st.progress(gauge_frac)
st.caption("0" + " " * 60 + "200,000 (subclinical threshold)" + " " * 40 +
           "500,000 (clinical threshold)" + " " * 30 + "1,000,000+")

# --- Rule-based explanation of top contributing factors ---
st.markdown("### What's driving this prediction?")
reasons = []
if dirt_score > 55:
    reasons.append(f"High dirt/moisture score ({dirt_score}) increases bacterial exposure.")
if bcs < 2.5:
    reasons.append(f"Low body condition score ({bcs:.1f}) suggests energy deficit / weaker immunity.")
if bcs > 3.5:
    reasons.append(f"High body condition score ({bcs:.1f}) suggests over-conditioning / metabolic stress.")
if pct_lying_n > 55:
    reasons.append(f"Elevated lying time ({pct_lying_n:.0f}%) is a classic early illness sign.")
if pct_eating_n < 22:
    reasons.append(f"Reduced eating time ({pct_eating_n:.0f}%) suggests reduced appetite.")
if pct_walking_n < 15:
    reasons.append(f"Low activity level ({pct_walking_n:.0f}%) suggests lethargy.")
expected_yield_ref = 28 - 0.6 * parity - 0.3 * (age - 2)
if milk_yield < expected_yield_ref - 3:
    reasons.append(f"Milk yield ({milk_yield:.1f} L/day) is well below the "
                    f"expected ~{expected_yield_ref:.1f} L/day for this cow's age/parity.")
if parity >= 5:
    reasons.append(f"High parity ({parity}) is associated with higher baseline SCC.")

if reasons:
    for r in reasons:
        st.markdown(f"- {r}")
else:
    st.markdown("- All inputs are within healthy reference ranges. No major risk drivers detected.")

# --- 7-day trend simulation ---
st.markdown("### Simulated 7-Day Trend (illustrative)")
days = np.arange(1, 8)
trend_scc = pred_scc * (0.55 + 0.45 * (days / 7) ** 1.5)
trend_df = pd.DataFrame({"Day": days, "Projected SCC": trend_scc})
st.line_chart(trend_df.set_index("Day"))
st.caption("Illustrates how SCC might trend upward over 7 days if current conditions persist — "
           "the goal of the real system is to catch this rise before clinical symptoms appear.")

# --- Model diagnostics ---
with st.expander("Model Diagnostics (synthetic training data)"):
    st.write(f"SCC model R² (log space): {metrics['scc_r2_log']:.3f}")
    st.write(f"SCC model MAE: {metrics['scc_mae']:,.0f} cells/mL")
    st.write(f"EC model R²: {metrics['ec_r2']:.3f}")
    st.write(f"EC model MAE: {metrics['ec_mae']:.3f} mS/cm")
    st.write("Training set size:", len(train_df))
    st.download_button("Download synthetic training dataset (CSV)",
                        train_df.to_csv(index=False), file_name="synthetic_mastitis_data.csv")

st.markdown("---")
st.caption("⚠️ This demo uses synthetic data with domain-informed rules "
           "(e.g., lower milk yield, higher dirt score, and more parity all raise predicted SCC). "
           "Replace generate_synthetic_data() with real collected SCC/EC data once available.")
