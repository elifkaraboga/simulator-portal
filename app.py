import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# -----------------------------
#  Helper functions
# -----------------------------
def project_pilots(current, monthly_in, monthly_out, months):
    """Basit projeksiyon: her ay sabit in/out varsayalım."""
    values = []
    total = current
    for m in range(months):
        total = total + monthly_in - monthly_out
        if total < 0:
            total = 0
        values.append(total)
    return np.array(values)


def compute_recurrent_demand(pilots, cycle_months=6):
    """6 ayda 1 recurrent varsayımıyla, talep ≈ pilot_sayısı / 6."""
    return pilots / cycle_months


def compute_capacity(sim_count, slots_per_day, days_per_month, utilization, other_trainings):
    raw = sim_count * slots_per_day * days_per_month * utilization
    effective = raw - other_trainings
    return max(effective, 0)


# -----------------------------
#  Streamlit UI
# -----------------------------
st.set_page_config(page_title="Simulator Forecast Portal", layout="wide")

st.title("✈️ Simulator Capacity & Forecast Portal (MVP)")
st.write("Basit bir prototip: pilot sayıları ve sim kapasitesine göre "
         "önümüzdeki aylarda recurrent talebi ve kapasiteyi karşılaştıralım.")

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")

horizon_months = st.sidebar.slider("Forecast horizon (months)", 6, 36, 12)

# Date range
start_date = st.sidebar.date_input("Start month", datetime.today())
months = [ (start_date + relativedelta(months=i)).strftime("%Y-%m") for i in range(horizon_months) ]

st.sidebar.subheader("Pilot Counts (current)")
curr_a330 = st.sidebar.number_input("Current A330 pilots", min_value=0, value=400, step=10)
curr_a350 = st.sidebar.number_input("Current A350 pilots", min_value=0, value=300, step=10)
curr_dual = st.sidebar.number_input("Current DUAL (A330+A350) pilots", min_value=0, value=200, step=10)

st.sidebar.subheader("Monthly In / Out (average)")
in_a330 = st.sidebar.number_input("Monthly A330 inflow", min_value=0, value=5)
out_a330 = st.sidebar.number_input("Monthly A330 outflow", min_value=0, value=2)

in_a350 = st.sidebar.number_input("Monthly A350 inflow", min_value=0, value=8)
out_a350 = st.sidebar.number_input("Monthly A350 outflow", min_value=0, value=3)

in_dual = st.sidebar.number_input("Monthly DUAL inflow", min_value=0, value=3)
out_dual = st.sidebar.number_input("Monthly DUAL outflow", min_value=0, value=1)

st.sidebar.subheader("Simulator Capacity")
sim_count = st.sidebar.number_input("Total simulators (all fleets)", min_value=1, value=6)
slots_per_day = st.sidebar.number_input("Slots per simulator per day", min_value=1, value=5)
days_per_month = st.sidebar.number_input("Days per month (average)", min_value=1, max_value=31, value=30)
utilization = st.sidebar.slider("Target utilization", 0.5, 1.0, 0.8)
other_trainings = st.sidebar.number_input("Other training load (sessions/month)", min_value=0, value=50)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Forecast")

# -----------------------------
#  Main logic
# -----------------------------
if run_button:
    # 1) Pilot projections
    a330_pilots = project_pilots(curr_a330, in_a330, out_a330, horizon_months)
    a350_pilots = project_pilots(curr_a350, in_a350, out_a350, horizon_months)
    dual_pilots = project_pilots(curr_dual, in_dual, out_dual, horizon_months)

    # DUAL pilotları basitçe yarı yarıya dağıtalım (isteğe göre değiştirilebilir)
    a330_effective = a330_pilots + 0.5 * dual_pilots
    a350_effective = a350_pilots + 0.5 * dual_pilots

    # 2) Demand (recurrent)
    demand_a330 = compute_recurrent_demand(a330_effective)
    demand_a350 = compute_recurrent_demand(a350_effective)
    total_demand = demand_a330 + demand_a350

    # 3) Capacity (same for all months in this simple MVP)
    monthly_capacity = compute_capacity(
        sim_count=sim_count,
        slots_per_day=slots_per_day,
        days_per_month=days_per_month,
        utilization=utilization,
        other_trainings=other_trainings
    )

    capacity_series = np.array([monthly_capacity] * horizon_months)

    # 4) Build DataFrame
    df = pd.DataFrame({
        "Month": months,
        "A330 pilots (eff)": a330_effective.round(1),
        "A350 pilots (eff)": a350_effective.round(1),
        "Demand A330 (recurrent)": demand_a330.round(1),
        "Demand A350 (recurrent)": demand_a350.round(1),
        "Total Demand": total_demand.round(1),
        "Capacity": capacity_series.round(1),
    })

    df["Utilization %"] = (df["Total Demand"] / df["Capacity"] * 100).round(1)
    df["Deficit"] = (df["Total Demand"] - df["Capacity"]).round(1)

    # 5) Show summary
    st.subheader("Forecast Summary Table")
    st.dataframe(df, use_container_width=True)

    # 6) Plot demand vs capacity
    st.subheader("Total Demand vs Capacity")
    chart_df = df[["Month", "Total Demand", "Capacity"]].set_index("Month")
    st.line_chart(chart_df)

    # 7) Highlight first capacity breach, if any
    breach_rows = df[df["Deficit"] > 0]
    if not breach_rows.empty:
        first_breach = breach_rows.iloc[0]
        st.error(
            f"⚠️ Capacity breach starts at **{first_breach['Month']}** "
            f"(Demand={first_breach['Total Demand']}, Capacity={first_breach['Capacity']})."
        )
    else:
        st.success("✅ No capacity breach within the selected horizon.")

else:
    st.info("Soldaki parametreleri doldurup **Run Forecast** butonuna basarak sonuçları görebilirsin.")
