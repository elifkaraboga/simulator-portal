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
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pulp

# -----------------------------
# Helper functions
# -----------------------------
def project_pilots(current, monthly_in, monthly_out, months):
    """Basit projeksiyon: her ay sabit in/out varsayalım."""
values = []
total = current
for _ in range(months):
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


def optimize_for_sim_count(total_demand, sim_count,
slots_per_day, days_per_month,
utilization, other_trainings, c_out):
    """Verilen simülatör sayısı için aylık talep ve kapasiteye bakarak outsourcing miktarını minimize eden basit bir MILP kurar."""
T = len(total_demand)
capacity = compute_capacity(sim_count, slots_per_day, days_per_month, utilization, other_trainings)

# MILP modeli
model = pulp.LpProblem("Sim_Capacity_Optimization", pulp.LpMinimize)

# Değişkenler: her ay için served ve outsource (integer)
served = pulp.LpVariable.dicts("served", range(T), lowBound=0, cat="Integer")
outsource = pulp.LpVariable.dicts("outsource", range(T), lowBound=0, cat="Integer")

# Amaç fonksiyonu: outsourcing maliyetini minimize et
model += c_out * pulp.lpSum(outsource[t] for t in range(T))

# Kısıtlar
for t in range(T):
# İçeride yapılan seans kapasiteyi aşamaz
model += served[t] <= capacity

# Toplam karşılanan talep >= gerçek talep
model += served[t] + outsource[t] >= float(total_demand[t])

# Modeli çöz
model.solve(pulp.PULP_CBC_CMD(msg=False))

served_vals = np.array([served[t].value() for t in range(T)])
outsource_vals = np.array([outsource[t].value() for t in range(T)])
total_out = float(outsource_vals.sum())

# İlk outsourcing yapılan ay (varsa)
first_out_month_idx = None
for t in range(T):
if outsource_vals[t] > 0.1:
first_out_month_idx = t
break

return {
"sim_count": sim_count,
"capacity_per_month": capacity,
"total_outsource": total_out,
"served": served_vals,
"outsource": outsource_vals,
"first_out_idx": first_out_month_idx,
}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Simulator Forecast Portal", layout="wide")

st.title("✈️ Simulator Capacity & Forecast Portal")
st.write(
    "Bu prototip, pilot sayıları ve simülatör kapasitesine göre önümüzdeki aylarda recurrent talebi ve kapasiteyi karşılaştırır, ayrıca farklı sim sayıları için MILP tabanlı bir outsourcing optimizasyonu çalıştırır."
)

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")

horizon_months = st.sidebar.slider("Forecast horizon (months)", 6, 36, 12)

# Date range
start_date = st.sidebar.date_input("Start month", datetime.today())
months = [(start_date + relativedelta(months=i)).strftime("%Y-%m") for i in range(horizon_months)]

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

st.sidebar.subheader("Simulator Capacity (Base)")
sim_count = st.sidebar.number_input("Total simulators (all fleets)", min_value=1, value=6)
slots_per_day = st.sidebar.number_input("Slots per simulator per day", min_value=1, value=5)
days_per_month = st.sidebar.number_input("Days per month (average)", min_value=1, max_value=31, value=30)
utilization = st.sidebar.slider("Target utilization", 0.5, 1.0, 0.8)
other_trainings = st.sidebar.number_input("Other training load (sessions/month)", min_value=0, value=50)

st.sidebar.subheader("Optimization Settings (MILP)")
sim_min = st.sidebar.number_input("Min simulators (scenario range)", min_value=1, value=6)
sim_max = st.sidebar.number_input("Max simulators (scenario range)", min_value=sim_min, value=8)
c_out = st.sidebar.number_input("Outsourcing cost per session (C_out)", min_value=0.0, value=1000.0, step=100.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Forecast & Optimization")

# -----------------------------
# Main logic
# -----------------------------
if run_button:
# 1) Pilot projections
a330_pilots = project_pilots(curr_a330, in_a330, out_a330, horizon_months)
a350_pilots = project_pilots(curr_a350, in_a350, out_a350, horizon_months)
dual_pilots = project_pilots(curr_dual, in_dual, out_dual, horizon_months)

# DUAL pilotları basitçe yarı yarıya dağıtalım
a330_effective = a330_pilots + 0.5 * dual_pilots
a350_effective = a350_pilots + 0.5 * dual_pilots

# 2) Demand (recurrent)
demand_a330 = compute_recurrent_demand(a330_effective)
demand_a350 = compute_recurrent_demand(a350_effective)
total_demand = demand_a330 + demand_a350

# 3) Capacity (base scenario: sim_count)
monthly_capacity = compute_capacity(
sim_count=sim_count,
slots_per_day=slots_per_day,
days_per_month=days_per_month,
utilization=utilization,
other_trainings=other_trainings,
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
"Capacity (base sims)": capacity_series.round(1),
})

df["Utilization %"] = (df["Total Demand"] / df["Capacity (base sims)"] * 100).round(1)
df["Deficit"] = (df["Total Demand"] - df["Capacity (base sims)"]).round(1)

# 5) Show summary
st.subheader("Forecast Summary Table (Base Scenario)")
st.dataframe(df, use_container_width=True)

# 6) Plot demand vs capacity
st.subheader("Total Demand vs Capacity (Base Simulator Count)")
chart_df = df[["Month", "Total Demand", "Capacity (base sims)"]].set_index("Month")
st.line_chart(chart_df)

# 7) Highlight first capacity breach, if any
breach_rows = df[df["Deficit"] > 0]
if not breach_rows.empty:
first_breach = breach_rows.iloc[0]
st.error(
f"⚠️ Capacity breach (base sims) starts at **{first_breach['Month']}** "
f"(Demand={first_breach['Total Demand']}, Capacity={first_breach['Capacity (base sims)']})."
)
else:
st.success("✅ No capacity breach with the current simulator count within the selected horizon.")

# -----------------------------
# MILP Optimization Section
# -----------------------------
st.subheader("MILP-based Optimization: Simulator Count Scenarios")

if sim_max < sim_min:
st.error("Max simulators value must be >= Min simulators value.")
else:
results = []
for s in range(int(sim_min), int(sim_max) + 1):
res = optimize_for_sim_count(
total_demand=total_demand,
sim_count=s,
slots_per_day=slots_per_day,
days_per_month=days_per_month,
utilization=utilization,
other_trainings=other_trainings,
c_out=c_out,
)
results.append(res)

opt_rows = []
for res in results:
if res["first_out_idx"] is not None:
first_out_month = months[res["first_out_idx"]]
else:
first_out_month = "No outsourcing"

opt_rows.append({
"Simulators": res["sim_count"],
"Capacity per month": round(res["capacity_per_month"], 1),
"Total outsourcing (sessions)": int(res["total_outsource"]),
"First outsourcing month": first_out_month,
"Total outsourcing cost": int(res["total_outsource"] * c_out),
})

opt_df = pd.DataFrame(opt_rows)
st.dataframe(opt_df, use_container_width=True)

st.subheader("Total Outsourcing Cost vs Simulator Count")
cost_chart = opt_df[["Simulators", "Total outsourcing cost"]].set_index("Simulators")
st.bar_chart(cost_chart)

# En düşük maliyetli sim sayısını vurgula
best_row = opt_df.loc[opt_df["Total outsourcing cost"].idxmin()]
st.success(
f"💡 Minimum outsourcing cost is achieved with **{int(best_row['Simulators'])} simulators** "
f"(Total outsourcing cost = {int(best_row['Total outsourcing cost'])})."
)

else:
st.info("Soldaki parametreleri doldurup **Run Forecast & Optimization** butonuna basarak sonuçları görebilirsin.")

