# app.py
# app.py
import streamlit as st
import pandas as pd
import pulp
import gc

# -----------------------------
# Basit sabitler
# -----------------------------
FLEETS = ["A330", "A350"]
TRAINING_TYPES = ["OPC", "LPC", "OTHER"]

# Sadece gösterim için ay isimleri
MONTHS = [
    {"id": 1, "name": "Ocak"},
    {"id": 2, "name": "Şubat"},
    {"id": 3, "name": "Mart"},
    {"id": 4, "name": "Nisan"},
    {"id": 5, "name": "Mayıs"},
    {"id": 6, "name": "Haziran"},
    {"id": 7, "name": "Temmuz"},
    {"id": 8, "name": "Ağustos"},
    {"id": 9, "name": "Eylül"},
    {"id": 10, "name": "Ekim"},
    {"id": 11, "name": "Kasım"},
    {"id": 12, "name": "Aralık"},
]


# -----------------------------
# Optimizasyon fonksiyonu
# -----------------------------
def build_and_solve_model(
    sims_per_fleet: dict,
    slots_per_day: int,
    days_per_month: int,
    capacity_factor: float,
    yearly_demand: dict,
):
    """
    Çok hafif MILP modeli:
      Değişken: x[f, t, m] = filo f, eğitim tipi t, ay m için seans sayısı (integer)
      Amaç: toplam seansı maksimize et
      Kısıtlar:
        - Her filo+eğitim için yıllık toplam <= talep
        - Her filo+ay için toplam seans <= aylık sim kapasitesi
    """
    prob = pulp.LpProblem("Sim_Optimization", pulp.LpMaximize)

    # Değişkenler
    x = {}
    for f in FLEETS:
        for t in TRAINING_TYPES:
            for m in MONTHS:
                key = (f, t, m["id"])
                x[key] = pulp.LpVariable(
                    f"x_{f}_{t}_M{m['id']}",
                    lowBound=0,
                    cat="Integer",
                )

    # Amaç fonksiyonu
    prob += pulp.lpSum(x.values()), "Total_Planned_Sessions"

    # Yıllık talep kısıtları
    for f in FLEETS:
        for t in TRAINING_TYPES:
            demand_ft = yearly_demand.get((f, t), 0)
            prob += (
                pulp.lpSum(x[(f, t, m["id"])] for m in MONTHS) <= demand_ft,
                f"Yearly_Demand_{f}_{t}",
            )

    # Aylık kapasite kısıtları (aynı gün sayısı varsayımıyla)
    for f in FLEETS:
        sims = sims_per_fleet.get(f, 0)
        monthly_capacity = sims * slots_per_day * days_per_month * capacity_factor
        for m in MONTHS:
            prob += (
                pulp.lpSum(x[(f, t, m["id"])] for t in TRAINING_TYPES)
                <= monthly_capacity,
                f"Capacity_{f}_M{m['id']}",
            )

    # Çöz
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    objective_value = int(pulp.value(prob.objective) or 0)

    # Sonuçları küçük DataFrame'lere dök
    rows = []
    for f in FLEETS:
        for t in TRAINING_TYPES:
            for m in MONTHS:
                key = (f, t, m["id"])
                val = x[key].varValue
                if val is None:
                    val = 0
                rows.append(
                    {
                        "Filo": f,
                        "Eğitim Tipi": t,
                        "Ay": m["id"],
                        "Ay Adı": m["name"],
                        "Planlanan Seans": int(round(val)),
                    }
                )

    sessions_df = pd.DataFrame(rows)

    # Talep karşılama özeti (çok küçük tablo)
    summary_rows = []
    for f in FLEETS:
        for t in TRAINING_TYPES:
            demand_ft = yearly_demand.get((f, t), 0)
            planned_ft = sessions_df[
                (sessions_df["Filo"] == f)
                & (sessions_df["Eğitim Tipi"] == t)
            ]["Planlanan Seans"].sum()
            unmet = max(0, demand_ft - planned_ft)
            ratio = 0.0
            if demand_ft > 0:
                ratio = planned_ft / demand_ft * 100.0

            summary_rows.append(
                {
                    "Filo": f,
                    "Eğitim Tipi": t,
                    "Yıllık Talep": demand_ft,
                    "Planlanan": int(planned_ft),
                    "Karşılanmayan Talep": int(unmet),
                    "Karşılama Oranı %": round(ratio, 1),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    # Temizlik (RAM için)
    del prob
    del x
    gc.collect()

    return status, objective_value, sessions_df, summary_df


# -----------------------------
# Streamlit arayüzü (minimal)
# -----------------------------
st.set_page_config(page_title="Sim Optimizer (Light)", layout="wide")

st.title("✈️ Simulator Optimization")
st.write(
    "Bu sayfa, filo ve eğitim tiplerine göre bir simülatör "
    "optimizasyon modeli çalıştırır. Amaç: yıllık talebi ve aylık kapasiteyi "
    "dikkate alarak maksimum seans sayısını planlamak."
)

st.sidebar.header("Model Parametreleri")

# Sim kapasite parametreleri
st.sidebar.subheader("Sim Kapasitesi")
sims_per_fleet = {}
for f in FLEETS:
    sims_per_fleet[f] = st.sidebar.number_input(
        f"{f} sim sayısı",
        min_value=0,
        max_value=20,
        value=4 if f == "A330" else 2,
        step=1,
        key=f"sim_{f}",
    )

slots_per_day = st.sidebar.number_input(
    "Günlük slot sayısı (sim başına)",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

days_per_month = st.sidebar.number_input(
    "Ay başına gün (ortalama)",
    min_value=1,
    max_value=31,
    value=30,
    step=1,
)

capacity_factor = st.sidebar.slider(
    "Kapasite kullanım oranı",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.05,
)

# Talep parametreleri
st.sidebar.subheader("Yıllık Talep (Seans)")
yearly_demand = {}
for f in FLEETS:
    st.sidebar.markdown(f"**{f}**")
    cols = st.sidebar.columns(len(TRAINING_TYPES))
    for i, t in enumerate(TRAINING_TYPES):
        with cols[i]:
            default_val = 800 if t in ["OPC", "LPC"] else 400
            val = st.number_input(
                t,
                min_value=0,
                max_value=10000,
                value=default_val,
                step=10,
                key=f"demand_{f}_{t}",
            )
            yearly_demand[(f, t)] = val

run_button = st.sidebar.button("Optimizasyonu Çalıştır")

if run_button:
    with st.spinner("Model çözülüyor..."):
        status, objective_value, sessions_df, summary_df = build_and_solve_model(
            sims_per_fleet=sims_per_fleet,
            slots_per_day=slots_per_day,
            days_per_month=days_per_month,
            capacity_factor=capacity_factor,
            yearly_demand=yearly_demand,
        )

    st.success(f"Çözüm durumu: **{status}**")
    st.markdown(f"**Toplam planlanan seans:** `{objective_value}`")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Aylık Planlanan Seanslar")
        st.dataframe(
            sessions_df.sort_values(["Filo", "Ay", "Eğitim Tipi"]).reset_index(drop=True),
            use_container_width=True,
        )

    with col2:
        st.subheader("Talep Karşılama Özeti")
        st.dataframe(
            summary_df.sort_values(["Filo", "Eğitim Tipi"]).reset_index(drop=True),
            use_container_width=True,
        )
else:
    st.info("Soldaki parametreleri ayarlayıp **“Optimizasyonu Çalıştır”** butonuna bas.")

