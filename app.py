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
# app.py
import streamlit as st
import pandas as pd
import pulp
import gc

# ----------------------------
# Sabit parametreler
# ----------------------------

FLEETS = ["A330", "A350"]
TRAINING_TYPES = ["OPC", "LPC", "OTHER"]  # İstersen buraya yeni eğitim tipleri ekleyebilirsin

MONTHS = [
    {"id": 1, "name": "Ocak",      "days": 31},
    {"id": 2, "name": "Şubat",     "days": 28},  # Artık yıllar için istersen dinamik yaparsın
    {"id": 3, "name": "Mart",      "days": 31},
    {"id": 4, "name": "Nisan",     "days": 30},
    {"id": 5, "name": "Mayıs",     "days": 31},
    {"id": 6, "name": "Haziran",   "days": 30},
    {"id": 7, "name": "Temmuz",    "days": 31},
    {"id": 8, "name": "Ağustos",   "days": 31},
    {"id": 9, "name": "Eylül",     "days": 30},
    {"id": 10, "name": "Ekim",     "days": 31},
    {"id": 11, "name": "Kasım",    "days": 30},
    {"id": 12, "name": "Aralık",   "days": 31},
]

# Tipik THY senaryosu: 4 adet A330 sim, 2 adet A350 sim
DEFAULT_SIMS_PER_FLEET = {
    "A330": 4,
    "A350": 2,
}

SLOTS_PER_DAY = 5           # her sim günde 5 slot
HOURS_PER_SLOT = 4          # her slot 4 saat


# ----------------------------
# Yardımcı fonksiyonlar
# ----------------------------

def build_and_solve_model(
    year: int,
    sims_per_fleet: dict,
    capacity_factor: float,
    yearly_demand: dict,
):
    """
    Hafif MIP modeli kurar ve çözer.
    Decision variable:
        x[f, t, m] = ilgili yıl içindeki ay m'de, filo f ve eğitim tipi t için planlanan seans sayısı (integer)
    Amaç:
        Toplam planlanan seans sayısını maksimize etmek.
    Kısıtlar:
        1) Her filo+eğitim için yıllık toplam seans <= talep
        2) Her filo+ay için sim kapasitesi sınırı
    """
    # Modeli kur
    prob = pulp.LpProblem(f"Sim_Optimization_{year}", pulp.LpMaximize)

    # Karar değişkenleri
    x = {}
    for f in FLEETS:
        for t in TRAINING_TYPES:
            for m in MONTHS:
                key = (f, t, m["id"])
                # Seans sayısı negatif olamaz, integer olsun
                x[key] = pulp.LpVariable(
                    f"x_{f}_{t}_M{m['id']}",
                    lowBound=0,
                    cat="Integer"
                )

    # Amaç fonksiyonu: toplam seansı maksimize et
    prob += pulp.lpSum(x.values()), "Total_Planned_Sessions"

    # 1) Yıllık talep kısıtları: sum_m x[f, t, m] <= yearly_demand[f, t]
    for f in FLEETS:
        for t in TRAINING_TYPES:
            demand_ft = yearly_demand.get((f, t), 0)
            prob += (
                pulp.lpSum(x[(f, t, m["id"])] for m in MONTHS) <= demand_ft,
                f"Yearly_Demand_{f}_{t}"
            )

    # 2) Her filo + ay için sim kapasitesi kısıtı:
    # sum_t x[f, t, m] <= sims[f] * slots_per_day * days_in_month * capacity_factor
    for f in FLEETS:
        sims = sims_per_fleet.get(f, 0)
        for m in MONTHS:
            month_days = m["days"]
            capacity_sessions = sims * SLOTS_PER_DAY * month_days * capacity_factor
            prob += (
                pulp.lpSum(x[(f, t, m["id"])] for t in TRAINING_TYPES)
                <= capacity_sessions,
                f"Sim_Capacity_{f}_M{m['id']}"
            )

    # Modeli çöz
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    objective_value = pulp.value(prob.objective)

    # Sonuçları DataFrame'e dök
    rows = []
    for f in FLEETS:
        for t in TRAINING_TYPES:
            for m in MONTHS:
                key = (f, t, m["id"])
                value = x[key].varValue if x[key].varValue is not None else 0
                rows.append({
                    "Filo": f,
                    "Eğitim Tipi": t,
                    "Ay": m["id"],
                    "Ay Adı": m["name"],
                    "Planlanan Seans": int(round(value)),
                })

    sessions_df = pd.DataFrame(rows)

    # Filo + ay bazlı kapasite ve doluluk
    util_rows = []
    for f in FLEETS:
        sims = sims_per_fleet.get(f, 0)
        for m in MONTHS:
            month_days = m["days"]
            capacity_sessions = sims * SLOTS_PER_DAY * month_days * capacity_factor
            planned_sessions = sessions_df[
                (sessions_df["Filo"] == f) & (sessions_df["Ay"] == m["id"])
            ]["Planlanan Seans"].sum()
            utilization = 0.0
            if capacity_sessions > 0:
                utilization = planned_sessions / capacity_sessions * 100

            util_rows.append({
                "Filo": f,
                "Ay": m["id"],
                "Ay Adı": m["name"],
                "Kapasite (Seans)": capacity_sessions,
                "Planlanan Seans": planned_sessions,
                "Doluluk %": round(utilization, 1),
            })

    utilization_df = pd.DataFrame(util_rows)

    # Talep karşılama özeti
    demand_rows = []
    for f in FLEETS:
        for t in TRAINING_TYPES:
            demand_ft = yearly_demand.get((f, t), 0)
            planned_ft = sessions_df[
                (sessions_df["Filo"] == f) & (sessions_df["Eğitim Tipi"] == t)
            ]["Planlanan Seans"].sum()
            unmet = demand_ft - planned_ft
            demand_rows.append({
                "Filo": f,
                "Eğitim Tipi": t,
                "Yıllık Talep": demand_ft,
                "Planlanan": planned_ft,
                "Karşılanmayan Talep": max(0, unmet),
                "Karşılama Oranı %": round((planned_ft / demand_ft * 100) if demand_ft > 0 else 0, 1),
            })

    demand_df = pd.DataFrame(demand_rows)

    # Bellek temizliği (Streamlit'in RAM'ini şişirmemek için)
    del prob
    del x
    gc.collect()

    return status, objective_value, sessions_df, utilization_df, demand_df


# ----------------------------
# Streamlit Arayüzü
# ----------------------------

def main():
    st.set_page_config(
        page_title="Simülatör Kaynak Optimizasyonu (Hafif Model)",
        layout="wide",
    )

    st.title("✈️ Simülatör Kaynak Optimizasyonu (Hafif Model)")
    st.markdown(
        """
Bu arayüz, **filo × eğitim tipi × ay** düzeyinde basitleştirilmiş bir optimizasyon modeli uygular.

- Pilot bazına inmez → **çok daha hafif** ve Streamlit dostu 🧠
- Yıllık talebi ve kapasiteyi vererek, her ay için **maksimum planlanabilir seans** sayısını hesaplar.
        """
    )

    st.sidebar.header("Genel Parametreler")

    year = st.sidebar.number_input("Planlama Yılı", min_value=2024, max_value=2100, value=2025, step=1)

    capacity_factor = st.sidebar.slider(
        "Sim Kapasite Kullanım Oranı",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Örneğin 0.8 = yıllık/aylık sim kapasitesinin %80'i kullanılabilir kabul edilir.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simülatör Sayıları (Filo Bazında)")
    sims_per_fleet = {}
    for f in FLEETS:
        sims_per_fleet[f] = st.sidebar.number_input(
            f"{f} Sim Sayısı",
            min_value=0,
            max_value=20,
            value=DEFAULT_SIMS_PER_FLEET.get(f, 0),
            step=1,
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Yıllık Talep (Seans Sayısı)")

    yearly_demand = {}
    for f in FLEETS:
        st.sidebar.markdown(f"**{f}**")
        cols = st.sidebar.columns(len(TRAINING_TYPES))
        for i, t in enumerate(TRAINING_TYPES):
            with cols[i]:
                default_val = 800 if t in ["OPC", "LPC"] else 400  # sadece örnek default
                val = st.number_input(
                    f"{t}",
                    min_value=0,
                    max_value=10000,
                    value=default_val,
                    step=10,
                    key=f"demand_{f}_{t}",
                    help=f"{f} filosu için yıllık {t} seans talebi (toplam seans sayısı).",
                )
                yearly_demand[(f, t)] = val

    st.markdown("### 🔧 Optimizasyon")

    if st.button("Optimizasyonu Çalıştır"):
        with st.spinner("Model çözülüyor, lütfen bekleyin..."):
            status, objective_value, sessions_df, utilization_df, demand_df = build_and_solve_model(
                year=year,
                sims_per_fleet=sims_per_fleet,
                capacity_factor=capacity_factor,
                yearly_demand=yearly_demand,
            )

        st.success(f"Model çözümü tamamlandı. Çözüm durumu: **{status}**")

        st.markdown(f"**Toplam Planlanan Seans (Amaç Fonksiyonu):** `{int(objective_value)}`")

        # Sonuçları sekmelerde göster
        tab1, tab2, tab3 = st.tabs(["📅 Aylık Seans Dağılımı", "📊 Sim Doluluk Oranları", "📈 Talep Karşılama"])

        with tab1:
            st.markdown("#### Aylık Seans Dağılımı (Filo × Eğitim Tipi × Ay)")
            st.dataframe(
                sessions_df.sort_values(["Filo", "Ay", "Eğitim Tipi"]).reset_index(drop=True),
                use_container_width=True,
            )

        with tab2:
            st.markdown("#### Sim Kapasite ve Doluluk Oranları")
            st.dataframe(
                utilization_df.sort_values(["Filo", "Ay"]).reset_index(drop=True),
                use_container_width=True,
            )

            st.markdown("##### Doluluk Oranı Grafiği (Filo Bazında)")
            for f in FLEETS:
                st.markdown(f"**{f}**")
                df_plot = utilization_df[utilization_df["Filo"] == f].set_index("Ay Adı")
                st.bar_chart(df_plot["Doluluk %"])

        with tab3:
            st.markdown("#### Talep Karşılama Özeti")
            st.dataframe(
                demand_df.sort_values(["Filo", "Eğitim Tipi"]).reset_index(drop=True),
                use_container_width=True,
            )

            st.markdown(
                """
- **Yıllık Talep**: İlgili filo + eğitim tipi için girilen toplam yıllık seans talebi  
- **Planlanan**: Modelin planladığı toplam seans sayısı  
- **Karşılanmayan Talep**: Talep eksi planlanan (varsa)  
- **Karşılama Oranı**: Planlanan / Talep
                """
            )
    else:
        st.info("Sol taraftan parametreleri ayarlayıp **“Optimizasyonu Çalıştır”** butonuna basabilirsin.")


if __name__ == "__main__":
    main()
