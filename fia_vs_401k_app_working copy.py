from pathlib import Path

# Define the file path
final_script_path = Path("/mnt/data/fia_vs_401k_app_final.py")

# Final integrated Streamlit app code
final_script_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import math

# Load combined index return dataset
@st.cache_data
def load_combined_returns():
    df = pd.read_csv("sample_index_returns.csv")
    return df

# User input section
def get_user_inputs(index_names):
    st.sidebar.header("Simulation Inputs")
    index_choice = st.sidebar.selectbox("Index Reference Input", index_names)
    ptp_interval = st.sidebar.selectbox("Point to Point", [1, 2], index=0, help="Interval in years over which returns are calculated.")
    start_age = st.sidebar.number_input("Starting Age", min_value=40, max_value=80, value=55, step=1)
    premium = st.sidebar.number_input("Starting Balance", min_value=0.0, value=1000000.0, step=10000.0)
    pr_start = st.sidebar.number_input("Starting FIA PR (%)", min_value=0.0, max_value=400.0, value=100.0, step=1.0) / 100
    pr_end = st.sidebar.number_input("Ending FIA PR (%)", min_value=0.0, max_value=400.0, value=35.0, step=1.0) / 100
    cap_input = st.sidebar.number_input("Cap Rate (%) [0 = randomize]", min_value=0.0, max_value=50.0, value=0.0, step=0.1) / 100
    floor = st.sidebar.number_input("FIA Floor Rate (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1) / 100
    spread = st.sidebar.number_input("FIA Spread Rate (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1) / 100
    fee = st.sidebar.number_input("401(k) Annual Fee (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100
    inflation = st.sidebar.number_input("Annual Inflation Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100
    tax = st.sidebar.number_input("Tax Rate on RMDs (%)", min_value=0.0, max_value=50.0, value=30.0, step=1.0) / 100
    return index_choice, ptp_interval, start_age, premium, pr_start, pr_end, cap_input, floor, spread, fee, inflation, tax

# Growth compounding logic
def compound_growth(start_balance, returns):
    balances = [start_balance]
    for r in returns:
        balances.append(balances[-1] * (1 + r))
    return balances[:-1]

# RMD Calculation logic
def calculate_rmds(balances, start_age):
    rmd_table = {age: max(1.9, 27 - 0.5 * (age - 72)) for age in range(72, 105)}
    rmd_values = []
    net_values = []
    adj_values = []
    for i, age in enumerate(range(start_age, 105)):
        if age >= 72:
            rmd = balances[i] / rmd_table.get(age, 1.9)
            net = rmd * (1 - 0.30)
            adj = net * (1 - 0.03)
        else:
            rmd = net = adj = 0
        rmd_values.append(rmd)
        net_values.append(net)
        adj_values.append(adj)
    return balances, rmd_values, net_values, adj_values

# Simulation logic
def run_simulation(index_choice, ptp_interval, start_age, premium, pr_start, pr_end, cap_input, floor, spread, fee, inflation_rate, tax_rate, combined_df):
    st.subheader("Simulation Results")
    st.markdown(f"<h6 style='margin-top: -16px;'>Index Reference - {index_choice}</h6>", unsafe_allow_html=True)

    selected_data = combined_df[combined_df['Index'] == index_choice][['Year', 'Return']]
    selected_returns = selected_data['Return'].tolist()

    # Point-to-point calculation
    returns_ptp = []
    for i in range(0, len(selected_returns) - ptp_interval + 1):
        cumulative_return = np.prod([(1 + r) for r in selected_returns[i:i + ptp_interval]]) - 1
        annualized_return = (1 + cumulative_return) ** (1 / ptp_interval) - 1
        returns_ptp.append(annualized_return)

    # Age range and extended returns
    ages = list(range(start_age, 105))
    repeat_factor = math.ceil(len(ages) / len(returns_ptp))
    returns_extended = (returns_ptp * repeat_factor)[:len(ages)]

    np.random.seed(42)
    pr_decay = np.linspace(pr_start, pr_end, len(ages))
    caps = np.full(len(ages), cap_input) if cap_input > 0 else np.random.uniform(0.03, 0.15, size=len(ages))

    fia_returns = []
    for pr, r, cap in zip(pr_decay, returns_extended, caps):
        raw_return = pr * r
        capped_return = min(raw_return, cap)
        adjusted_return = max(floor, capped_return - spread)
        fia_returns.append(adjusted_return)

    k401_returns = [(1 + r) * (1 - fee) - 1 for r in returns_extended]
    fia_bal = compound_growth(premium, fia_returns)
    k401_bal = compound_growth(premium, k401_returns)

    fia_start, fia_rmd, fia_net, fia_adj = calculate_rmds(fia_bal, start_age)
    k401_start, k401_rmd, k401_net, k401_adj = calculate_rmds(k401_bal, start_age)

    df = pd.DataFrame({
        "Year": list(range(1, len(ages) + 1)),
        "Age": ages,
        "FIA Balance": fia_start,
        "FIA RMD": fia_rmd,
        "FIA Net": fia_net,
        "FIA After Inflation": fia_adj,
        "401k Balance": k401_start,
        "401k RMD": k401_rmd,
        "401k Net": k401_net,
        "401k After Inflation": k401_adj
    })

    st.dataframe(df)

    # CSV Export
    df_export = df.copy()
    for col in df_export.columns:
        if col in ["Year", "Age"]:
            continue
        if "Balance" in col or "RMD" in col or "Net" in col or "Inflation" in col:
            df_export[col] = df_export[col].apply(lambda x: f"${x:,.0f}")
        elif "Rate" in col:
            df_export[col] = df_export[col].apply(lambda x: f"{x * 100:.1f}%")
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results as CSV", data=csv, file_name="fia_vs_401k_simulation.csv", mime="text/csv")

# App runner
if __name__ == "__main__":
    combined_df = load_combined_returns()
    index_names = combined_df['Index'].unique().tolist()
    inputs = get_user_inputs(index_names)

    if st.button("Run Simulation", key="run_button"):
        run_simulation(*inputs, combined_df)
'''

# Write the final script
final_script_path.write_text(final_script_code)
final_script_path.name