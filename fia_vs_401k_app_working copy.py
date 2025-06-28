import streamlit as st
import numpy as np
import pandas as pd
import math

# Load index return dataset
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

# Growth compounding
def compound_growth(start, returns):
    balances = [start]
    for r in returns[:-1]:
        start *= (1 + r)
        balances.append(start)
    return balances

# RMDs calculation
def calculate_rmds(balances, ages, tax_rate, inflation_rate):
    rmd_divisors = {
        age: div for age, div in zip(range(73, 105), [
            26.5, 25.5, 24.6, 23.7, 22.9, 22.0, 21.1, 20.2,
            19.4, 18.5, 17.7, 16.8, 16.0, 15.2, 14.4, 13.7,
            12.9, 12.2, 11.5, 10.8, 10.1, 9.5, 8.9, 8.4,
            7.8, 7.3, 6.8, 6.4, 6.0, 5.6, 5.2, 4.9
        ])
    }
    start_bal, rmd, net_rmd, infl_adj_rmd = [], [], [], []
    infl_factor = 1.0
    for i, age in enumerate(ages):
        bal = balances[i]
        dist = bal / rmd_divisors[age] if age in rmd_divisors else 0
        net = dist * (1 - tax_rate)
        start_bal.append(bal)
        rmd.append(dist)
        net_rmd.append(net)
        infl_adj_rmd.append(net / infl_factor)
        infl_factor *= (1 + inflation_rate)
    return start_bal, rmd, net_rmd, infl_adj_rmd

# Simulation logic
def run_simulation(index_choice, ptp_interval, start_age, premium, pr_start, pr_end, cap_input, floor, spread, fee, inflation_rate, tax_rate, combined_df):
    st.header("Simulation Results")
    st.markdown(f"<div style='font-size: 20px; margin-top: -10px; color: grey;'>Index Reference {index_choice} | {ptp_interval}-Year Point to Point</div>", unsafe_allow_html=True)
  
    selected_data = combined_df[combined_df['Index'] == index_choice][['Year', 'Return']]
    selected_returns = selected_data['Return'].tolist()

    ages = list(range(start_age, 105))
    years = list(range(1, len(ages) + 1))

       # Apply point-to-point interval logic
    returns_ptp = []
    for i in range(0, len(selected_returns) - ptp_interval + 1):
        cumulative_return = np.prod([(1 + r) for r in selected_returns[i:i + ptp_interval]]) - 1
        annualized_return = (1 + cumulative_return) ** (1 / ptp_interval) - 1
        returns_ptp.append(annualized_return)

    repeat_factor = math.ceil(len(ages) / len(returns_ptp))
    returns_extended = (returns_ptp * repeat_factor)[:len(ages)]   
    
    # Participation rates
    pr_decay = np.linspace(pr_start, pr_end, len(ages))

    # Caps: user-defined or randomized
    if cap_input > 0:
        cap_val = [cap_input] * len(ages)
    else:
        np.random.seed(42)
        caps = np.random.uniform(0.03, 0.15, size=len(ages))

    # FIA and 401k returns
    fia_returns = []
    for pr, r, cap_val in zip(pr_decay, returns_extended, caps):
        raw_return = pr * r
        capped_return = min(raw_return, cap_val)
        adjusted_return = max(floor, capped_return - spread)
        fia_returns.append(adjusted_return)
    
    k401_returns = [max(-1, r - fee) for r in returns_extended]  # Improved net return calc with floor

    fia_bal = compound_growth(premium, fia_returns)
    k401_bal = compound_growth(premium, k401_returns)

    fia_start, fia_rmd, fia_net, fia_adj = calculate_rmds(fia_bal, ages, tax_rate, inflation_rate)
    k401_start, k401_rmd, k401_net, k401_adj = calculate_rmds(k401_bal, ages, tax_rate, inflation_rate)

    st.dataframe(df.reset_index(drop=True))

    df = pd.DataFrame({
        "Year": years,
        "Age": ages,
        "FIA Balance": fia_start,
        "FIA RMD": fia_rmd,
        "FIA After-Tax RMD": fia_net,
        "FIA Infl-Adj RMD": fia_adj,
        "401k Balance": k401_start,
        "401k RMD": k401_rmd,
        "401k After-Tax RMD": k401_net,
        "401k Infl-Adj RMD": k401_adj,
    })

    st.dataframe(df.style.format({
        "FIA Balance": "${:,.0f}",
        "FIA RMD": "${:,.0f}",
        "FIA After-Tax RMD": "${:,.0f}",
        "FIA Infl-Adj RMD": "${:,.0f}",
        "401k Balance": "${:,.0f}",
        "401k RMD": "${:,.0f}",
        "401k After-Tax RMD": "${:,.0f}",
        "401k Infl-Adj RMD": "${:,.0f}"
    }))


    df_export = df.copy()
    for col in df_export.columns:
        if "Balance" in col or "RMD" in col:
            df_export[col] = df_export[col].apply(lambda x: f"${x:,.0f}")

    # Remove index column
    df.to_csv("fia_401k_comparison.csv", index=False)
    
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"fia_vs_401k_{index_choice.replace(' ', '_')}.csv", "text/csv")

# App runner
if __name__ == "__main__":
    combined_df = load_combined_returns()
    index_names = combined_df['Index'].unique().tolist()
    inputs = get_user_inputs(index_names)

    if st.button("Run Simulation", key="run_button"):
        run_simulation(*inputs, combined_df)