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
    selected_indexes = st.sidebar.multiselect("Select Indexes", options=index_names, default=index_names[:2])

    ptp_intervals = {}
    for idx in selected_indexes:
        ptp_intervals[idx] = st.sidebar.selectbox(f"PtP Interval for {idx}", [1, 2], key=f"ptp_{idx}")

    start_age = st.sidebar.number_input("Starting Age", min_value=40, max_value=80, value=55, step=1)
    premium = st.sidebar.number_input("Starting Balance", min_value=0.0, value=1000000.0, step=10000.0)
    pr_start = st.sidebar.number_input("Starting FIA PR (%)", min_value=0.0, max_value=400.0, value=100.0, step=1.0) / 100
    cap_input = st.sidebar.number_input("Cap Rate (%) [0 = randomize]", min_value=0.0, max_value=50.0, value=0.0, step=0.1) / 100
    floor = st.sidebar.number_input("FIA Floor Rate (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1) / 100
    spread = st.sidebar.number_input("FIA Spread Rate (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1) / 100
    fee = st.sidebar.number_input("401(k) Annual Fee (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100
    inflation = st.sidebar.number_input("Annual Inflation Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100
    tax = st.sidebar.number_input("Tax Rate on RMDs (%)", min_value=0.0, max_value=50.0, value=30.0, step=1.0) / 100
    return selected_indexes, ptp_intervals, start_age, premium, pr_start, cap_input, floor, spread, fee, inflation, tax

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
def run_simulation(selected_indexes, ptp_intervals, start_age, premium, pr_start, cap_input, floor, spread, fee, inflation_rate, tax_rate, combined_df):
    st.header("Simulation Results")
    st.markdown(f"<div style='font-size: 20px; margin-top: -10px; color: grey;'>Indexes: {', '.join(selected_indexes)}</div>", unsafe_allow_html=True)

    ages = list(range(start_age, 105))
    years = list(range(1, len(ages) + 1))

    # Collect weighted and ptp-adjusted returns for each index
    weighted_returns = []
    for idx in selected_indexes:
        ptp = ptp_intervals[idx]
        index_data = combined_df[combined_df['Index'] == idx][['Year', 'Return']]
        returns = index_data['Return'].tolist()

        ptp_returns = []
        for i in range(0, len(returns) - ptp + 1):
            cumulative = np.prod([(1 + r) for r in returns[i:i + ptp]]) - 1
            annualized = (1 + cumulative) ** (1 / ptp) - 1
            ptp_returns.append(annualized)

        repeat_factor = math.ceil(len(ages) / len(ptp_returns))
        extended = (ptp_returns * repeat_factor)[:len(ages)]
        weighted_returns.append(extended)

    returns_extended = np.mean(weighted_returns, axis=0)

    # PR decay: 20% up/down year over year capped at 300% max
    np.random.seed(42)
    pr_decay = [pr_start]
    for _ in range(1, len(ages)):
        last = pr_decay[-1]
        low = max(0, last * 0.8)
        high = min(last * 1.2, 3.0)
        pr_decay.append(np.random.uniform(low, high))

    # Caps: user-defined or randomized
    if cap_input > 0:
        caps = [cap_input] * len(ages)
    else:
        caps = np.random.uniform(0.03, 0.15, size=len(ages))

    # FIA and 401k returns
    fia_returns = []
    for pr, r, cap in zip(pr_decay, returns_extended, caps):
        raw_return = pr * r
        capped_return = min(raw_return, cap)
        adjusted_return = max(floor, capped_return - spread)
        fia_returns.append(adjusted_return)

    k401_returns = [max(-1, r - fee) for r in returns_extended]

    fia_bal = compound_growth(premium, fia_returns)
    k401_bal = compound_growth(premium, k401_returns)

    fia_start, fia_rmd, fia_net, fia_adj = calculate_rmds(fia_bal, ages, tax_rate, inflation_rate)
    k401_start, k401_rmd, k401_net, k401_adj = calculate_rmds(k401_bal, ages, tax_rate, inflation_rate)

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

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"fia_vs_401k_comparison.csv", "text/csv")

# App runner
if __name__ == "__main__":
    combined_df = load_combined_returns()
    index_names = combined_df['Index'].unique().tolist()
    inputs = get_user_inputs(index_names)

    if st.button("Run Simulation", key="run_button"):
        run_simulation(*inputs, combined_df)
