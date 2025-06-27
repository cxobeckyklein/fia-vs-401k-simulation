import streamlit as st
import numpy as np
import pandas as pd

# Define index return datasets
index_data = {
    "S&P 500 (2003–2022)": [
        0.2638, 0.0899, 0.0300, 0.1362, 0.0310, -0.3849, 0.2345, 0.1284, 0.0000, 0.1351,
        0.2960, 0.1139, -0.0007, 0.0954, 0.1922, -0.0624, 0.2888, 0.1633, 0.2651, -0.1954
    ],
    "Conservative Blend": [
        0.05, 0.04, 0.06, 0.03, 0.04, 0.05, 0.03, 0.04, 0.06, 0.03,
        0.04, 0.05, 0.03, 0.04, 0.06, 0.03, 0.04, 0.05, 0.03, 0.04
    ],
    "Aggressive (hypothetical)": [
        0.30, 0.25, 0.15, 0.22, 0.18, -0.25, 0.27, 0.30, -0.10, 0.28,
        0.34, 0.20, 0.12, 0.18, 0.25, -0.12, 0.31, 0.21, 0.26, -0.18
    ]
}

def get_user_inputs():
    index_choice = st.selectbox("Choose Index Dataset", list(index_data.keys()))
    premium = st.number_input("Enter Starting Balance", min_value=0.0, value=1000000.0, step=10000.0)
    pr_start = st.number_input("Starting FIA Participation Rate", 0.0, 1.0, 1.0, 0.01)
    pr_end = st.number_input("Ending FIA Participation Rate", 0.0, 1.0, 0.35, 0.01)
    floor = st.number_input("FIA Floor Rate", 0.0, 0.10, 0.0, 0.01)
    fee = st.number_input("401(k) Annual Fee", 0.0, 0.05, 0.02, 0.005)
    inflation = st.number_input("Annual Inflation Rate", 0.0, 0.1, 0.03, 0.005)
    tax = st.number_input("Tax Rate on RMDs", 0.0, 0.5, 0.30, 0.01)
    return index_choice, premium, pr_start, pr_end, floor, fee, inflation, tax

def compound_growth(start, returns):
    balances = [start]
    for r in returns[:-1]:
        start *= (1 + r)
        balances.append(start)
    return balances

def calculate_rmds(balances, ages, tax_rate, inflation_rate):
    rmd_divisors = dict(zip(range(73, 95), [
        26.5, 25.5, 24.6, 23.7, 22.9, 22.0, 21.1, 20.2,
        19.4, 18.5, 17.7, 16.8, 16.0, 15.2, 14.4, 13.7,
        12.9, 12.2, 11.5, 10.8, 10.1, 9.5
    ]))
    start_bal, rmd, net_rmd, infl_adj_rmd = [], [], [], []
    infl_factor = 1.0
    for i, age in enumerate(range(55, 95)):
        bal = balances[i]
        dist = bal / rmd_divisors.get(age, 1e10)
        net = dist * (1 - tax_rate)
        start_bal.append(bal)
        rmd.append(dist)
        net_rmd.append(net)
        infl_adj_rmd.append(net / infl_factor)
        infl_factor *= (1 + inflation_rate)
    return start_bal, rmd, net_rmd, infl_adj_rmd

def run_simulation():
    st.title("FIA vs 401(k) Comparison Tool")

    index_choice, premium, pr_start, pr_end, floor, fee, inflation_rate, tax_rate = get_user_inputs()

    # Extend index returns for 40 years
    raw_returns = index_data[index_choice] * 2
    pr_decay = np.linspace(pr_start, pr_end, 40)

    fia_returns = np.maximum(floor, pr_decay * np.array(raw_returns))
    k401_returns = [(1 + r) * (1 - fee) - 1 for r in raw_returns]

    fia_bal = compound_growth(premium, fia_returns)
    k401_bal = compound_growth(premium, k401_returns)

    ages = list(range(55, 95))
    years = list(range(1, 41))

    fia_start, fia_rmd, fia_net, fia_adj = calculate_rmds(fia_bal, ages, tax_rate, inflation_rate)
    k401_start, k401_rmd, k401_net, k401_adj = calculate_rmds(k401_bal, ages, tax_rate, inflation_rate)

    df = pd.DataFrame({
        "Year": years,
        "Age": ages,
        "FIA Start Balance": fia_start,
        "FIA RMD": fia_rmd,
        "FIA After-Tax RMD": fia_net,
        "FIA Infl-Adj RMD": fia_adj,
        "401k Start Balance": k401_start,
        "401k RMD": k401_rmd,
        "401k After-Tax RMD": k401_net,
        "401k Infl-Adj RMD": k401_adj,
    })

    st.dataframe(df.style.format({
        "FIA Start Balance": "${:,.0f}",
        "FIA RMD": "${:,.0f}",
        "FIA After-Tax RMD": "${:,.0f}",
        "FIA Infl-Adj RMD": "${:,.0f}",
        "401k Start Balance": "${:,.0f}",
        "401k RMD": "${:,.0f}",
        "401k After-Tax RMD": "${:,.0f}",
        "401k Infl-Adj RMD": "${:,.0f}"
    }))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "fia_vs_401k_results.csv", "text/csv")

if __name__ == "__main__":
    run_simulation()
