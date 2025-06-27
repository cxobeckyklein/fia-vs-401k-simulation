import streamlit as st
import numpy as np
import pandas as pd
import math

# Load combined index return dataset
@st.cache_data
def load_combined_returns():
    df = pd.read_csv("index_returns_combined.csv")
    return df

def get_user_inputs(index_names):
    st.sidebar.header("Simulation Inputs")
    index_choice = st.sidebar.selectbox("Choose Index Dataset", index_names)
    premium = st.sidebar.number_input("Enter Starting Balance", min_value=0.0, value=1000000.0, step=10000.0)
    pr_start = st.sidebar.number_input("Starting FIA Participation Rate", 0.0, 1.0, 1.0, 0.01)
    pr_end = st.sidebar.number_input("Ending FIA Participation Rate", 0.0, 1.0, 0.35, 0.01)
    floor = st.sidebar.number_input("FIA Floor Rate", 0.0, 0.10, 0.0, 0.01)
    fee = st.sidebar.number_input("401(k) Annual Fee", 0.0, 0.05, 0.02, 0.005)
    inflation = st.sidebar.number_input("Annual Inflation Rate", 0.0, 0.1, 0.03, 0.005)
    tax = st.sidebar.number_input("Tax Rate on RMDs", 0.0, 0.5, 0.30, 0.01)
    return index_choice, premium, pr_start, pr_end, floor, fee, inflation, tax

def compound_growth(start, returns):
    balances = [start]
    for r in returns[:-1]:
        start *= (1 + r)
        balances.append(start)
    return balances

def calculate_rmds(balances, ages, tax_rate, inflation_rate):
    rmd_divisors = {
        age: div for age, div in zip(range(73, 95), [
            26.5, 25.5, 24.6, 23.7, 22.9, 22.0, 21.1, 20.2,
            19.4, 18.5, 17.7, 16.8, 16.0, 15.2, 14.4, 13.7,
            12.9, 12.2, 11.5, 10.8, 10.1, 9.5
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

def run_simulation():
    st.title("FIA vs 401(k) Comparison Tool")

    combined_df = load_combined_returns()
    index_names = combined_df['Index'].unique().tolist()

    index_choice, premium, pr_start, pr_end, floor, fee, inflation_rate, tax_rate = get_user_inputs(index_names)

    selected_data = combined_df[combined_df['Index'] == index_choice][['Year', 'Return']]
    st.write("### Annual Returns for Selected Index")
    st.dataframe(selected_data.style.format({"Return": "{:.2%}"}))

    selected_returns = selected_data['Return'].tolist()
    repeat_factor = math.ceil(40 / len(selected_returns))
    returns_40yr = (selected_returns * repeat_factor)[:40]

    pr_decay = np.linspace(pr_start, pr_end, 40)
    fia_returns = np.maximum(floor, pr_decay * np.array(returns_40yr))
    k401_returns = [(1 + r) * (1 - fee) - 1 for r in returns_40yr]

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

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "fia_vs_401k_results.csv", "text/csv")

if __name__ == "__main__":
    run_simulation()