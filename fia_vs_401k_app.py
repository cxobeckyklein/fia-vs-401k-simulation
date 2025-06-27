import streamlit as st
import numpy as np
import pandas as pd
import math

# Load combined index return dataset
@st.cache_data
def load_combined_returns():
    df = pd.read_csv("sample_index_returns.csv")
    return df

def formatted_percent_input(label, min_val, max_val, default_val, step=0.1):
    col1, col2 = st.sidebar.columns([4, 1])
    val = col1.number_input(label, min_value=min_val, max_value=max_val, value=default_val, step=step)
    col2.markdown("### %")
    return val / 100  # Convert to decimal

def get_user_inputs(index_names):
    st.sidebar.header("Simulation Inputs")
    index_choice = st.sidebar.selectbox("Index Benchmark", index_names)
    start_age = st.sidebar.number_input("Starting Age", min_value=40, max_value=85, value=55, step=1)
    premium = st.sidebar.number_input("Starting Balance", min_value=0, value=1000000, step=1000)
    pr_start = formatted_percent_input("Starting FIA PR", 0.0, 1000.0, 100.0)
    pr_end = formatted_percent_input("Ending FIA PR", 0.0, 100.0, 35.0)
    floor = formatted_percent_input("FIA Floor Rate", 0.0, 10.0, 0.0)
    fee = formatted_percent_input("401(k) Annual Fee Rate", 0.0, 10.0, 2.0)
    inflation = formatted_percent_input("Annual Inflation Rate", 0.0, 99.0, 3.0)
    tax = formatted_percent_input("Tax Rate on RMDs", 0.0, 100.0, 20.0) 
    return index_choice, start_age, premium, pr_start, pr_end, floor, fee, inflation, tax

def compound_growth(start, returns):
    balances = [start]
    for r in returns[:-1]:
        start *= (1 + r)
        balances.append(start)
    return balances

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

def run_simulation(index_choice, start_age, premium, pr_start, pr_end, floor, fee, inflation_rate, tax_rate, combined_df):
    if st.button("Run Simulation"):
        st.subheader(f"Simulation Results - {index_choice}")
        # rest of your simulation logic

    # Get returns for selected index
    selected_data = combined_df[combined_df['Index'] == index_choice][['Year', 'Return']]
    selected_returns = selected_data['Return'].tolist()

    # Extend returns to cover all ages (start_age to 104)
    ages = list(range(start_age, 105))
    years = list(range(1, len(ages) + 1))
    repeat_factor = math.ceil(len(ages) / len(selected_returns))
    returns_extended = (selected_returns * repeat_factor)[:len(ages)]

    # Calculate FIA and 401(k) returns
    pr_decay = np.linspace(pr_start, pr_end, len(ages))
    fia_returns = np.maximum(floor, pr_decay * np.array(returns_extended))
    k401_returns = [(1 + r) * (1 - fee) - 1 for r in returns_extended]

    # Grow balances
    fia_bal = compound_growth(premium, fia_returns)
    k401_bal = compound_growth(premium, k401_returns)

    # Calculate RMDs
    fia_start, fia_rmd, fia_net, fia_adj = calculate_rmds(fia_bal, ages, tax_rate, inflation_rate)
    k401_start, k401_rmd, k401_net, k401_adj = calculate_rmds(k401_bal, ages, tax_rate, inflation_rate)

    # Build results DataFrame
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
        "401k Infl-Adj RMD": k401_adj
    })

    # Display results with formatting
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

    # Format export with % and $ values
    export_df = df.copy()
    export_df["FIA PR Start"] = f"{pr_start:.2f}%"
    export_df["FIA PR End"] = f"{pr_end:.2f}%"
    export_df["FIA Floor"] = f"{floor:.2f}%"
    export_df["401(k) Fee"] = f"{fee:.2f}%"
    export_df["Inflation"] = f"{inflation_rate:.2f}%"
    export_df["Tax Rate"] = f"{tax_rate:.2f}%"

    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results as CSV", csv, "fia_vs_401k_results.csv", "text/csv")

# ===== Streamlit App Entry Point =====
st.title("FIA vs 401(k) Comparison Tool")

combined_df = load_combined_returns()
index_names = combined_df['Index'].unique().tolist()

inputs = get_user_inputs(index_names)

if st.button("Run Simulation"):
    run_simulation(*inputs, combined_df)
