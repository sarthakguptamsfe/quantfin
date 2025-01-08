import streamlit as st
import numpy as np
import pandas as pd
import requests
import math
import plotly.express as px
import plotly.graph_objs as go
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
import scipy.stats as stats
from scipy.optimize import newton
from py_vollib.black_scholes.greeks import analytical

###############################################################################
#                              GLOBAL CONFIG
###############################################################################
st.set_page_config(
    page_title="Unified Portfolio Visualizer (Plotly + Backtesting)",
    layout="wide",
)

# Replace with your own API key for FinancialModelingPrep
API_KEY = '0uTB4phKEr4dHcB2zJMmVmKUcywpkxDQ'

###############################################################################
#                           DATA FETCH FUNCTION
###############################################################################
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from FinancialModelingPrep.
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}: Unable to fetch data for {ticker}."
        
        data = response.json()
        if 'historical' not in data or not data['historical']:
            return None, f"No historical data found for {ticker} in the specified date range."

        prices = pd.DataFrame(data['historical'])
        prices['date'] = pd.to_datetime(prices['date'])
        prices.set_index('date', inplace=True)
        return prices['close'].sort_index(), None

    except Exception as e:
        return None, f"Error fetching data for {ticker}: {str(e)}"

###############################################################################
#                          PAGE 1: HOME
###############################################################################
def page_home():
    st.title("Welcome to the Unified Portfolio Visualizer (Plotly + Backtesting)")
    st.markdown(
        """
        **This application demonstrates**:
        - **Backtesting** a multi-stock portfolio over a chosen date range
        - **Value at Risk (VaR)** for a multi-equity portfolio
        - **Fixed Income** (bond pricing, yield, duration, convexity)
        - **Options Pricing** via the **Black–Scholes** model
        - A **Smart Visualization** page that unifies data

        Use the **sidebar** to navigate between modules.
        """
    )
    # Optional banner image
    st.image("https://i.postimg.cc/jd3b7X91/Screenshot-2024-05-10-at-12-33-02-AM.png", use_column_width=True)


###############################################################################
#                          PAGE 2: PORTFOLIO BACKTESTING
###############################################################################
# Fetch benchmark data
def fetch_benchmark_data(benchmark, start_date, end_date):
    """
    Fetch historical benchmark data for comparison.
    """
    benchmarks = {"S&P 500": "SPY", "NASDAQ 100": "QQQ", "Russell 2000": "IWM"}
    ticker = benchmarks.get(benchmark)
    if not ticker:
        return None, f"Benchmark '{benchmark}' not supported."
    return fetch_stock_data(ticker, start_date, end_date)  


# Backtest portfolio
def backtest_portfolio(stocks, weights, initial_capital, start_date, end_date):
    """
    Perform backtest of portfolio with given stocks and weights.
    """
    portfolio = pd.DataFrame()
    total_weight = sum(weights)

    # Fetch and align stock prices
    for i, (ticker, weight) in enumerate(zip(stocks, weights)):
        print(f"Fetching data for {ticker}...")  # Debugging
        prices, err = fetch_stock_data(ticker, start_date, end_date)
        if err:
            print(f"Error for {ticker}: {err}")  # Debugging
            return None, err
        portfolio[ticker] = prices

    if portfolio.empty:
        return None, "No valid data found for any stock."

    portfolio.dropna(inplace=True)  # Remove rows with missing data

    # Calculate portfolio value
    allocation = [initial_capital * (w / total_weight) for w in weights]
    shares = [allocation[i] / portfolio[stocks[i]].iloc[0] for i in range(len(stocks))]
    portfolio['PortfolioValue'] = sum([shares[i] * portfolio[stocks[i]] for i in range(len(stocks))])

    # Calculate returns and drawdowns
    portfolio['DailyReturn'] = portfolio['PortfolioValue'].pct_change().fillna(0)
    portfolio['CumulativeReturn'] = (1 + portfolio['DailyReturn']).cumprod() - 1
    rolling_max = portfolio['PortfolioValue'].cummax()
    portfolio['Drawdown'] = (portfolio['PortfolioValue'] - rolling_max) / rolling_max

    return portfolio, None

# Calculate performance metrics
def calculate_metrics(df, initial_capital):
    """
    Calculate performance metrics for the portfolio or benchmark.
    """
    value_column = 'PortfolioValue' if 'PortfolioValue' in df.columns else 'BenchmarkValue'
    start_balance = df[value_column].iloc[0]
    end_balance = df[value_column].iloc[-1]
    total_return = end_balance / start_balance - 1
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1
    daily_volatility = df['DailyReturn'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

    # Ensure Drawdown exists
    max_drawdown = df['Drawdown'].min() if 'Drawdown' in df.columns else None

    return {
        'Start Balance': start_balance,
        'End Balance': end_balance,
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown
    }


###############################################################################
#                          PERFORMANCE SUMMARY EXTENSIONS
###############################################################################
def calculate_additional_metrics(df):
    """Calculate additional metrics for performance summary."""
    best_year = df.resample('Y')['DailyReturn'].sum().max()
    worst_year = df.resample('Y')['DailyReturn'].sum().min()
    sortino_ratio = df['DailyReturn'].mean() / df[df['DailyReturn'] < 0]['DailyReturn'].std()
    benchmark_correlation = df['DailyReturn'].corr(df['BenchmarkReturn'])

    return {
        'Best Year': best_year,
        'Worst Year': worst_year,
        'Sortino Ratio': sortino_ratio,
        'Benchmark Correlation': benchmark_correlation
         }

###############################################################################
#                           ENHANCED MAIN FUNCTION
###############################################################################
def main():
    st.title("Enhanced Portfolio Backtesting Results")

    # User Inputs
    initial_capital = st.number_input("Initial Capital (USD):", min_value=0.0, value=100000.0, step=1000.0)
    num_stocks = st.slider("Number of Stocks (1–10):", 1, 10, 1)
    stocks, weights = [], []

    for i in range(num_stocks):
        cols = st.columns(2)
        with cols[0]:
            ticker = st.text_input(f"Stock Ticker {i + 1}:", key=f"ticker_{i}")
        with cols[1]:
            weight = st.number_input(f"Weight for {ticker or 'Stock'} (%):", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"weight_{i}")
        stocks.append(ticker)
        weights.append(weight)

    total_weight = sum(weights)
    st.write(f"**Total Weight:** {total_weight:.2f}%")

    benchmark = st.selectbox("Select Benchmark:", ["S&P 500", "NASDAQ 100", "Russell 2000"])
    colA, colB = st.columns(2)
    with colA:
        start_date = st.date_input("Start Date:", value=pd.to_datetime("2000-01-01"))
    with colB:
        end_date = st.date_input("End Date:", value=pd.to_datetime("2025-01-01"))

    if st.button("Run Backtest"):
        if abs(total_weight - 100) > 1e-9:
            st.error("Total weights must sum to 100%.")
            return

        # Perform Portfolio Backtest
        portfolio, err = backtest_portfolio(stocks, weights, initial_capital, start_date, end_date)
        if err:
            st.error(err)
            return

        benchmark_prices, err = fetch_benchmark_data(benchmark, start_date, end_date)
        if err:
            st.error(err)
            return

        benchmark_prices = benchmark_prices.loc[portfolio.index]
        benchmark_df = pd.DataFrame({
            "BenchmarkValue": initial_capital * (benchmark_prices / benchmark_prices.iloc[0])
        })
        benchmark_df['DailyReturn'] = benchmark_df['BenchmarkValue'].pct_change().fillna(0)
        benchmark_df['CumulativeReturn'] = (1 + benchmark_df['DailyReturn']).cumprod() - 1
        rolling_max_benchmark = benchmark_df['BenchmarkValue'].cummax()
        benchmark_df['Drawdown'] = (benchmark_df['BenchmarkValue'] - rolling_max_benchmark) / rolling_max_benchmark

        portfolio['BenchmarkReturn'] = benchmark_df['DailyReturn']

        # Save results in session state
        st.session_state['portfolio'] = portfolio
        st.session_state['benchmark_df'] = benchmark_df

    # Retrieve data from session state
    if 'portfolio' in st.session_state and 'benchmark_df' in st.session_state:
        portfolio = st.session_state['portfolio']
        benchmark_df = st.session_state['benchmark_df']

        # Calculate Metrics
        portfolio_metrics = calculate_metrics(portfolio, initial_capital)
        benchmark_metrics = calculate_metrics(benchmark_df, initial_capital)
        additional_metrics = calculate_additional_metrics(portfolio)

        # Display Allocation Pie Chart
        st.subheader("Portfolio Allocation")
        allocation_df = pd.DataFrame({
            "Ticker": stocks,
            "Weight (%)": weights
        })
        st.table(allocation_df)
        fig_pie = px.pie(allocation_df, values="Weight (%)", names="Ticker", title="Portfolio Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Display Performance Summary
        st.subheader("Performance Summary")
        summary_df = pd.DataFrame({
            "Metric": [
                "Start Balance", "End Balance", "Total Return", "Annualized Return (CAGR)", "Annualized Volatility",
                "Sharpe Ratio", "Maximum Drawdown", "Best Year", "Worst Year", "Sortino Ratio", "Benchmark Correlation"
            ],
            "Portfolio": [
                f"${portfolio_metrics['Start Balance']:,.2f}",
                f"${portfolio_metrics['End Balance']:,.2f}",
                f"{portfolio_metrics['Total Return']:.2%}",
                f"{portfolio_metrics['Annualized Return']:.2%}",
                f"{portfolio_metrics['Annualized Volatility']:.2%}",
                f"{portfolio_metrics['Sharpe Ratio']:.2f}",
                f"{portfolio_metrics['Maximum Drawdown']:.2%}",
                f"{additional_metrics['Best Year']:.2%}",
                f"{additional_metrics['Worst Year']:.2%}",
                f"{additional_metrics['Sortino Ratio']:.2f}",
                f"{additional_metrics['Benchmark Correlation']:.2f}"
            ],
            benchmark: [
                f"${benchmark_metrics['Start Balance']:,.2f}",
                f"${benchmark_metrics['End Balance']:,.2f}",
                f"{benchmark_metrics['Total Return']:.2%}",
                f"{benchmark_metrics['Annualized Return']:.2%}",
                f"{benchmark_metrics['Annualized Volatility']:.2%}",
                f"{benchmark_metrics['Sharpe Ratio']:.2f}",
                f"{benchmark_metrics['Maximum Drawdown']:.2%}",
                "N/A", "N/A", "N/A", "N/A"
            ]
        })
        st.table(summary_df)

        # Charts
        st.subheader("Portfolio vs. Benchmark Growth")
        growth_df = pd.concat([
            portfolio[['PortfolioValue']].rename(columns={'PortfolioValue': 'Portfolio'}),
            benchmark_df[['BenchmarkValue']].rename(columns={'BenchmarkValue': benchmark})
        ], axis=1)

        col1, col2 = st.columns(2)
        with col1:
            log_scale = st.checkbox("Logarithmic Scale", key="log_scale")
        with col2:
            inflation_adjust = st.checkbox("Inflation Adjusted", key="inflation_adjust")

        if inflation_adjust:
            inflation_factor = 1.02  # Placeholder for actual inflation adjustment logic
            growth_df = growth_df / inflation_factor

        fig = px.line(growth_df, title="Portfolio vs. Benchmark Growth", log_y=log_scale)
        st.plotly_chart(fig, use_container_width=True)

        # Annual Returns Chart
        st.subheader("Annual Returns")
        annual_returns = portfolio['DailyReturn'].resample('Y').sum() * 100
        benchmark_annual_returns = benchmark_df['DailyReturn'].resample('Y').sum() * 100
        annual_returns_df = pd.DataFrame({
            "Year": annual_returns.index.year,
            "Portfolio": annual_returns.values,
            benchmark: benchmark_annual_returns.values
        })
        fig_bar = px.bar(annual_returns_df, x="Year", y=["Portfolio", benchmark],
                         barmode="group", title="Annual Returns")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Trailing Returns Table
        st.subheader("Trailing Returns")
        trailing_periods = {
            "3 Month": 63, "Year To Date": portfolio.index[-1].timetuple().tm_yday,
            "1 Year": 252, "3 Year": 756, "5 Year": 1260, "10 Year": 2520, "Full": len(portfolio)
        }
        trailing_returns = {}
        for period_name, days in trailing_periods.items():
            if days <= len(portfolio):
                portfolio_annualized_return = (1 + portfolio['DailyReturn'].iloc[-days:].mean()) ** 252 - 1
                portfolio_annualized_volatility = portfolio['DailyReturn'].iloc[-days:].std() * np.sqrt(252)

                benchmark_annualized_return = (1 + benchmark_df['DailyReturn'].iloc[-days:].mean()) ** 252 - 1
                benchmark_annualized_volatility = benchmark_df['DailyReturn'].iloc[-days:].std() * np.sqrt(252)

                trailing_returns[period_name] = [
                    portfolio['DailyReturn'].iloc[-days:].sum() * 100,
                    benchmark_df['DailyReturn'].iloc[-days:].sum() * 100,
                    portfolio_annualized_return * 100,
                    portfolio_annualized_volatility * 100,
                    benchmark_annualized_return * 100,
                    benchmark_annualized_volatility * 100
                ]

        trailing_df = pd.DataFrame(trailing_returns, index=[
            "Total Return (Portfolio)", "Total Return (Benchmark)",
            "Annualized Return (Portfolio)", "Annualized Std Dev (Portfolio)",
            "Annualized Return (Benchmark)", "Annualized Std Dev (Benchmark)"
        ]).T
        st.table(trailing_df)

if __name__ == "__main__":
    main()

###############################################################################
#                          PAGE 3: MULTI-EQUITY VaR
###############################################################################
def page_var():
    st.title("Multiple-Equity Portfolio Value at Risk (VaR) Calculator (Plotly)")

    # Input for initial portfolio value
    portfolio_value = st.number_input(
        "Enter Initial Portfolio Value (USD):",
        min_value=0.0, value=100000.0, step=1000.0, format="%.2f"
    )

    # Number of stocks, stock symbols, weights
    n = st.slider("Select number of stocks (1 to 10):", 1, 10, 1)
    stock_names = []
    weights = []
    for i in range(n):
        cols = st.columns(2)
        with cols[0]:
            symbol = st.text_input(f"Stock Symbol {i+1} (e.g., AAPL):", key=f"stock_{i}")
        with cols[1]:
            w = st.number_input(f"Weight for Stock {i+1} (%):", 0.0, 100.0, step=5.0, key=f"weight_{i}")
        stock_names.append(symbol)
        weights.append(w)

    total_weight = sum(weights)
    st.write(f"**Total Weight:** {total_weight:.2f}%")

    # Date range
    colA, colB = st.columns(2)
    with colA:
        start_date = st.date_input("Start Date (for VaR):", date(2022, 1, 1))
    with colB:
        end_date = st.date_input("End Date (for VaR):", date.today())

    # Session state for portfolio returns
    if 'portfolio_returns' not in st.session_state:
        st.session_state['portfolio_returns'] = np.array([])

    # Fetch data & compute returns
    if st.button("Fetch Data and Calculate Statistics"):
        if total_weight != 100:
            st.error("Total weight must be exactly 100%. Adjust the weights.")
        else:
            portfolio_returns = None
            daily_return_matrix = []
            common_length = None

            for i in range(n):
                stock_data, msg = fetch_stock_data(stock_names[i], start_date.isoformat(), end_date.isoformat(), API_KEY)
                if stock_data:
                    # Compute daily returns
                    returns_array = np.diff(stock_data) / stock_data[:-1]
                    if common_length is None:
                        common_length = len(returns_array)
                    else:
                        common_length = min(common_length, len(returns_array))
                    daily_return_matrix.append(returns_array)
                else:
                    st.error(msg)
                    break

            if daily_return_matrix and common_length:
                # Align data to common length
                daily_return_matrix = [arr[-common_length:] for arr in daily_return_matrix]
                weighted_portfolio_returns = np.zeros(common_length)
                for i, arr in enumerate(daily_return_matrix):
                    weighted_portfolio_returns += (weights[i] / 100.0) * arr

                st.session_state['portfolio_returns'] = weighted_portfolio_returns
                st.success("Data fetched and statistics calculated successfully.")
            else:
                st.error("Failed to fetch data for one or more stocks.")

    # Plot returns with Plotly
    def plot_portfolio_returns(returns_array):
        df_plot = pd.DataFrame({
            'Day': np.arange(len(returns_array)),
            'Daily Returns': returns_array
        })
        fig = px.line(
            df_plot, x='Day', y='Daily Returns',
            title="Portfolio Returns Over Time",
            labels={'Day': 'Day', 'Daily Returns': 'Returns'}
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    # VaR calculation
    def calculate_var(returns, confidence_level, method, portfolio_val):
        if method == "Historical":
            # For a left-tail measure, percentile is (100 - c_level).
            var_sample = np.percentile(returns, 100 - confidence_level)
            var = -var_sample
        elif method == "Variance-Covariance":
            mu = np.mean(returns)
            sigma = np.std(returns)
            z = -(stats.norm.ppf(1 - confidence_level / 100))
            var = -(mu + z * sigma)
        elif method == "Monte Carlo":
            sims = 10000
            mu = np.mean(returns)
            sigma = np.std(returns)
            sim_returns = np.random.normal(mu, sigma, sims)
            var = -np.percentile(sim_returns, 100 - confidence_level)
        return var * portfolio_val

    # VaR method & confidence
    var_method = st.selectbox("VaR Method:", ["Historical", "Variance-Covariance", "Monte Carlo"])
    confidence = st.slider("Confidence Level:", 0.0, 99.0, 97.5, 0.5, format="%.1f")

    # Button to calculate & display VaR
    if st.button("Calculate VaR"):
        if 'portfolio_returns' not in st.session_state or st.session_state['portfolio_returns'].size == 0:
            st.error("No portfolio data to calculate VaR. Fetch data first.")
        else:
            # Plot
            plot_portfolio_returns(st.session_state['portfolio_returns'])
            # Compute VaR
            var_value = calculate_var(st.session_state['portfolio_returns'], confidence, var_method, portfolio_value)
            st.write(
                f"**Value at Risk (VaR)** at {confidence}% confidence, "
                f"using **{var_method}** method: **${var_value:,.2f}**"
            )


###############################################################################
#                          PAGE 4: FIXED INCOME
###############################################################################
def page_fixed_income():
    st.title("Fixed Income: Bond Price & Yield Calculator (Plotly)")

    # ----- Functions for Bond Calculations -----
    def calculate_ytm(price, par, coupon_rate, n_periods, freq):
        coupon = coupon_rate / 100 * par / freq
        guess = 0.05
        def bond_price(ytm):
            return sum(
                [coupon / (1 + ytm / freq) ** t for t in range(1, n_periods + 1)]
            ) + par / (1 + ytm / freq) ** n_periods
        def ytm_func(ytm):
            return price - bond_price(ytm)
        return newton(ytm_func, guess) * 100 * freq

    def calculate_ytc(price, par, coupon_rate, call_price, call_date, settlement_date, freq):
        coupon = coupon_rate / 100 * par / freq
        n_periods_call = (call_date - settlement_date).days // (365 // freq)
        guess = 0.05
        def bond_price(ytc):
            return sum([coupon / (1 + ytc / freq) ** t for t in range(1, n_periods_call + 1)]) \
                   + call_price / (1 + ytc / freq) ** n_periods_call
        def ytc_func(ytc):
            return price - bond_price(ytc)
        return newton(ytc_func, guess) * 100 * freq

    def calculate_price(par, coupon_rate, ytm, n_periods, freq):
        coupon = coupon_rate / 100 * par / freq
        cash_flows = [coupon] * n_periods
        cash_flows[-1] += par  # add par to last CF
        discounts = [(1 + ytm / (100*freq)) ** (-i) for i in range(1, n_periods + 1)]
        return sum(cf * df for cf, df in zip(cash_flows, discounts))

    def calculate_macaulay_duration(par, coupon_rate, ytm, n_periods, freq):
        coupon = coupon_rate / 100 * par / freq
        cash_flows = [coupon] * n_periods
        cash_flows[-1] += par
        discounts = [(1 + ytm/(100*freq))**(-i) for i in range(1, n_periods+1)]
        pv = [cf * d for cf, d in zip(cash_flows, discounts)]
        weighted_times = [t * v for t, v in enumerate(pv, start=1)]
        macaulay = sum(weighted_times) / sum(pv)
        return macaulay / freq

    def calculate_modified_duration(macaulay, ytm, freq):
        return macaulay / (1 + (ytm / (100 * freq)))

    def calculate_key_rate_duration(price, par, coupon_rate, ytm, n_periods, freq):
        shock = 0.01
        durations = []
        for _period in range(1, n_periods+1):
            price_up = calculate_price(par, coupon_rate, ytm + 100*shock, n_periods, freq)
            price_down = calculate_price(par, coupon_rate, ytm - 100*shock, n_periods, freq)
            kr_dur = (price_down - price_up) / (2 * price * shock)
            durations.append(kr_dur)
        return np.mean(durations)

    def calculate_convexity(price, par, coupon_rate, ytm, n_periods, freq):
        coupon = coupon_rate / 100 * par / freq
        conv_sum = 0
        for t in range(1, n_periods + 1):
            c_flow = coupon if t < n_periods else (coupon + par)
            term = (c_flow * t * (t+1)) / ((1 + ytm/(100*freq))**(t+2))
            conv_sum += term
        return (conv_sum / (price * freq**2))/10

    def calculate_convexity_callable(price, par, coupon_rate, call_price, call_date, ytm, settlement_date, freq):
        coupon = coupon_rate / 100 * par / freq
        n_periods_call = (call_date - settlement_date).days // (365 // freq)
        conv_sum = 0
        for t in range(1, n_periods_call + 1):
            c_flow = coupon if t < n_periods_call else (coupon + call_price)
            term = (c_flow * t * (t + 1)) / ((1 + ytm/(100*freq)) ** (t + 2))
            conv_sum += term
        return (conv_sum / (price * freq**2))/10

    # ----- UI for Bond Analysis -----
    bond_type = st.selectbox("Bond Type:", ["Corporate", "Treasury", "Municipal", "Agency/GSE", "Fixed Rate"])
    price = st.number_input("Market Price:", min_value=0.0, value=98.5, step=0.01)
    annual_coupon_rate = st.number_input("Annual Coupon Rate (%):", min_value=0.0, value=5.0, step=0.01)
    coupon_freq = st.selectbox("Coupon Frequency:", ["Annual", "Semi-Annual", "Quarterly", "Monthly/GSE"])
    maturity_date = st.date_input("Maturity Date:", value=(datetime.today().date() + relativedelta(years=10)))

    callable_bond = False
    error_msg = ""
    if bond_type == "Corporate":
        callable_bond = st.checkbox("Callable?")
        if callable_bond:
            call_date = st.date_input("Call Date:", value=maturity_date - relativedelta(years=1))
            call_price = st.number_input("Call Price:", min_value=0.0, value=100.0, step=0.01)
            if call_date >= maturity_date:
                error_msg = "Error: Call date must be earlier than maturity date."

    par_value = st.number_input("Par Value:", min_value=0.0, value=100.0, step=0.01)
    quantity = st.number_input("Quantity:", min_value=1, value=10, step=1)
    settlement_date = st.date_input("Settlement Date:", value=datetime.today().date())
    total_markup = st.number_input("Total Markup (USD):", min_value=0.0, value=0.0, step=0.01)
    duration_type = st.selectbox("Duration Type:", ["Macaulay", "Modified", "Key Rate"])

    if error_msg:
        st.error(error_msg)

    freq_map = {
        "Annual": 1,
        "Semi-Annual": 2,
        "Quarterly": 4,
        "Monthly/GSE": 12
    }

    colA, colB, _ = st.columns([2,1,6])
    if colA.button("Calculate"):
        if error_msg:
            st.error("Cannot calculate. Invalid call date.")
        else:
            freq = freq_map[coupon_freq]
            n_periods = (maturity_date - settlement_date).days // (365 // freq)
            if n_periods <= 0:
                st.error("Settlement date must be before the maturity date.")
            else:
                coupon_payment = (annual_coupon_rate/100)*par_value/freq
                ytm_val = calculate_ytm(price, par_value, annual_coupon_rate, n_periods, freq)

                # If callable, do YTC & callable convexity
                ytc_val = None
                conv_callable = None
                dur_callable = None
                if callable_bond:
                    ytc_val = calculate_ytc(price, par_value, annual_coupon_rate, call_price, call_date, settlement_date, freq)
                    conv_callable = calculate_convexity_callable(
                        price, par_value, annual_coupon_rate, 
                        call_price, call_date, ytm_val, settlement_date, freq
                    )
                    dur_callable = calculate_macaulay_duration(par_value, annual_coupon_rate, ytc_val, n_periods, freq)
                    if duration_type == "Modified":
                        dur_callable = calculate_modified_duration(dur_callable, ytc_val, freq)
                    elif duration_type == "Key Rate":
                        dur_callable = calculate_key_rate_duration(price, par_value, annual_coupon_rate, ytc_val, n_periods, freq)

                # Non-callable or baseline durations
                mac_dur = calculate_macaulay_duration(par_value, annual_coupon_rate, ytm_val, n_periods, freq)
                if duration_type == "Macaulay":
                    duration = mac_dur
                elif duration_type == "Modified":
                    duration = calculate_modified_duration(mac_dur, ytm_val, freq)
                else:  # Key Rate
                    duration = calculate_key_rate_duration(price, par_value, annual_coupon_rate, ytm_val, n_periods, freq)

                convexity = calculate_convexity(price, par_value, annual_coupon_rate, ytm_val, n_periods, freq)
                accrued_interest = (datetime.now().date() - settlement_date).days / 365 * (annual_coupon_rate/100)*par_value
                total_cost = price * quantity + total_markup

                results = {
                    "Metric": [
                        "Coupon Payment", "Number of Periods", "Accrued Interest",
                        "Total Cost", "Yield to Maturity (YTM)", "Duration", "Convexity"
                    ],
                    "Value": [
                        f"${coupon_payment:.2f}",
                        n_periods,
                        f"${accrued_interest:.2f}",
                        f"${total_cost:.2f}",
                        f"{ytm_val:.2f}%",
                        f"{duration:.2f} years",
                        f"{convexity:.2f}"
                    ]
                }
                if callable_bond:
                    results["Metric"].extend(["Yield to Call (YTC)", "Duration (Callable)", "Convexity (Callable)"])
                    results["Value"].extend([f"{ytc_val:.2f}%", f"{dur_callable:.2f} years", f"{conv_callable:.2f}"])

                df_out = pd.DataFrame(results)
                st.write(df_out)

                # Plot Price vs Yield
                price_range = np.linspace(price-10, price+10, 50)
                ytm_values = [calculate_ytm(p, par_value, annual_coupon_rate, n_periods, freq) for p in price_range]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=ytm_values, y=price_range,
                        mode='lines',
                        name='Yield to Maturity'
                    )
                )
                if callable_bond:
                    ytc_values = [
                        calculate_ytc(p, par_value, annual_coupon_rate, call_price, call_date, settlement_date, freq) 
                        for p in price_range
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=ytc_values, y=price_range,
                            mode='lines', line=dict(dash='dash'),
                            name='Yield to Call'
                        )
                    )
                fig.update_layout(
                    title="Price vs. Yield",
                    xaxis_title="Yield (%)",
                    yaxis_title="Price",
                    template="plotly_white"
                )
                st.plotly_chart(fig)

    # If your Streamlit version >= 1.12, you can use st.experimental_rerun
    # Otherwise, remove or replace with clearing session_state + st.stop()
    if colB.button("Reset"):
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.stop()


###############################################################################
#                          PAGE 5: OPTIONS (BLACK-SCHOLES)
###############################################################################
def page_options():
    st.title("Quantitative Finance: Black-Scholes Option Pricing (Plotly)")

    stock_symbol = st.text_input("Stock Symbol (e.g. AAPL):", 'AAPL')
    default_date = date(2024, 5, 1)
    selected_date = st.date_input("Date for stock price:", default_date)
    strike_price = st.number_input("Strike Price:", value=100.0)
    option_type = st.selectbox("Option Type:", ('call', 'put'))
    days_to_expiry = st.number_input("Days to Expiry:", min_value=1, max_value=3650, value=30, step=1)
    T = days_to_expiry / 365.25
    vol = st.number_input("Volatility (decimal):", value=0.2)
    r = st.number_input("Risk-free Rate (decimal):", value=0.05)
    q = st.number_input("Dividend Yield (decimal):", value=0.01)

    if st.button("Fetch & Calculate Option Price"):
        # Fetch stock data from FMP
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{stock_symbol}?from={selected_date}&to={selected_date}&apikey={API_KEY}"
            resp = requests.get(url)
            data = resp.json()
            if 'historical' not in data or not data['historical']:
                st.error("No data for the chosen date.")
            else:
                S = data['historical'][0]['close']
                st.write(f"**Stock Price** on {selected_date}: ${S:.2f}")

                # Adjust for dividend yield
                S_adj = S * math.exp(-q * T)

                # d1, d2
                def calc_d1_d2(Sa, K, T_, r_, sigma_):
                    d1_ = (math.log(Sa / K) + (r_ + 0.5 * sigma_**2)*T_) / (sigma_*math.sqrt(T_))
                    d2_ = d1_ - sigma_*math.sqrt(T_)
                    return d1_, d2_

                d1, d2 = calc_d1_d2(S_adj, strike_price, T, r, vol)

                # Price
                if option_type == 'call':
                    price = S_adj * norm.cdf(d1) - strike_price*math.exp(-r*T)*norm.cdf(d2)
                else:
                    price = strike_price*math.exp(-r*T)*norm.cdf(-d2) - S_adj*norm.cdf(-d1)

                st.subheader(f"Option Price ({option_type.capitalize()}): ${price:.2f}")

                # Greeks using py_vollib
                greeks = {
                    "Delta": analytical.delta(option_type[0], S_adj, strike_price, T, r, vol),
                    "Gamma": analytical.gamma(option_type[0], S_adj, strike_price, T, r, vol),
                    "Theta": analytical.theta(option_type[0], S_adj, strike_price, T, r, vol),
                    "Vega": analytical.vega(option_type[0], S_adj, strike_price, T, r, vol),
                    "Rho": analytical.rho(option_type[0], S_adj, strike_price, T, r, vol)
                }
                st.subheader(f"{option_type.capitalize()} Option Greeks")
                for g, val in greeks.items():
                    st.write(f"**{g}:** {val:.4f}")

        except Exception as e:
            st.error(f"Error fetching stock price: {str(e)}")


###############################################################################
#             PAGE 6: COMBINED SMART PORTFOLIO VISUALIZER
###############################################################################
def page_smart_visualizer():
    """
    Demonstrates a combined dashboard for:
      - Equity returns (VaR),
      - Bond or Option metrics,
      - or any extra data, all in one page.
    """
    st.title("Smart Portfolio Visualizer")

    st.write("""
    This page merges data from:
    - The **VaR** page's equity portfolio returns (if fetched),
    - The **Backtest** page's final results (if run),
    - Optionally, placeholders for bond or option data.

    In a real scenario, you'd unify these time-series from session_state or a database.
    """)

    # 1. Check if we have VaR data
    if 'portfolio_returns' in st.session_state and len(st.session_state['portfolio_returns']) > 0:
        eq_returns = st.session_state['portfolio_returns']
        length = len(eq_returns)
        df_equity = pd.DataFrame({
            'Day': np.arange(length),
            'EquityReturns': eq_returns.cumsum(),  # Cumulative sum
        })
    else:
        df_equity = pd.DataFrame()
        st.warning("No equity portfolio returns found (VaR page).")

    # 2. Check if we have backtest data
    if 'bt_portfolio_df' in st.session_state and not st.session_state['bt_portfolio_df'].empty:
        df_bt = st.session_state['bt_portfolio_df']
    else:
        df_bt = pd.DataFrame()
        st.warning("No backtest data found (Backtesting page).")

    # Build a combined figure
    fig = go.Figure()

    # Add equity returns (VaR)
    if not df_equity.empty:
        fig.add_trace(
            go.Scatter(
                x=df_equity['Day'], 
                y=df_equity['EquityReturns'], 
                mode='lines',
                name='Equity (VaR) Returns',
                line=dict(color='blue')
            )
        )

    # Add backtest portfolio value
    if not df_bt.empty:
        fig.add_trace(
            go.Scatter(
                x=df_bt['Day'],
                y=df_bt['PortfolioValue'],
                mode='lines',
                name='Backtest Portfolio Value',
                line=dict(color='green')
            )
        )

    # If you wanted placeholders for bond or options data, you could add them similarly
    # bond_returns = ...
    # option_pnl = ...
    # etc.

    # Configure the layout
    fig.update_layout(
        title="Smart Combined Visualization",
        xaxis_title="Day",
        yaxis_title="Value",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    This is just a **demo** of how you might combine:
    - The daily returns from the VaR page (blue line).
    - The backtested portfolio value (green line).
    - Additional lines for bonds/options if desired.
    """)


###############################################################################
#                                  MAIN
###############################################################################
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        [
            "Home",
            "Backtesting (New!)",
            "Value at Risk (VaR)",
            "Fixed Income",
            "Option Pricing",
            "Smart Portfolio Visualizer"
        ]
    )

    if page == "Home":
        page_home()
    elif page == "Backtesting (New!)":
        page_backtesting()
    elif page == "Value at Risk (VaR)":
        page_var()
    elif page == "Fixed Income":
        page_fixed_income()
    elif page == "Option Pricing":
        page_options()
    elif page == "Smart Portfolio Visualizer":
        page_smart_visualizer()

if __name__ == "__main__":
    main()
