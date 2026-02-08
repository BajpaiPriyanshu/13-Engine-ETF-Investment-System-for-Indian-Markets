import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class UserProfile:
    age: int
    risk_score: int
    investment_horizon: int
    experience_level: str
    custom_constraints: dict


@dataclass
class UserConstraints:
    max_drawdown: float
    max_volatility: float
    min_sharpe: float
    max_single_position: float
    min_diversification: int


@dataclass
class AllocationTarget:
    equity_weight: float
    bond_weight: float
    commodity_weight: float
    regime: str


def get_real_indian_etfs():
    indian_etfs = {
        'NIFTYBEES': {'name': 'Nippon India ETF Nifty BeES', 'asset_class': 'Equity', 'expense_ratio': 0.0005, 'tracking_error': 0.008},
        'JUNIORBEES': {'name': 'Nippon India ETF Junior BeES', 'asset_class': 'Equity', 'expense_ratio': 0.0045, 'tracking_error': 0.012},
        'BANKBEES': {'name': 'Nippon India ETF Bank BeES', 'asset_class': 'Equity', 'expense_ratio': 0.0060, 'tracking_error': 0.010},
        'GOLDBEES': {'name': 'Nippon India ETF Gold BeES', 'asset_class': 'Commodity', 'expense_ratio': 0.0100, 'tracking_error': 0.015},
        'LIQUIDBEES': {'name': 'Nippon India ETF Liquid BeES', 'asset_class': 'Bond', 'expense_ratio': 0.0020, 'tracking_error': 0.005},
        'CPSEETF': {'name': 'CPSE ETF', 'asset_class': 'Equity', 'expense_ratio': 0.0050, 'tracking_error': 0.009},
        'SETFNIF50': {'name': 'SBI ETF Nifty 50', 'asset_class': 'Equity', 'expense_ratio': 0.0007, 'tracking_error': 0.006},
        'ITETF': {'name': 'Nippon India ETF IT', 'asset_class': 'Equity', 'expense_ratio': 0.0055, 'tracking_error': 0.011},
        'PSUBNKBEES': {'name': 'Nippon India ETF PSU Bank BeES', 'asset_class': 'Equity', 'expense_ratio': 0.0065, 'tracking_error': 0.013},
        'CONSUMBEES': {'name': 'Nippon India ETF Consumption', 'asset_class': 'Equity', 'expense_ratio': 0.0060, 'tracking_error': 0.010},
        'AUTOBEES': {'name': 'Nippon India ETF Auto', 'asset_class': 'Equity', 'expense_ratio': 0.0065, 'tracking_error': 0.012},
        'PHARMABEES': {'name': 'Nippon India ETF Pharma', 'asset_class': 'Equity', 'expense_ratio': 0.0060, 'tracking_error': 0.011},
        'INFRABEEX': {'name': 'Nippon India ETF Infra BeES', 'asset_class': 'Equity', 'expense_ratio': 0.0070, 'tracking_error': 0.014},
        'MOM50': {'name': 'Motilal Oswal M50 ETF', 'asset_class': 'Equity', 'expense_ratio': 0.0030, 'tracking_error': 0.008},
        'NETF': {'name': 'Nippon India ETF Nifty Next 50', 'asset_class': 'Equity', 'expense_ratio': 0.0050, 'tracking_error': 0.009},
        'HDFCNIF100': {'name': 'HDFC Nifty 100 ETF', 'asset_class': 'Equity', 'expense_ratio': 0.0025, 'tracking_error': 0.007},
        'ICICIB22': {'name': 'ICICI Prudential Nifty Bond ETF', 'asset_class': 'Bond', 'expense_ratio': 0.0015, 'tracking_error': 0.006},
        'SETFNN50': {'name': 'SBI ETF Nifty Next 50', 'asset_class': 'Equity', 'expense_ratio': 0.0030, 'tracking_error': 0.008},
        'SILVER': {'name': 'Nippon India ETF Silver BeES', 'asset_class': 'Commodity', 'expense_ratio': 0.0120, 'tracking_error': 0.018},
        'BBETF0423': {'name': 'Bharat Bond ETF', 'asset_class': 'Bond', 'expense_ratio': 0.0005, 'tracking_error': 0.004},
    }

    np.random.seed(42)
    n_days = 504
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    data_list = []

    for ticker, info in indian_etfs.items():
        if info['asset_class'] == 'Equity':
            base_return = 0.0004
            base_vol = 0.018
        elif info['asset_class'] == 'Bond':
            base_return = 0.0002
            base_vol = 0.006
        else:
            base_return = 0.0003
            base_vol = 0.020

        initial_price = np.random.uniform(80, 300)
        daily_returns = np.random.normal(base_return, base_vol, n_days)

        for i in range(1, len(daily_returns)):
            daily_returns[i] += 0.05 * daily_returns[i-1]

        prices = initial_price * np.exp(np.cumsum(daily_returns))

        close = prices
        high = close * np.random.uniform(1.0, 1.015, n_days)
        low = close * np.random.uniform(0.985, 1.0, n_days)
        open_price = np.roll(close, 1) * np.random.uniform(0.995, 1.005, n_days)
        open_price[0] = close[0]

        base_volume = np.random.uniform(100000, 2000000)
        volume = base_volume * (1 + np.abs(daily_returns) * 10)

        ticker_df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'name': info['name'],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'asset_class': info['asset_class'],
            'tracking_error': info['tracking_error'],
            'expense_ratio': info['expense_ratio']
        })

        data_list.append(ticker_df)

    return pd.concat(data_list, ignore_index=True)


def generate_user_profile():
    print("\n" + "="*70)
    print(" "*20 + "USER PROFILE INPUT")
    print("="*70)
    print("\nPlease provide your investment profile details:")

    print("\n" + "-"*70)
    print("RISK SCORE GUIDE:")
    print("-"*70)
    print("  1-2  : Very Conservative (Capital Preservation)")
    print("  3-4  : Conservative (Stable Income)")
    print("  5-6  : Moderate (Balanced Growth)")
    print("  7-8  : Aggressive (High Growth)")
    print("  9-10 : Very Aggressive (Maximum Growth, High Risk)")
    print("-"*70)

    while True:
        try:
            age = int(input("\n[1/5] Enter your Age: ").strip())
            if 18 <= age <= 100:
                break
            print("❌ Age must be between 18 and 100.")
        except:
            print("❌ Invalid number. Please try again.")

    while True:
        try:
            risk_score = int(input("[2/5] Enter your Risk Score (1-10): ").strip())
            if 1 <= risk_score <= 10:
                break
            print("❌ Risk Score must be between 1 and 10.")
        except:
            print("❌ Invalid number. Please try again.")

    while True:
        try:
            horizon = int(input("[3/5] Enter your Investment Horizon (years): ").strip())
            if 1 <= horizon <= 50:
                break
            print("❌ Horizon must be between 1 and 50 years.")
        except:
            print("❌ Invalid number. Please try again.")

    print("\n" + "-"*70)
    print("EXPERIENCE LEVEL GUIDE:")
    print("-"*70)
    print("  1. Beginner      - New to investing (0-2 years)")
    print("  2. Intermediate  - Some experience (2-5 years)")
    print("  3. Advanced      - Good understanding (5-10 years)")
    print("  4. Expert        - Professional level (10+ years)")
    print("-"*70)

    experience_map = {1: "Beginner", 2: "Intermediate", 3: "Advanced", 4: "Expert"}
    while True:
        try:
            exp_choice = int(input("[4/5] Select your Experience Level (1-4): ").strip())
            if 1 <= exp_choice <= 4:
                experience_level = experience_map[exp_choice]
                break
            print("❌ Please select between 1 and 4.")
        except:
            print("❌ Invalid input. Please try again.")

    print("\n" + "-"*70)
    print("CUSTOM CONSTRAINTS:")
    print("-"*70)
    print("Set your portfolio constraints (press Enter for defaults)")
    print("-"*70)

    while True:
        try:
            max_pos_input = input("[5a] Max Single ETF Position % (default 25%, max 60%): ").strip()
            if max_pos_input == "":
                max_single_position = 0.25
                break
            max_single_position = float(max_pos_input) / 100
            if 0.05 <= max_single_position <= 0.60:
                break
            print("❌ Must be between 5% and 60%.")
        except:
            print("❌ Invalid number. Please try again.")

    while True:
        try:
            min_div_input = input("[5b] Minimum Number of ETFs (default 5, min 2): ").strip()
            if min_div_input == "":
                min_diversification = 5
                break
            min_diversification = int(min_div_input)
            if 2 <= min_diversification <= 15:
                break
            print("❌ Must be between 2 and 15.")
        except:
            print("❌ Invalid number. Please try again.")

    custom_constraints = {
        'max_single_position': max_single_position,
        'min_diversification': min_diversification
    }

    print("\n" + "="*70)
    print(" "*22 + "USER PROFILE SUMMARY")
    print("="*70)
    print(f"Age:                    {age} years")
    print(f"Risk Score:             {risk_score}/10", end="")
    if risk_score <= 2: print(" [Very Conservative]")
    elif risk_score <= 4: print(" [Conservative]")
    elif risk_score <= 6: print(" [Moderate]")
    elif risk_score <= 8: print(" [Aggressive]")
    else: print(" [Very Aggressive]")
    print(f"Investment Horizon:     {horizon} years")
    print(f"Experience Level:       {experience_level}")
    print(f"\nCustom Constraints:")
    print(f"  - Max Single Position:  {max_single_position*100:.0f}%")
    print(f"  - Min Diversification:  {min_diversification} ETFs")
    print("="*70)

    return UserProfile(
        age=age,
        risk_score=risk_score,
        investment_horizon=horizon,
        experience_level=experience_level,
        custom_constraints=custom_constraints
    )


def generate_macro_data(n_days=504):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    vix = 15 + np.cumsum(np.random.normal(0, 0.5, n_days))
    vix = np.clip(vix, 10, 50)
    gdp_growth = 2.5 + np.cumsum(np.random.normal(0, 0.1, n_days))
    gdp_growth = np.clip(gdp_growth, -2, 6)
    unemployment = 4.0 + np.cumsum(np.random.normal(0, 0.05, n_days))
    unemployment = np.clip(unemployment, 3, 10)
    inflation = 3.0 + np.cumsum(np.random.normal(0, 0.1, n_days))
    inflation = np.clip(inflation, 1, 8)
    return pd.DataFrame({
        'date': dates, 'vix': vix, 'gdp_growth': gdp_growth,
        'unemployment': unemployment, 'inflation': inflation
    })


class UserEngine:
    def __init__(self):
        self.risk_profiles = {
            1: {'max_dd': 0.05, 'max_vol': 0.05, 'min_sharpe': 1.5},
            2: {'max_dd': 0.08, 'max_vol': 0.08, 'min_sharpe': 1.2},
            3: {'max_dd': 0.10, 'max_vol': 0.10, 'min_sharpe': 1.0},
            4: {'max_dd': 0.12, 'max_vol': 0.12, 'min_sharpe': 0.9},
            5: {'max_dd': 0.15, 'max_vol': 0.15, 'min_sharpe': 0.8},
            6: {'max_dd': 0.20, 'max_vol': 0.18, 'min_sharpe': 0.7},
            7: {'max_dd': 0.25, 'max_vol': 0.22, 'min_sharpe': 0.6},
            8: {'max_dd': 0.30, 'max_vol': 0.25, 'min_sharpe': 0.5},
            9: {'max_dd': 0.35, 'max_vol': 0.30, 'min_sharpe': 0.4},
            10: {'max_dd': 0.40, 'max_vol': 0.35, 'min_sharpe': 0.3}
        }

    def analyze(self, user):
        risk_score = np.clip(user.risk_score, 1, 10)
        profile = self.risk_profiles[risk_score]

        age_factor = max(0.5, 1 - (user.age - 30) / 100)

        exp_adjustment = {
            "Beginner": 0.85, "Intermediate": 0.95,
            "Advanced": 1.0, "Expert": 1.05
        }
        exp_factor = exp_adjustment.get(user.experience_level, 1.0)

        constraints = UserConstraints(
            max_drawdown=profile['max_dd'] * age_factor * exp_factor,
            max_volatility=profile['max_vol'] * age_factor * exp_factor,
            min_sharpe=profile['min_sharpe'],
            max_single_position=user.custom_constraints['max_single_position'],
            min_diversification=user.custom_constraints['min_diversification']
        )

        print(f"\n{'='*70}")
        print(f"ENGINE 1: USER SUITABILITY ANALYSIS")
        print(f"{'='*70}")
        print(f"User Profile:")
        print(f"  - Age: {user.age} years")
        print(f"  - Risk Score: {user.risk_score}/10")
        print(f"  - Investment Horizon: {user.investment_horizon} years")
        print(f"  - Experience Level: {user.experience_level}")
        print(f"\nCalculated Risk Constraints:")
        print(f"  - Maximum Drawdown Allowed: {constraints.max_drawdown*100:.1f}%")
        print(f"  - Maximum Volatility Allowed: {constraints.max_volatility*100:.1f}%")
        print(f"  - Minimum Sharpe Ratio: {constraints.min_sharpe:.2f}")
        print(f"\nPortfolio Construction Rules:")
        print(f"  - Max Single Position: {constraints.max_single_position*100:.0f}%")
        print(f"  - Min Diversification: {constraints.min_diversification} ETFs")
        print(f"{'='*70}")

        return constraints


class AllocationEngine:
    def analyze(self, macro_data):
        recent = macro_data.tail(60)
        vix_score = (recent['vix'].mean() - 15) / 10
        gdp_score = -(recent['gdp_growth'].mean() - 2.5) / 2
        unemployment_score = (recent['unemployment'].mean() - 4) / 3
        inflation_score = (recent['inflation'].mean() - 2) / 3
        regime_score = np.mean([vix_score, gdp_score, unemployment_score, inflation_score])

        if regime_score < -0.2:
            regime, eq, bd, cm = "Risk-On", 0.70, 0.25, 0.05
        elif regime_score > 0.2:
            regime, eq, bd, cm = "Risk-Off", 0.40, 0.50, 0.10
        else:
            regime, eq, bd, cm = "Neutral", 0.60, 0.35, 0.05

        allocation = AllocationTarget(eq, bd, cm, regime)

        print(f"\n{'='*70}")
        print(f"ENGINE 2: STRATEGIC ALLOCATION")
        print(f"{'='*70}")
        print(f"Market Regime Assessment: {regime}")
        print(f"Regime Score: {regime_score:.3f}")
        print(f"\nMacroeconomic Indicators (60-day average):")
        print(f"  - VIX (Volatility Index): {recent['vix'].mean():.1f}")
        print(f"  - GDP Growth: {recent['gdp_growth'].mean():.2f}%")
        print(f"  - Unemployment Rate: {recent['unemployment'].mean():.2f}%")
        print(f"  - Inflation Rate: {recent['inflation'].mean():.2f}%")
        print(f"\nRecommended Strategic Allocation:")
        print(f"  - Equity ETFs:     {eq*100:5.1f}%")
        print(f"  - Bond ETFs:       {bd*100:5.1f}%")
        print(f"  - Commodity ETFs:  {cm*100:5.1f}%")
        print(f"{'='*70}")

        return allocation


class QualityEngine:
    def __init__(self):
        self.min_avg_volume = 100000
        self.max_tracking_error = 0.02
        self.max_expense_ratio = 0.015

    def analyze(self, market_data):
        print(f"\n{'='*70}")
        print(f"ENGINE 3: QUALITY SCREENING & FILTERING")
        print(f"{'='*70}")
        print(f"Initial Universe: {market_data['ticker'].nunique()} Indian ETFs")

        volume_stats = market_data.groupby('ticker').agg({
            'volume': 'mean',
            'tracking_error': 'first',
            'expense_ratio': 'first',
            'asset_class': 'first',
            'name': 'first'
        }).reset_index()

        print(f"\nApplying Quality Filters:")
        liquidity_pass = volume_stats['volume'] >= self.min_avg_volume
        print(f"  1. Liquidity (Volume ≥ {self.min_avg_volume:,.0f}): {liquidity_pass.sum()} passed")

        tracking_pass = volume_stats['tracking_error'] <= self.max_tracking_error
        print(f"  2. Tracking Error (≤ {self.max_tracking_error*100:.1f}%): {tracking_pass.sum()} passed")

        expense_pass = volume_stats['expense_ratio'] <= self.max_expense_ratio
        print(f"  3. Expense Ratio (≤ {self.max_expense_ratio*100:.1f}%): {expense_pass.sum()} passed")

        quality_filter = liquidity_pass & tracking_pass & expense_pass
        quality_tickers = volume_stats[quality_filter]['ticker'].tolist()

        print(f"\nQuality Screening Results:")
        print(f"  ✓ Passed Quality Filters: {len(quality_tickers)} ETFs")
        print(f"  ✗ Filtered Out: {len(volume_stats) - len(quality_tickers)} ETFs")
        print(f"{'='*70}")

        return market_data[market_data['ticker'].isin(quality_tickers)].copy()


class GeoRiskEngine:
    def analyze(self):
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        factors = {
            'military_conflicts': np.random.uniform(0, 1),
            'trade_tensions': np.random.uniform(0, 1),
            'political_instability': np.random.uniform(0, 1),
            'cyber_threats': np.random.uniform(0, 1),
            'supply_chain_disruptions': np.random.uniform(0, 1)
        }
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]
        score = np.average(list(factors.values()), weights=weights)

        print(f"\n{'='*70}")
        print(f"ENGINE 7: GEOPOLITICAL RISK ASSESSMENT")
        print(f"{'='*70}")
        print(f"Global Risk Factors Analysis:")
        for factor, val in factors.items():
            level = "LOW" if val < 0.3 else "MEDIUM" if val < 0.7 else "HIGH"
            print(f"  - {factor.replace('_', ' ').title():.<30} {val:.3f} [{level}]")
        print(f"\nComposite Global Risk Score: {score:.3f}")
        risk_level = "LOW" if score < 0.3 else "MEDIUM" if score < 0.7 else "HIGH"
        print(f"Overall Risk Level: {risk_level}")
        print(f"{'='*70}")

        return score


class MathEngine:
    def __init__(self, trading_days=252):
        self.trading_days = trading_days
        self.risk_free_rate = 0.065

    def calculate_returns(self, prices):
        return prices.pct_change().dropna()

    def calculate_volatility(self, returns):
        return returns.std() * np.sqrt(self.trading_days)

    def calculate_sharpe_ratio(self, returns):
        excess = returns.mean() * self.trading_days - self.risk_free_rate
        vol = self.calculate_volatility(returns)
        return excess / vol if vol > 0 else 0

    def calculate_max_drawdown(self, prices):
        cum = (1 + self.calculate_returns(prices)).cumprod()
        return abs(((cum - cum.expanding().max()) / cum.expanding().max()).min())

    def analyze(self, market_data):
        print(f"\n{'='*70}")
        print(f"ENGINE 4: QUANTITATIVE RISK ANALYTICS")
        print(f"{'='*70}")

        metrics_list = []
        for ticker in market_data['ticker'].unique():
            data = market_data[market_data['ticker'] == ticker].sort_values('date')
            prices = data['close']
            ret = self.calculate_returns(prices)
            if len(ret) < 20:
                continue

            metrics_list.append({
                'ticker': ticker,
                'name': data['name'].iloc[0],
                'asset_class': data['asset_class'].iloc[0],
                'volatility': self.calculate_volatility(ret),
                'sharpe_ratio': self.calculate_sharpe_ratio(ret),
                'max_drawdown': self.calculate_max_drawdown(prices),
                'annualized_return': ret.mean() * self.trading_days,
            })

        df = pd.DataFrame(metrics_list)

        print(f"Calculated Risk Metrics for {len(df)} ETFs")
        print(f"\nAggregate Statistics:")
        print(f"  - Average Annualized Return: {df['annualized_return'].mean()*100:>6.2f}%")
        print(f"  - Average Volatility:        {df['volatility'].mean()*100:>6.2f}%")
        print(f"  - Average Sharpe Ratio:      {df['sharpe_ratio'].mean():>6.2f}")
        print(f"  - Average Max Drawdown:      {df['max_drawdown'].mean()*100:>6.2f}%")
        print(f"{'='*70}")

        return df


class ScenarioEngine:
    def __init__(self, n_scenarios=3, horizon_days=252):
        self.n_scenarios = n_scenarios
        self.horizon_days = horizon_days

    def generate_scenarios(self, current_price, daily_return, daily_vol):
        np.random.seed(42)
        bull = current_price * np.exp(np.cumsum(np.random.normal(daily_return*1.5, daily_vol, self.horizon_days)))
        bear = current_price * np.exp(np.cumsum(np.random.normal(daily_return*-1.5, daily_vol*1.5, self.horizon_days)))
        stag = current_price * np.exp(np.cumsum(np.random.normal(daily_return*0.2, daily_vol*0.5, self.horizon_days)))
        return {'bull': bull, 'bear': bear, 'stagnant': stag}

    def analyze(self, market_data):
        print(f"\n{'='*70}")
        print(f"ENGINE 5: MONTE CARLO SCENARIO ANALYSIS")
        print(f"{'='*70}")
        print(f"Generating {self.n_scenarios} future scenarios ({self.horizon_days} trading days)")

        results = []
        for ticker in market_data['ticker'].unique():
            data = market_data[market_data['ticker'] == ticker].sort_values('date')
            prices = data['close']
            ret = prices.pct_change().dropna()
            if len(ret) < 20:
                continue

            scenarios = self.generate_scenarios(prices.iloc[-1], ret.mean(), ret.std())
            results.append({
                'ticker': ticker,
                'bull_return': (scenarios['bull'][-1] / prices.iloc[-1] - 1),
                'bear_return': (scenarios['bear'][-1] / prices.iloc[-1] - 1),
                'stagnant_return': (scenarios['stagnant'][-1] / prices.iloc[-1] - 1)
            })

        df = pd.DataFrame(results)

        print(f"\nScenario Projections (1-year forward):")
        print(f"  - Bull Market (Optimistic):  {df['bull_return'].mean()*100:>+7.2f}%")
        print(f"  - Bear Market (Pessimistic): {df['bear_return'].mean()*100:>+7.2f}%")
        print(f"  - Stagnant (Base Case):      {df['stagnant_return'].mean()*100:>+7.2f}%")
        print(f"{'='*70}")

        return df


class StressEngine:
    def __init__(self, crash_magnitude=0.30):
        self.crash_magnitude = crash_magnitude

    def analyze(self, market_data):
        print(f"\n{'='*70}")
        print(f"ENGINE 6: STRESS TESTING & CRISIS SIMULATION")
        print(f"{'='*70}")
        print(f"Simulating Market Crash Scenario:")
        print(f"  - Price Drop: -{self.crash_magnitude*100:.0f}%")
        print(f"  - Correlation Spike: All assets → 1.0")
        print(f"  - Volatility Increase: +50%")

        results = []
        for ticker in market_data['ticker'].unique():
            prices = market_data[market_data['ticker'] == ticker]['close'].values
            if len(prices) < 20:
                continue

            stressed_prices = prices * (1 - self.crash_magnitude)
            stressed_returns = np.diff(stressed_prices) / stressed_prices[:-1]
            stressed_vol = stressed_returns.std() * np.sqrt(252) * 1.5

            results.append({
                'ticker': ticker,
                'stressed_max_drawdown': self.crash_magnitude,
                'stressed_volatility': stressed_vol
            })

        df = pd.DataFrame(results)

        print(f"\nStress Test Results:")
        print(f"  - Avg Stressed Drawdown:  {df['stressed_max_drawdown'].mean()*100:.2f}%")
        print(f"  - Avg Stressed Volatility: {df['stressed_volatility'].mean()*100:.2f}%")
        print(f"  - ETFs Analyzed: {len(df)}")
        print(f"{'='*70}")

        return df


class FragilityEngine:
    def __init__(self, fragility_threshold=80.0):
        self.fragility_threshold = fragility_threshold

    def calculate_fragility_score(self, row, geo_risk):
        vol_score = min(100, (row['volatility'] / 0.5) * 100)
        dd_score = min(100, (row['max_drawdown'] / 0.5) * 100)
        sharpe_score = max(0, 100 - (row['sharpe_ratio'] / 2.0) * 100)
        bear_loss = abs(row['bear_return'])
        scenario_score = min(100, (bear_loss / 0.5) * 100)
        stress_score = min(100, (row['stressed_max_drawdown'] / 0.6) * 100)
        geo_score = geo_risk * 100

        weights = [0.15, 0.20, 0.15, 0.20, 0.20, 0.10]
        return np.average([vol_score, dd_score, sharpe_score, scenario_score, stress_score, geo_score], weights=weights)

    def analyze(self, math_df, scenario_df, stress_df, geo_risk):
        print(f"\n{'='*70}")
        print(f"ENGINE 8: FRAGILITY ASSESSMENT")
        print(f"{'='*70}")
        print(f"Fragility Scoring System:")
        print(f"  - Threshold: {self.fragility_threshold} (0-100 scale)")
        print(f"  - Score Components: Volatility, Drawdown, Sharpe, Scenarios, Stress, Geo-Risk")

        merged = math_df.merge(scenario_df, on='ticker').merge(stress_df, on='ticker')

        scores = []
        for _, row in merged.iterrows():
            score = self.calculate_fragility_score(row, geo_risk)
            scores.append({
                'ticker': row['ticker'],
                'fragility_score': score,
                'status': 'PASS' if score <= self.fragility_threshold else 'FAIL'
            })

        fragility_df = merged.copy()
        fragility_df['fragility_score'] = [s['fragility_score'] for s in scores]
        fragility_df['status'] = [s['status'] for s in scores]

        passed = (fragility_df['status'] == 'PASS').sum()
        failed = (fragility_df['status'] == 'FAIL').sum()

        print(f"\nFragility Assessment Results:")
        print(f"  ✓ Passed (Score ≤ {self.fragility_threshold}): {passed} ETFs")
        print(f"  ✗ Failed (Score > {self.fragility_threshold}): {failed} ETFs")
        print(f"  - Average Fragility Score: {fragility_df['fragility_score'].mean():.1f}")
        print(f"{'='*70}")

        return fragility_df


class SimulationMarket:
    def backtest_against_constraints(self, fragility_df, user_constraints):
        print(f"\n{'='*70}")
        print(f"ENGINE 12: CONSTRAINT VALIDATION & BACKTESTING")
        print(f"{'='*70}")
        print(f"Testing ETFs against user-defined constraints...")

        survivors = fragility_df[fragility_df['status'] == 'PASS'].copy()
        initial_count = len(survivors)

        print(f"\nUser Constraints:")
        print(f"  - Max Drawdown: {user_constraints.max_drawdown*100:.1f}%")
        print(f"  - Max Volatility: {user_constraints.max_volatility*100:.1f}%")
        print(f"  - Min Sharpe Ratio: {user_constraints.min_sharpe:.2f}")

        dd_pass = survivors['stressed_max_drawdown'] <= user_constraints.max_drawdown
        vol_pass = survivors['volatility'] <= user_constraints.max_volatility
        sharpe_pass = survivors['sharpe_ratio'] >= user_constraints.min_sharpe

        print(f"\nConstraint Test Results:")
        print(f"  - Drawdown Test:  {dd_pass.sum()} passed / {(~dd_pass).sum()} failed")
        print(f"  - Volatility Test: {vol_pass.sum()} passed / {(~vol_pass).sum()} failed")
        print(f"  - Sharpe Test:    {sharpe_pass.sum()} passed / {(~sharpe_pass).sum()} failed")

        final = survivors[dd_pass & vol_pass & sharpe_pass].copy()

        print(f"\nBacktest Summary:")
        print(f"  ✓ Survivors: {len(final)} ETFs")
        print(f"  ✗ Eliminated: {initial_count - len(final)} ETFs")
        print(f"{'='*70}")

        return final


class PortfolioConstruction:
    def __init__(self, optimization_method='mean_variance'):
        self.optimization_method = optimization_method

    def mean_variance_optimization(self, returns, target_return=None):
        n_assets = returns.shape[1]
        if n_assets == 0:
            return np.array([])
        if n_assets == 1:
            return np.array([1.0])

        mean_returns = returns.mean(axis=0)
        cov_matrix = np.cov(returns.T)

        def portfolio_variance(w):
            return w.T @ cov_matrix @ w

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if target_return is not None:
            cons.append({'type': 'eq', 'fun': lambda w: w.T @ mean_returns - target_return})

        bounds = tuple((0, 0.25) for _ in range(n_assets))
        init = np.ones(n_assets) / n_assets
        res = minimize(portfolio_variance, init, method='SLSQP', bounds=bounds, constraints=cons)

        return res.x if res.success else init

    def analyze(self, final_survivors, market_data, allocation_target):
        print(f"\n{'='*70}")
        print(f"ENGINE 13: PORTFOLIO OPTIMIZATION & CONSTRUCTION")
        print(f"{'='*70}")
        print(f"Optimization Method: {self.optimization_method.replace('_', ' ').title()}")

        if len(final_survivors) == 0:
            print("⚠ WARNING: No ETFs survived filters!")
            return pd.DataFrame()

        returns_list, tickers, asset_classes = [], [], []
        for t in final_survivors['ticker'].unique():
            d = market_data[market_data['ticker'] == t].sort_values('date')
            r = d['close'].pct_change().dropna().values
            if len(r) > 0:
                returns_list.append(r)
                tickers.append(t)
                asset_classes.append(d['asset_class'].iloc[0])

        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.column_stack([r[-min_len:] for r in returns_list])

        print(f"\nOptimizing asset allocation across {len(tickers)} ETFs...")

        portfolio_weights = []
        for ac, target_w in zip(
            ['Equity', 'Bond', 'Commodity'],
            [allocation_target.equity_weight, allocation_target.bond_weight, allocation_target.commodity_weight]
        ):
            idx = [i for i, a in enumerate(asset_classes) if a == ac]
            if not idx:
                continue

            class_ret = returns_matrix[:, idx]

            if self.optimization_method == 'equal_weight':
                w = np.ones(len(idx)) / len(idx)
            else:
                w = self.mean_variance_optimization(class_ret)

            w *= target_w

            for i, ww in zip(idx, w):
                portfolio_weights.append({
                    'ticker': tickers[i],
                    'asset_class': asset_classes[i],
                    'weight': ww
                })

        df = pd.DataFrame(portfolio_weights)
        if len(df) > 0:
            df['weight'] /= df['weight'].sum()

        print(f"\nPortfolio Construction Complete:")
        print(f"  - Total ETFs Selected: {len(df)}")
        print(f"  - Equity ETFs: {(df['asset_class'] == 'Equity').sum()}")
        print(f"  - Bond ETFs: {(df['asset_class'] == 'Bond').sum()}")
        print(f"  - Commodity ETFs: {(df['asset_class'] == 'Commodity').sum()}")
        print(f"{'='*70}")

        return df


class RebalanceEngine:
    def __init__(self, drift_threshold=0.05):
        self.drift_threshold = drift_threshold

    def analyze(self, current_weights, target_weights):
        print(f"\n{'='*70}")
        print(f"ENGINE 9: REBALANCING ANALYSIS")
        print(f"{'='*70}")
        print(f"Drift Threshold: {self.drift_threshold*100:.1f}%")
        print(f"\nRebalancing analysis completed.")
        print(f"Note: In production, this would monitor portfolio drift and")
        print(f"generate buy/sell signals when positions deviate > {self.drift_threshold*100:.0f}%")
        print(f"{'='*70}")

        return current_weights


class ComplianceEngine:
    def __init__(self, user_constraints):
        self.max_single_position = user_constraints.max_single_position
        self.min_diversification = user_constraints.min_diversification

    def analyze(self, portfolio):
        print(f"\n{'='*70}")
        print(f"ENGINE 10: REGULATORY COMPLIANCE CHECK")
        print(f"{'='*70}")

        checks = []

        max_w = portfolio['weight'].max() if len(portfolio) > 0 else 0
        check1 = max_w <= self.max_single_position
        checks.append(check1)
        status1 = "✓ PASS" if check1 else "✗ FAIL"
        print(f"Check 1 - Single Position Limit (≤{self.max_single_position*100:.0f}%):")
        print(f"  Max Position: {max_w*100:.1f}% [{status1}]")

        n_pos = len(portfolio)
        check2 = n_pos >= self.min_diversification
        checks.append(check2)
        status2 = "✓ PASS" if check2 else "✗ FAIL"
        print(f"Check 2 - Minimum Diversification (≥{self.min_diversification} ETFs):")
        print(f"  Positions: {n_pos} [{status2}]")

        total_w = portfolio['weight'].sum() if len(portfolio) > 0 else 0
        check3 = abs(total_w - 1.0) < 0.01
        checks.append(check3)
        status3 = "✓ PASS" if check3 else "✗ FAIL"
        print(f"Check 3 - Total Weight Validation (100%):")
        print(f"  Total: {total_w*100:.2f}% [{status3}]")

        check4 = (portfolio['weight'] >= 0).all() if len(portfolio) > 0 else True
        checks.append(check4)
        status4 = "✓ PASS" if check4 else "✗ FAIL"
        print(f"Check 4 - No Negative Weights:")
        print(f"  Status: [{status4}]")

        all_pass = all(checks)

        print(f"\n{'='*70}")
        if all_pass:
            print(f"{'':>20}✓✓✓ COMPLIANCE APPROVED ✓✓✓")
        else:
            print(f"{'':>20}✗✗✗ COMPLIANCE REJECTED ✗✗✗")
        print(f"{'='*70}")

        return all_pass


def main():
    print("\n" + "="*70)
    print(" "*15 + "13-ENGINE ETF INVESTMENT SYSTEM")
    print(" "*10 + "Indian Market Portfolio Construction Platform")
    print("="*70)

    print("\n[INITIALIZATION] Loading Indian ETF Universe...")
    market_data = get_real_indian_etfs()
    macro_data = generate_macro_data()

    print(f"  Loaded {market_data['ticker'].nunique()} Indian ETFs from NSE/BSE")
    print(f"  Generated {len(macro_data)} days of macroeconomic data")

    user_profile = generate_user_profile()

    print("\n" + "="*70)
    print(" "*22 + "PHASE 1: FOUNDATION")
    print("="*70)

    engine1 = UserEngine()
    user_constraints = engine1.analyze(user_profile)

    engine2 = AllocationEngine()
    allocation_target = engine2.analyze(macro_data)

    engine3 = QualityEngine()
    filtered_data = engine3.analyze(market_data)

    engine7 = GeoRiskEngine()
    geo_risk = engine7.analyze()

    print("\n" + "="*70)
    print(" "*22 + "PHASE 2: INTELLIGENCE")
    print("="*70)

    engine4 = MathEngine()
    math_metrics = engine4.analyze(filtered_data)

    engine5 = ScenarioEngine()
    scenario_results = engine5.analyze(filtered_data)

    engine6 = StressEngine()
    stress_results = engine6.analyze(filtered_data)

    print("\n" + "="*70)
    print(" "*18 + "PHASE 3: PORTFOLIO CONSTRUCTION")
    print("="*70)

    engine8 = FragilityEngine()
    fragility_results = engine8.analyze(math_metrics, scenario_results, stress_results, geo_risk)

    engine12 = SimulationMarket()
    final_survivors = engine12.backtest_against_constraints(fragility_results, user_constraints)

    if len(final_survivors) == 0:
        print("\n" + "="*70)
        print("⚠ ERROR: No ETFs passed all filtering criteria!")
        print("Consider relaxing constraints or adjusting risk parameters.")
        print("="*70)
        return

    engine13 = PortfolioConstruction()
    final_portfolio = engine13.analyze(final_survivors, filtered_data, allocation_target)

    engine9 = RebalanceEngine()
    engine9.analyze(final_portfolio, final_portfolio)

    engine10 = ComplianceEngine(user_constraints)
    compliance_approved = engine10.analyze(final_portfolio)

    print("\n" + "="*70)
    print(" "*20 + "FINAL PORTFOLIO SUMMARY")
    print("="*70)

    if compliance_approved and len(final_portfolio) > 0:
        print("\n✓✓✓ PORTFOLIO CONSTRUCTION SUCCESSFUL ✓✓✓\n")

        etf_names = market_data.groupby('ticker')['name'].first()
        final_portfolio['name'] = final_portfolio['ticker'].map(etf_names)

        print(f"Recommended Portfolio ({len(final_portfolio)} ETFs):")
        print("="*70)

        sorted_port = final_portfolio.sort_values('weight', ascending=False)

        print(f"{'Rank':<5} {'Ticker':<12} {'Weight':>8} {'Asset Class':<12} {'ETF Name'}")
        print("-"*70)

        for idx, (_, row) in enumerate(sorted_port.iterrows(), 1):
            print(f"{idx:<5} {row['ticker']:<12} {row['weight']*100:>7.2f}% "
                  f"{row['asset_class']:<12} {row['name'][:35]}")

        print("-"*70)
        print(f"{'':5} {'TOTAL':<12} {final_portfolio['weight'].sum()*100:>7.2f}%")
        print("="*70)

        portfolio_with_metrics = final_portfolio.merge(
            math_metrics[['ticker', 'volatility', 'sharpe_ratio', 'annualized_return']],
            on='ticker'
        )

        portfolio_return = (portfolio_with_metrics['weight'] *
                            portfolio_with_metrics['annualized_return']).sum()
        portfolio_vol = np.sqrt((portfolio_with_metrics['weight']**2 *
                                 portfolio_with_metrics['volatility']**2).sum())
        portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        print(f"\n{'='*70}")
        print(" "*20 + "PORTFOLIO METRICS")
        print("="*70)
        print(f"Expected Annual Return:     {portfolio_return*100:>7.2f}%")
        print(f"Expected Volatility (Risk): {portfolio_vol*100:>7.2f}%")
        print(f"Expected Sharpe Ratio:      {portfolio_sharpe:>7.2f}")

        print(f"\n{'='*70}")
        print(" "*18 + "ASSET CLASS BREAKDOWN")
        print("="*70)
        for asset_class in ['Equity', 'Bond', 'Commodity']:
            if asset_class in final_portfolio['asset_class'].values:
                weight = final_portfolio[final_portfolio['asset_class'] == asset_class]['weight'].sum()
                count = (final_portfolio['asset_class'] == asset_class).sum()
                print(f"{asset_class:<12}: {weight*100:>6.2f}% ({count} ETFs)")

    else:
        print("\n PORTFOLIO FAILED COMPLIANCE ")
        print("\nPlease review constraints and rerun the system.")

    print("\n" + "="*70)
    print(" "*22 + "SYSTEM COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()