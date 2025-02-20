import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Define market move factors as a constant
MARKET_MOVES = {"SPY": 1.03, "IWM": 1.036, "QQQ": 1.041}

class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Each strategy must implement the execute() method.
    """
    @abstractmethod
    def execute(self, df, date, symbol):
        """
        Executes the strategy for a given date and symbol.
        Returns a DataFrame with trade details.
        """
        pass

# ATM Strategy: Buy at-the-money call options
class ATMStrategy(TradingStrategy):
    def execute(self, df, date, symbol):
        daily_data = df[(df["Date"] == date) & (df["Symbol"] == symbol)]
        if daily_data.empty:
            return None

        stock_price = daily_data["Undl Price"].iloc[0]
        new_price = stock_price * MARKET_MOVES[symbol]

        # Find the ATM option (strike closest to or equal to stock price)
        atm_options = daily_data[daily_data["Strike"] >= stock_price]
        if atm_options.empty:
            return None

        atm_option = atm_options.iloc[0]
        K = atm_option["Strike"]
        old_intrinsic = max(stock_price - K, 0)
        new_intrinsic = max(new_price - K, 0)
        exposure_per_contract = 100 * (new_intrinsic - old_intrinsic)

        if exposure_per_contract == 0:
            return None

        # Calculate contracts needed to achieve $10M exposure
        contracts_needed = np.ceil(10_000_000 / exposure_per_contract)
        contracts_available = atm_option["Ask Size"]
        contracts_to_buy = min(contracts_needed, contracts_available)

        cost = contracts_to_buy * 100 * atm_option["Ask Price"]
        exposure_achieved = contracts_to_buy * exposure_per_contract

        result = {
            "Date": date,
            "Symbol": symbol,
            "Strategy": "ATM",
            "Strike": K,
            "Contracts Bought": contracts_to_buy,
            "Cost": cost,
            "Exposure Achieved": exposure_achieved,
        }
        return pd.DataFrame([result])

# OTM Strategy: Buy out-of-the-money call options
class OTMStrategy(TradingStrategy):
    def execute(self, df, date, symbol):
        daily_data = df[(df["Date"] == date) & (df["Symbol"] == symbol)]
        if daily_data.empty:
            return None

        stock_price = daily_data["Undl Price"].iloc[0]
        new_price = stock_price * MARKET_MOVES[symbol]

        # Find the OTM option (strike strictly greater than the stock price)
        otm_options = daily_data[daily_data["Strike"] > stock_price]
        if otm_options.empty:
            return None

        otm_option = otm_options.iloc[0]
        K = otm_option["Strike"]
        old_intrinsic = max(stock_price - K, 0)
        new_intrinsic = max(new_price - K, 0)
        exposure_per_contract = 100 * (new_intrinsic - old_intrinsic)

        if exposure_per_contract == 0:
            return None

        contracts_needed = np.ceil(10_000_000 / exposure_per_contract)
        contracts_available = otm_option["Ask Size"]
        contracts_to_buy = min(contracts_needed, contracts_available)

        cost = contracts_to_buy * 100 * otm_option["Ask Price"]
        exposure_achieved = contracts_to_buy * exposure_per_contract

        result = {
            "Date": date,
            "Symbol": symbol,
            "Strategy": "OTM",
            "Strike": K,
            "Contracts Bought": contracts_to_buy,
            "Cost": cost,
            "Exposure Achieved": exposure_achieved,
        }
        return pd.DataFrame([result])

# BuyEverything Strategy: Buy all available options for each strike
class BuyEverythingStrategy(TradingStrategy):
    def execute(self, df, date, symbol):
        # Filter the data for the specific date and symbol.
        daily_data = df[(df["Date"] == date) & (df["Symbol"] == symbol)]
        if daily_data.empty:
            return None

        # List to collect summary information for each strike.
        summary_rows = []
        for _, row in daily_data.iterrows():
            stock_price = row["Undl Price"]
            new_price = stock_price * MARKET_MOVES[symbol]

            contracts_available = row["Ask Size"]
            cost = contracts_available * 100 * row["Ask Price"]

            strike = row["Strike"]
            old_intrinsic = max(stock_price - strike, 0)
            new_intrinsic = max(new_price - strike, 0)
            exposure_per_contract = 100 * (new_intrinsic - old_intrinsic)
            exposure_achieved = contracts_available * exposure_per_contract

            summary_rows.append({
                "Date": date,
                "Symbol": symbol,
                "Strategy": "Buy Everything",
                "Strike": strike,
                "Contracts Bought": contracts_available,
                "Cost": cost,
                "Exposure Achieved": exposure_achieved,
            })

        return pd.DataFrame(summary_rows)

# Simulation Script
class OptionTradingSimulator:
    def __init__(self, df):
        """
        Initializes the simulator with the dataset.
        """
        self.df = df

    def run_simulation(self, strategy):
        """
        Runs the provided strategy on each trading day and symbol.
        For each day, aggregates the exposure across SPY, IWM, and QQQ.
        Returns a DataFrame that includes:
          - Individual symbol results.
          - A summary row for each day (Symbol = "ALL") showing total exposure,
            total cost, and a flag indicating if the collective exposure meets the $10M requirement.
        """
        results = []
        for date in sorted(self.df["Date"].unique()):
            day_results = []
            for symbol in ["SPY", "IWM", "QQQ"]:
                trade_result = strategy.execute(self.df, date, symbol)
                if trade_result is not None:
                    # If the strategy returns a DataFrame with multiple rows, add each row.
                    day_results.extend(trade_result.to_dict("records"))
            if day_results:
                total_exposure = sum(r.get("Exposure Achieved", 0) for r in day_results)
                total_cost = sum(r.get("Cost", 0) for r in day_results)
                meets_requirement = total_exposure >= 10_000_000
                summary = {
                    "Date": date,
                    "Symbol": "ALL",
                    "Strategy": strategy.__class__.__name__,
                    "Strike": None,
                    "Contracts Bought": None,
                    "Cost": total_cost,
                    "Exposure Achieved": total_exposure,
                    "Meets Exposure Requirement": meets_requirement,
                }
                day_results.append(summary)
                results.extend(day_results)
        return pd.DataFrame(results)
