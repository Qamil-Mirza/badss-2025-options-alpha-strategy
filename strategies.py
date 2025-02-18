import numpy as np
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
        Returns a dictionary with trade details.
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
        exposure_met = exposure_achieved >= 10_000_000

        return {
            "Date": date,
            "Symbol": symbol,
            "Strategy": "ATM",
            "Strike": K,
            "Contracts Bought": contracts_to_buy,
            "Cost": cost,
            "Exposure Achieved": exposure_achieved,
            "Exposure Met": exposure_met
        }

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
        exposure_met = exposure_achieved >= 10_000_000

        return {
            "Date": date,
            "Symbol": symbol,
            "Strategy": "OTM",
            "Strike": K,
            "Contracts Bought": contracts_to_buy,
            "Cost": cost,
            "Exposure Achieved": exposure_achieved,
            "Exposure Met": exposure_met
        }
