import pandas as pd

class OptionTradingSimulator:
    def __init__(self, df):
        """
        Initializes the simulator with the dataset.
        """
        self.df = df

    def run_simulation(self, strategy):
        """
        Runs the provided strategy on each trading day and symbol.
        Returns a DataFrame with the simulation results.
        """
        results = []
        for date in self.df["Date"].unique():
            for symbol in ["SPY", "IWM", "QQQ"]:
                trade_result = strategy.execute(self.df, date, symbol)
                if trade_result:
                    results.append(trade_result)
        return pd.DataFrame(results)
