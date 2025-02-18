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
        For each day, aggregates the exposure across SPY, IWM, and QQQ.
        Returns a DataFrame that includes:
          - Individual symbol results.
          - A summary row for each day (Symbol = "ALL") showing total exposure and a flag
            indicating if the collective exposure meets the $10M requirement.
        """
        results = []
        for date in self.df["Date"].unique():
            day_results = []
            for symbol in ["SPY", "IWM", "QQQ"]:
                trade_result = strategy.execute(self.df, date, symbol)
                if trade_result:
                    day_results.append(trade_result)
                    results.append(trade_result)
            # Sum up exposure for the day across symbols.
            total_exposure = sum(r.get("Exposure Achieved", 0) for r in day_results)
            collective_exposure_met = total_exposure >= 10_000_000

            # Append a summary row for the day.
            results.append({
                "Date": date,
                "Symbol": "ALL",
                "Strategy": strategy.__class__.__name__,
                "Strike": None,
                "Contracts Bought": None,
                "Cost": None,
                "Exposure Achieved": total_exposure,
                "Exposure Met": collective_exposure_met,
                "Collective": True
            })
        return pd.DataFrame(results)
