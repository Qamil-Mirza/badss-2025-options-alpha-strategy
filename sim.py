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