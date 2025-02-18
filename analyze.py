import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


result_df = pd.read_csv("./results/ATMStrategy_simulation_results.csv")

# Calculate the total exposure for each day
daily_exposure = result_df.groupby("Date")["Exposure Achieved"].sum()

# Plot the daily exposure
plt.figure(figsize=(10, 6))
plt.plot(daily_exposure, marker="o")
plt.title("Daily Exposure Achieved by ATM Strategy")
plt.xlabel("Date")
plt.ylabel("Exposure ($)")
plt.hlines(10_000_000, daily_exposure.index[0], daily_exposure.index[-1], linestyles="--", colors="r")
plt.grid(True)
plt.xticks(rotation=45)
plt.savefig("./plots/ATM_strategy_daily_exposure.png")
plt.show()