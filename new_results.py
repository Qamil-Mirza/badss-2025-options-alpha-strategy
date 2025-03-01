import pandas as pd
import matplotlib.pyplot as plt

MODEL_NAME = "INTERN_TEST"

# Replace with the actual path to your daily summary CSV file
SUMMARY_FILE = f"./results/{MODEL_NAME}_daily_summary.csv"


# Read the daily summary CSV file; parse Date column as datetime
summary_df = pd.read_csv(SUMMARY_FILE, parse_dates=['Date'])
summary_df.sort_values("Date", inplace=True)

# ------------------------------
# Plot 1: Daily Total Exposure and Daily Premium Cost
# ------------------------------
plt.figure(figsize=(12, 6))
plt.plot(summary_df['Date'], summary_df['Portfolio_Exposure'], marker='o', label='Total Exposure')
plt.plot(summary_df['Date'], summary_df['New_Premium_Cost'], marker='o', label='Daily Premium Cost')
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.title(f'Daily Total Exposure and Daily Premium Cost ({MODEL_NAME})')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"./plots/daily_exposure_and_premium_cost_({MODEL_NAME}).png")
plt.show()

# ------------------------------
# Compute Cumulative PnL Adjustment
# ------------------------------
# Assuming "Cumulative PnL Adjustment" is defined as the cumulative sum of Daily_PnL.
summary_df['Cumulative_PnL'] = summary_df['Daily_PnL'].cumsum()

# ------------------------------
# Plot 2: Cumulative Premium Cost and Cumulative PnL Adjustment with Emphasis on Final Values
# ------------------------------
plt.figure(figsize=(12, 6))
plt.plot(summary_df['Date'], summary_df['Cumulative_Premium_Cost'], marker='o', label='Cumulative Premium Cost')
plt.plot(summary_df['Date'], summary_df['Cumulative_PnL'], marker='o', label='Cumulative PnL Adjustment')
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.title(f'Cumulative Premium Cost and Cumulative PnL Adjustment ({MODEL_NAME})')
plt.legend()
plt.xticks(rotation=45)

# Annotate final cumulative premium cost and cumulative PnL
final_date = summary_df['Date'].iloc[-1]
final_cum_premium = summary_df['Cumulative_Premium_Cost'].iloc[-1]
final_cum_pnl = summary_df['Cumulative_PnL'].iloc[-1]

plt.annotate(f'Final Premium: ${final_cum_premium:,.0f}', 
             xy=(final_date, final_cum_premium), 
             xytext=(final_date, final_cum_premium * 1.05),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10, color='red')

plt.annotate(f'Final PnL: ${final_cum_pnl:,.0f}', 
             xy=(final_date, final_cum_pnl), 
             xytext=(final_date, final_cum_pnl * 1.05),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10, color='green')

plt.tight_layout()
plt.savefig(f"./plots/cumulative_premium_cost_and_pnl_adjustment_{MODEL_NAME}.png")
plt.show()
