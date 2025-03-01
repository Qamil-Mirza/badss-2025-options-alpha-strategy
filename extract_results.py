import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = "MAIN_MODEL_TEST"
INCLUDE_SELL = False
MARKET_DATA_PATH = "./data/BADSS test data.csv"
OPTIMIZED_TRADES_PATH = f"./results/{MODEL_NAME}_optimized_trades.csv"
EXPOSURE_BASELINE = 1e7
COST_BASELINE = 2e5
COST_EXPOSURE_SAVE_PATH = f"./plots/cost_and_exposure_over_time_{MODEL_NAME}.png"
CUMULATIVE_COST_SAVE_PATH = f"./plots/cumulative_cost_over_time_{MODEL_NAME}.png"


def plot_daily_cost_and_total_exposure(daily_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_df["Date"], daily_df["Premium_Cost"], marker='o', label="Cost")
    ax.plot(daily_df["Date"], daily_df["Exposure"], marker='o', label="Exposure")
    ax.axhline(y=1e7, color='r', linestyle='--', label="10M Exposure baseline")
    ax.axhline(y=2e5, color='black', linestyle='--', label="200k Cost baseline")
    ax.set_title(f"Daily Total Exposure and Cost ({MODEL_NAME})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost / Total Exposure ($)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(COST_EXPOSURE_SAVE_PATH)

def plot_cumulative_cost(daily_df):
    cumsum = daily_df["Premium_Cost"].cumsum()
    
    total_end_cost = np.round(cumsum.iloc[-1], 2)

    plt.figure(figsize=(10, 6))
    plt.plot(daily_df["Date"], cumsum, marker='o', label="Cumulative Cost")

    # Highlight the last point
    plt.scatter(daily_df["Date"].iloc[-1], cumsum.iloc[-1], color='red', zorder=3)
    plt.annotate(f"{total_end_cost}",
                 xy=(daily_df["Date"].iloc[-1], cumsum.iloc[-1]),
                 xytext=(-30, -30), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", lw=1),
                 fontsize=12, color='red')

    # Chart formatting
    plt.title(f"Cumulative Cost Over Time ({MODEL_NAME})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Cost ($)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig(CUMULATIVE_COST_SAVE_PATH)

def create_pnl_df(market_df, optimized_trades_df):
    # Prep for merge
    if not INCLUDE_SELL:
        temp_df = optimized_trades_df[["Date", "Option_ID", "Buy", "Premium_Cost"]]
    else:
        temp_df = optimized_trades_df[["Date", "Option_ID", "Buy", "Sell", "Premium_Cost"]]

    # Merge on Option_ID and Date
    pnl_df = market_df.merge(temp_df, on=["Option_ID", "Date"], how="left", suffixes=("", "_y"))

    # Fill NaN values with 0
    pnl_df.fillna(0, inplace=True)

    pnl_df = pnl_df[pnl_df["Option_ID"].isin(optimized_trades_df["Option_ID"])]

    pnl_df["Date"] = pd.to_datetime(pnl_df["Date"])

    pnl_list = []
    for _, row in pnl_df.iterrows():
        # Check if this is a buy transaction
        if row["Buy"] > 0:
            # Calculate next day date
            next_day = row["Date"] + pd.DateOffset(1)

            # Find the next day's bid price for the same Option_ID
            next_day_row = pnl_df[(pnl_df["Date"] == next_day) & (pnl_df["Option_ID"] == row["Option_ID"])]

            if not next_day_row.empty:
                # Extract bid price for next day
                bid_price_day1 = next_day_row["Bid Price"].values[0]
                ask_price_day0 = row["Ask Price"]  # Purchase price

                # Calculate PnL
                pnl = row["Buy"] * 100 * (bid_price_day1 - ask_price_day0)
            else:
                # If next day's price is missing, assume full loss (Premium Cost)
                pnl = -row["Premium_Cost"]

    # TODO: Consider the case for sell transactions
        else:
            # If not a buy transaction, set PnL to 0
            pnl = 0

        # Append the result to the list
        pnl_list.append(pnl)

    # Add the PnL column to the DataFrame
    pnl_df["PnL"] = pnl_list

    daily_pnl = pnl_df.groupby("Date")["PnL"].sum().reset_index()

    return daily_pnl

def plot_cumulative_pnl(pnl_df):
    daily_pnl = pnl_df.copy()
    daily_pnl["Cumulative_PnL"] = daily_pnl["PnL"].cumsum()

    total_pnl = np.round(daily_pnl["Cumulative_PnL"].iloc[-1], 2)

    plt.figure(figsize=(10, 6))
    plt.plot(daily_pnl["Date"], daily_pnl["Cumulative_PnL"], marker='o', label="Cumulative PnL")
    plt.scatter(daily_pnl["Date"].iloc[-1], total_pnl, color='red', s=100, label="Final PnL", edgecolors='black', zorder=3)
    plt.annotate(f"{total_pnl}",
             (daily_pnl["Date"].iloc[-1], total_pnl),
             textcoords="offset points",
             xytext=(-20, 30),
             ha='center',
             fontsize=12,
             color='red',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white"))
    
    plt.title(f"Cumulative PnL Over Time ({MODEL_NAME})")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative PnL ($)", fontsize=12)

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"./plots/cumulative_pnl_{MODEL_NAME}.png")


if __name__ == "__main__":
    market_df = pd.read_csv(MARKET_DATA_PATH)
    market_df['Option_ID'] = market_df["Symbol"] + "_" + market_df["Maturity"] + "_" + market_df["Strike"].astype(str)
    market_df["Date"] = pd.to_datetime(market_df["Date"])

    # clean up optimized trades
    optimized_trades_df = pd.read_csv(OPTIMIZED_TRADES_PATH)
    optimized_trades_df.columns = optimized_trades_df.columns.str.strip()
    optimized_trades_df["Date"] = pd.to_datetime(optimized_trades_df["Date"])
    optimized_trades_df.sort_values(by="Date", inplace=True)

    daily_df = (
        optimized_trades_df.groupby("Date")
        .agg(Premium_Cost=("Premium_Cost", "sum"), Exposure=("Exposure", "mean"))
        .reset_index()
    )

    plot_daily_cost_and_total_exposure(daily_df=daily_df)
    plot_cumulative_cost(daily_df=daily_df)

    df_with_pnl = create_pnl_df(market_df=market_df, optimized_trades_df=optimized_trades_df)
    # df_with_pnl.to_csv(f"./results/pnl_df_{MODEL_NAME}.csv", index=False)
    
    plot_cumulative_pnl(pnl_df=df_with_pnl)
    print("Results extracted and plots saved successfully!")
