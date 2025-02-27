import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "./results/pnlmin_optimized_trades.csv"
EXPOSURE_BASELINE = 1e7
COST_BASELINE = 2e5
PLOT_TITLE = "Cost and Exposure Over Time (pnlmin)"
SAVE_PATH = "./plots/pnlmin.png"

# Read and preprocess data
def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values("Date", inplace=True)

    # Aggregate daily premium cost (sum) and exposure (mean, since it's identical per day)
    daily = df.groupby("Date").agg(
        Premium_Cost=("Premium_Cost", "sum"),
        Exposure=("Exposure", "mean")
    ).reset_index()

    # Plot cost and exposure over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily["Date"], daily["Premium_Cost"], marker='o', label="Cost")
    ax.plot(daily["Date"], daily["Exposure"], marker='o', label="Exposure")
    ax.axhline(y=EXPOSURE_BASELINE, color='r', linestyle='--', label="10M Exposure baseline")
    ax.axhline(y=COST_BASELINE, color='black', linestyle='--', label="200k Cost baseline")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost / Exposure")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    plt.show()

if __name__ == "__main__":
    main()
    print("Results saved!")
