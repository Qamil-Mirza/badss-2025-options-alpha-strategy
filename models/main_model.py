import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory

# ------------------------------
# HYPERPARAMETERS
# ------------------------------
WALL_TIME = 10  # seconds time limit
CONTRACT_SIZE = 100  # contract size for options
MIN_EXPOSURE = 10_000_000
MODEL_NAME = "MAIN_MODEL_TRAIN"
SOLVER = 'cbc'
MARKET_DATA_PATH = "../data/BADSS training data.csv"
OUTPUT_FILE = f"../results/{MODEL_NAME}_optimized_trades.csv"
SPOT_MOVES_DAILY_FILE = f"../results/{MODEL_NAME}_daily_spot_moves.csv"

# ------------------------------
# Data Preparation
# ------------------------------
data = pd.read_csv(MARKET_DATA_PATH)
data.columns = data.columns.str.strip().str.replace(" ", "_")
data['Option_ID'] = data['Symbol'] + "_" + data['Maturity'] + "_" + data['Strike'].astype(str)
option_ids = data['Option_ID'].tolist()

# Create mappings for option data (for the day the option is traded)
symbol = data.set_index('Option_ID')['Symbol'].to_dict()
ask_price = data.set_index('Option_ID')['Ask_Price'].to_dict()
bid_price = data.set_index('Option_ID')['Bid_Price'].to_dict()
ask_size  = data.set_index('Option_ID')['Ask_Size'].to_dict()
bid_size  = data.set_index('Option_ID')['Bid_Size'].to_dict()
undl_price= data.set_index('Option_ID')['Undl_Price'].to_dict()
strike    = data.set_index('Option_ID')['Strike'].to_dict()
date_map  = data.set_index('Option_ID')['Date'].to_dict()
maturity_map = data.set_index('Option_ID')['Maturity'].to_dict()

# ------------------------------
# Unique Dates and Symbols
# ------------------------------
unique_dates = sorted(data['Date'].unique(), key=lambda x: pd.to_datetime(x))
unique_symbols = data['Symbol'].unique().tolist()

# ------------------------------
# Generate dynamic spot moves for each date and symbol
# ------------------------------
np.random.seed(42)
spot_move_params = {
    'SPY': {'mean': 1.03, 'std': 0.01},
    'QQQ': {'mean': 1.041, 'std': 0.015},
    'IWM': {'mean': 1.036, 'std': 0.012},
}
daily_spot_moves = {}
for date in unique_dates:
    daily_spot_moves[date] = {}
    for sym in unique_symbols:
        if sym in spot_move_params:
            params = spot_move_params[sym]
            move = np.random.normal(params['mean'], params['std'])
            daily_spot_moves[date][sym] = move

# ------------------------------
# Create mapping for next trading day
# ------------------------------
next_date = {d: unique_dates[i+1] if i+1 < len(unique_dates) else None 
             for i, d in enumerate(unique_dates)}

# ------------------------------
# Define Exposure Functions
# ------------------------------
# For new trades: compute the exposure increment based on today's underlying price.
def exposure_increment(undl, k, sym, date):
    move = daily_spot_moves[date][sym]
    return max(undl * move - k, 0) - max(undl - k, 0)

# For carried positions, we assume the exposure was "locked in" on the purchase day.
# (In a more dynamic model, you could re-calc exposure using today's market data.)
def carried_exposure(trade):
    # trade already stores its base_exposure (computed on the day of purchase)
    return trade['quantity'] * CONTRACT_SIZE * trade['base_exposure']

# ------------------------------
# Initialize Portfolio to hold carried-over positions
# ------------------------------
portfolio = []  # list of dicts; each dict has keys: Option_ID, quantity, base_exposure, Maturity, Symbol

# ------------------------------
# Daily Optimization Loop (Adjusted to include carried exposure)
# ------------------------------
all_trades = []        # To store trade results from each day
daily_exposure_list = []  # To store total exposure (carried + new) for each day

for d in unique_dates:
    nd = next_date[d]
    if nd is None:
        continue  # Skip last day since we need a next day for exposure calculation

    # Remove expired positions from the portfolio (if maturity date < current day d)
    portfolio = [trade for trade in portfolio if pd.to_datetime(trade['Maturity']) >= pd.to_datetime(d)]
    
    # Compute carried exposure from previous trades
    carried_exposure_value = sum(carried_exposure(trade) for trade in portfolio)
    
    # ------------------------------
    # Filter options available for trading on day d
    # ------------------------------
    options_today = [
        o for o in option_ids 
        if date_map[o] == d and pd.to_datetime(maturity_map[o]) >= pd.to_datetime(nd)
    ]
    if not options_today:
        # Record the day's exposure (only carried positions)
        daily_exposure_list.append({"Date": d, "Exposure": carried_exposure_value})
        continue

    # Compute exposure for each new trade option on day d (using today's data)
    exposure_today = {
        o: exposure_increment(undl_price[o], strike[o], symbol[o], d)
        for o in options_today
    }

    # ------------------------------
    # Build the Pyomo Model for Day d (New Trades)
    # ------------------------------
    model_day = ConcreteModel()
    model_day.OPTIONS = Set(initialize=options_today)
    model_day.buy = Var(model_day.OPTIONS, domain=NonNegativeIntegers)

    # Bound constraint: cannot buy more than available ask size
    def buy_bound_rule(model, o):
        return model.buy[o] <= ask_size[o]
    model_day.buy_bound = Constraint(model_day.OPTIONS, rule=buy_bound_rule)

    # Exposure constraint: carried exposure plus new exposure must meet MIN_EXPOSURE
    def exposure_rule(model):
        new_exposure = sum(model.buy[o] * CONTRACT_SIZE * exposure_today[o] for o in model.OPTIONS)
        return carried_exposure_value + new_exposure >= MIN_EXPOSURE
    model_day.exposure_constraint = Constraint(rule=exposure_rule)

    # Objective: minimize total premium cost for new trades
    def objective_rule(model):
        return sum(model.buy[o] * ask_price[o] * CONTRACT_SIZE for o in model.OPTIONS)
    model_day.obj = Objective(rule=objective_rule, sense=minimize)

    # ------------------------------
    # Solve the Model for Day d
    # ------------------------------
    solver = SolverFactory(SOLVER)
    solver.options['sec'] = WALL_TIME
    result = solver.solve(model_day, tee=True)

    # ------------------------------
    # Save Results for Day d (New Trades)
    # ------------------------------
    trades_today = []
    for o in model_day.OPTIONS:
        buy_qty = model_day.buy[o].value
        if buy_qty is None or buy_qty == 0:
            continue
        premium_cost = buy_qty * CONTRACT_SIZE * ask_price[o]
        trade_info = {
            "Date": d,
            "Option_ID": o,
            "Symbol": symbol[o],
            "Maturity": maturity_map[o],
            "Buy": buy_qty,
            "Ask_Price": ask_price[o],
            "Bid_Price": bid_price[o],
            "Premium_Cost": premium_cost,
            # Store base_exposure for carried positions (from today's calculation)
            "base_exposure": exposure_today[o]
        }
        trades_today.append(trade_info)
        # Add new trade to portfolio (to be carried over to future days)
        portfolio.append({
            "Option_ID": o,
            "quantity": buy_qty,
            "base_exposure": exposure_today[o],
            "Maturity": maturity_map[o],
            "Symbol": symbol[o]
        })
    all_trades.extend(trades_today)
    
    # Total exposure for day d = carried exposure + new trades' exposure (using today's exposures)
    new_exposure_value = sum(model_day.buy[o].value * CONTRACT_SIZE * exposure_today[o] for o in model_day.OPTIONS)
    total_exposure = carried_exposure_value + new_exposure_value
    daily_exposure_list.append({"Date": d, "Exposure": total_exposure})

# ------------------------------
# Save and Output Results
# ------------------------------
results_df = pd.DataFrame(all_trades)
exposure_df = pd.DataFrame(daily_exposure_list)
out_df = results_df.merge(exposure_df, on="Date", how="left")
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE}")

# Save spot moves for analysis
spot_moves_data = [
    {"Date": date, "Symbol": sym, "Spot_Move": daily_spot_moves[date][sym]}
    for date in unique_dates 
    for sym in unique_symbols 
    if sym in daily_spot_moves[date]
]
spot_moves_df = pd.DataFrame(spot_moves_data)
spot_moves_df.to_csv(SPOT_MOVES_DAILY_FILE, index=False)
print(f"Daily spot moves saved to {SPOT_MOVES_DAILY_FILE}")
