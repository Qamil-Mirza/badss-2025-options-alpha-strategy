import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory

# ------------------------------
# HYPERPARAMETERS
# ------------------------------
# TODO: Stop based on optimality gap
WALL_TIME = 10  # 5-minute time limit
CONTRACT_SIZE = 100  # contract size for options
MIN_EXPOSURE = 10_000_000
MODEL_NAME = "MILP_NO_SELL"
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

# Create mappings for option data
symbol = data.set_index('Option_ID')['Symbol'].to_dict()
ask_price = data.set_index('Option_ID')['Ask_Price'].to_dict()
bid_price = data.set_index('Option_ID')['Bid_Price'].to_dict()
ask_size  = data.set_index('Option_ID')['Ask_Size'].to_dict()
bid_size  = data.set_index('Option_ID')['Bid_Size'].to_dict()
undl_price= data.set_index('Option_ID')['Undl_Price'].to_dict()
strike    = data.set_index('Option_ID')['Strike'].to_dict()
date_map  = data.set_index('Option_ID')['Date'].to_dict()
maturity_map = data.set_index('Option_ID')['Maturity'].to_dict()

unique_dates = data['Date'].unique().tolist()
unique_symbols = data['Symbol'].unique().tolist()

# ------------------------------
# Generate dynamic spot moves for each date and symbol
# ------------------------------
np.random.seed(42)

# Create base parameters for each symbol's moves
spot_move_params = {
    'SPY': {'mean': 1.03, 'std': 0.01},  # Mean 3% up, std 1%
    'QQQ': {'mean': 1.041, 'std': 0.015},  # Mean 4.1% up, std 1.5%
    'IWM': {'mean': 1.036, 'std': 0.012},  # Mean 3.6% up, std 1.2%
}

# Generate daily spot moves for each symbol
daily_spot_moves = {}
for date in unique_dates:
    daily_spot_moves[date] = {}
    for sym in unique_symbols:
        if sym in spot_move_params:
            # Sample from normal distribution based on symbol parameters
            params = spot_move_params[sym]
            move = np.random.normal(params['mean'], params['std'])
            # TODO: Check what happens if the move is negative
            # Ensure no negative moves (optional, remove if you want to allow downside)
            # move = max(move, 1.0)
            daily_spot_moves[date][sym] = move

# ------------------------------
# Create mapping for next trading day
# ------------------------------
sorted_dates = sorted(unique_dates, key=lambda x: pd.to_datetime(x))
next_date = {d: sorted_dates[i+1] if i+1 < len(sorted_dates) else None 
             for i, d in enumerate(sorted_dates)}

# ------------------------------
# Define exposure parameters for a call option with dynamic spot moves
# ------------------------------
def exposure_increment(undl, k, sym, date):
    # Get the spot move for this symbol on this date
    move = daily_spot_moves[date][sym]
    return max(undl * move - k, 0) - max(undl - k, 0)

# Create exposure dictionary for each option and its trade date
exposure = {o: exposure_increment(undl_price[o], strike[o], symbol[o], date_map[o]) for o in option_ids}

# ------------------------------
# Pyomo Model Setup
# ------------------------------
model = ConcreteModel()
model.OPTIONS = Set(initialize=option_ids)
model.DATES = Set(initialize=unique_dates)

model.buy = Var(model.OPTIONS, domain=NonNegativeIntegers)

# ------------------------------
# Constraints
# ------------------------------

# Bound constraints on buys
def buy_bound_rule(model, o):
    return model.buy[o] <= ask_size[o]
model.buy_bound = Constraint(model.OPTIONS, rule=buy_bound_rule)

# Adjust expired option constraints:
# If the trade date is on or after the maturity date, do not allow trading.
def expired_option_rule_buy(model, o):
    if pd.to_datetime(date_map[o]) >= pd.to_datetime(maturity_map[o]):
        return model.buy[o] == 0
    else:
        return Constraint.Skip
model.expired_buy = Constraint(model.OPTIONS, rule=expired_option_rule_buy)

# Exposure constraint
def exposure_rule(model, d):
    # Only enforce the exposure constraint if there is a "next day"
    nd = next_date[d]
    if nd is None:
         return Constraint.Skip
    expr = sum(
        (model.buy[o]) * CONTRACT_SIZE * exposure[o]
        for o in model.OPTIONS 
        if (date_map[o] == d and pd.to_datetime(maturity_map[o]) >= pd.to_datetime(nd))
    )
    return expr >= MIN_EXPOSURE
model.exposure_constraint = Constraint(model.DATES, rule=exposure_rule)

# ------------------------------
# Objective Function
# ------------------------------
def objective_rule(model):
    return sum(model.buy[o] * ask_price[o] * CONTRACT_SIZE
               for o in model.OPTIONS)
model.obj = Objective(rule=objective_rule, sense=minimize)

# ------------------------------
# Solve the Model
# ------------------------------
solver = SolverFactory(SOLVER)
solver.options['sec'] = WALL_TIME
results = solver.solve(model, tee=True)

# ------------------------------
# Save Results
# ------------------------------
results_list = []
for o in model.OPTIONS:
    buy_qty = model.buy[o].value
    maturity_date = maturity_map[o]
    # Set trade quantities to 0 if the option has expired
    if pd.to_datetime(date_map[o]) >= pd.to_datetime(maturity_date):
        buy_qty = 0
    if (buy_qty is not None and buy_qty > 0):
        premium_cost = buy_qty * CONTRACT_SIZE * ask_price[o]
        results_list.append({
            "Date": date_map[o],
            "Option_ID": o,
            "Symbol": symbol[o],
            "Maturity": maturity_map[o],
            "Buy": buy_qty,
            "Ask_Price": ask_price[o],
            "Bid_Price": bid_price[o],
            "Premium_Cost": premium_cost,
        })

daily_exposure = []
for d in model.DATES:
    exposure_sum = sum((model.buy[o].value) * CONTRACT_SIZE * exposure[o]
                       for o in model.OPTIONS if date_map[o] == d)
    
    # Get the spot moves for this date to include in output
    spot_moves_for_date = {sym: daily_spot_moves[d][sym] for sym in unique_symbols if sym in daily_spot_moves[d]}
    
    daily_exposure.append({
        "Date": d, 
        "Exposure": exposure_sum
    })

exposure_df = pd.DataFrame(daily_exposure)
results_df = pd.DataFrame(results_list)
out_df = results_df.merge(exposure_df, on="Date", how="left")
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE}")

# Save spot moves to a separate file for analysis
spot_moves_df = pd.DataFrame([
    {"Date": date, "Symbol": sym, "Spot_Move": daily_spot_moves[date][sym]}
    for date in unique_dates
    for sym in unique_symbols
    if sym in daily_spot_moves[date]
])
spot_moves_df.to_csv(SPOT_MOVES_DAILY_FILE, index=False)
print(f"Daily spot moves saved to {SPOT_MOVES_DAILY_FILE}")
