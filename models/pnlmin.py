import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory

# ------------------------------
# HYPERPARAMETERS
# ------------------------------
WALL_TIME = 10  # 5-minute time limit
CONTRACT_SIZE = 100  # contract size for options
MIN_EXPOSURE = 10_000_000
SOLVER = 'cbc'
OUTPUT_FILE = "../results/pnlmin_optimized_trades.csv"

# ------------------------------
# Data Preparation
# ------------------------------
data = pd.read_csv("../data/BADSS training data.csv")
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
# Set random seed for reproducibility (optional)
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
            # Ensure no negative moves (optional, remove if you want to allow downside)
            move = max(move, 1.0)
            daily_spot_moves[date][sym] = move

# ------------------------------
# Create mapping for next trading day
# ------------------------------
sorted_dates = sorted(unique_dates, key=lambda x: pd.to_datetime(x))
next_date = {d: sorted_dates[i+1] if i+1 < len(sorted_dates) else None 
             for i, d in enumerate(sorted_dates)}

# Find for each option ID the corresponding option for the next date (if exists)
def find_next_day_option(option_id, next_d):
    """Find the option with the same symbol, maturity, and strike on the next trading day"""
    if next_d is None:
        return None
        
    parts = option_id.split("_")
    sym = parts[0]
    maturity = parts[1]
    strike_val = parts[2]
    
    # Search for a matching option
    next_option = [o for o in option_ids 
                  if symbol[o] == sym 
                  and maturity_map[o] == maturity 
                  and strike[o] == float(strike_val)
                  and date_map[o] == next_d]
    
    return next_option[0] if next_option else None

# Create dictionary mapping option IDs to their next day equivalents
next_day_option = {o: find_next_day_option(o, next_date.get(date_map[o])) for o in option_ids}

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
model.sell = Var(model.OPTIONS, domain=NonNegativeIntegers)

# ------------------------------
# Constraints
# ------------------------------

# Bound constraints on buys and sells
def buy_bound_rule(model, o):
    return model.buy[o] <= ask_size[o]
model.buy_bound = Constraint(model.OPTIONS, rule=buy_bound_rule)

def sell_bound_rule(model, o):
    return model.sell[o] <= bid_size[o]
model.sell_bound = Constraint(model.OPTIONS, rule=sell_bound_rule)

# Ensure you cannot sell more than you buy
def net_position_rule(model, o):
    return model.sell[o] <= model.buy[o]
model.net_position = Constraint(model.OPTIONS, rule=net_position_rule)

# Adjust expired option constraints:
# If the trade date is on or after the maturity date, do not allow trading.
def expired_option_rule_buy(model, o):
    if pd.to_datetime(date_map[o]) >= pd.to_datetime(maturity_map[o]):
        return model.buy[o] == 0
    else:
        return Constraint.Skip
model.expired_buy = Constraint(model.OPTIONS, rule=expired_option_rule_buy)

def expired_option_rule_sell(model, o):
    if pd.to_datetime(date_map[o]) >= pd.to_datetime(maturity_map[o]):
        return model.sell[o] == 0
    else:
        return Constraint.Skip
model.expired_sell = Constraint(model.OPTIONS, rule=expired_option_rule_sell)

# Exposure constraint
def exposure_rule(model, d):
    # Only enforce the exposure constraint if there is a "next day"
    nd = next_date[d]
    if nd is None:
         return Constraint.Skip
    expr = sum(
        (model.buy[o] - model.sell[o]) * CONTRACT_SIZE * exposure[o]
        for o in model.OPTIONS 
        if (date_map[o] == d and pd.to_datetime(maturity_map[o]) >= pd.to_datetime(nd))
    )
    return expr >= MIN_EXPOSURE
model.exposure_constraint = Constraint(model.DATES, rule=exposure_rule)

# ------------------------------
# Objective Function - Minimize Premium P&L
# ------------------------------
def objective_rule(model):
    p_and_l = 0
    
    # For each option
    for o in model.OPTIONS:
        d = date_map[o]
        next_d = next_date.get(d)
        if next_d is None:
            continue
            
        # Get the next day's equivalent option (if it exists)
        next_o = next_day_option.get(o)
        
        if next_o is not None:
            # Premium paid today
            premium_today = model.buy[o] * ask_price[o] * CONTRACT_SIZE - model.sell[o] * bid_price[o] * CONTRACT_SIZE
            
            # Calculate the P&L when liquidating the position on the next day
            # This is a simplification - in reality you might want a more complex valuation model
            premium_next_day = model.buy[o] * bid_price[next_o] * CONTRACT_SIZE - model.sell[o] * ask_price[next_o] * CONTRACT_SIZE
            
            # Add to total P&L (negative because we want to minimize losses)
            p_and_l += (premium_today - premium_next_day)
    
    return p_and_l
    
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
    sell_qty = model.sell[o].value
    maturity_date = maturity_map[o]
    # Set trade quantities to 0 if the option has expired
    if pd.to_datetime(date_map[o]) >= pd.to_datetime(maturity_date):
        buy_qty = 0
        sell_qty = 0
    if (buy_qty is not None and buy_qty > 0) or (sell_qty is not None and sell_qty > 0):
        premium_cost = (buy_qty * CONTRACT_SIZE * ask_price[o]) - (sell_qty * CONTRACT_SIZE * bid_price[o])
        
        # Calculate next day values for reporting
        next_d = next_date.get(date_map[o])
        next_o = next_day_option.get(o)
        next_day_premium = 0
        estimated_pnl = None
        
        if next_o is not None:
            next_day_premium = (buy_qty * CONTRACT_SIZE * bid_price[next_o]) - (sell_qty * CONTRACT_SIZE * ask_price[next_o])
            estimated_pnl = next_day_premium - premium_cost
        
        results_list.append({
            "Date": date_map[o],
            "Option_ID": o,
            "Symbol": symbol[o],
            "Maturity": maturity_map[o],
            "Strike": strike[o],
            "Buy": buy_qty,
            "Sell": sell_qty,
            "Ask_Price": ask_price[o],
            "Bid_Price": bid_price[o],
            "Premium_Cost": premium_cost,
            "Next_Day_Premium_Value": next_day_premium,
            "Estimated_PnL": estimated_pnl,
            "Spot_Move": daily_spot_moves[date_map[o]][symbol[o]]
        })

daily_exposure = []
total_pnl = 0

for d in model.DATES:
    exposure_sum = sum((model.buy[o].value - model.sell[o].value) * CONTRACT_SIZE * exposure[o]
                       for o in model.OPTIONS if date_map[o] == d)
    
    # Calculate daily P&L
    daily_pnl = 0
    for o in [o for o in model.OPTIONS if date_map[o] == d]:
        buy_qty = model.buy[o].value or 0
        sell_qty = model.sell[o].value or 0
        premium_cost = (buy_qty * CONTRACT_SIZE * ask_price[o]) - (sell_qty * CONTRACT_SIZE * bid_price[o])
        
        next_d = next_date.get(d)
        next_o = next_day_option.get(o)
        
        if next_o is not None:
            next_day_premium = (buy_qty * CONTRACT_SIZE * bid_price[next_o]) - (sell_qty * CONTRACT_SIZE * ask_price[next_o])
            daily_pnl += (next_day_premium - premium_cost)
    
    total_pnl += daily_pnl
    
    # Get the spot moves for this date to include in output
    spot_moves_for_date = {sym: daily_spot_moves[d][sym] for sym in unique_symbols if sym in daily_spot_moves[d]}
    
    daily_exposure.append({
        "Date": d, 
        "Exposure": exposure_sum,
        "Daily_PnL": daily_pnl,
        "Cumulative_PnL": total_pnl,
        **{f"{sym}_Spot_Move": spot_moves_for_date.get(sym, None) for sym in unique_symbols}
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
spot_moves_df.to_csv("../results/pnlmin_daily_spot_moves.csv", index=False)
print(f"Daily spot moves saved to ../results/daily_spot_moves.csv")

# Print summary statistics
print(f"\nOptimization Summary:")
print(f"Total P&L: ${total_pnl:.2f}")
print(f"Average Daily P&L: ${total_pnl/len(unique_dates):.2f}")
print(f"Minimum Daily Exposure: ${exposure_df['Exposure'].min():.2f}")
print(f"Maximum Daily Exposure: ${exposure_df['Exposure'].max():.2f}")
