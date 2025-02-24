import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory

# ------------------------------
# Data Preparation
# ------------------------------
# Load option market data
data = pd.read_csv("./data/BADSS training data.csv")

# Clean up column names
data.columns = data.columns.str.strip().str.replace(" ", "_")

# Create a unique option ID for each row
data['Option_ID'] = data['Symbol'] + "_" + data['Maturity'] + "_" + data['Strike'].astype(str)

# Create dictionaries for parameters keyed by Option_ID
option_ids = data['Option_ID'].tolist()

ask_price = data.set_index('Option_ID')['Ask_Price'].to_dict()
bid_price = data.set_index('Option_ID')['Bid_Price'].to_dict()
ask_size  = data.set_index('Option_ID')['Ask_Size'].to_dict()
bid_size  = data.set_index('Option_ID')['Bid_Size'].to_dict()
undl_price= data.set_index('Option_ID')['Undl_Price'].to_dict()
strike    = data.set_index('Option_ID')['Strike'].to_dict()
# Also create a mapping from Option_ID to Date for the exposure constraint
date_map  = data.set_index('Option_ID')['Date'].to_dict()

# Get unique dates for the daily exposure constraint
unique_dates = data['Date'].unique().tolist()

# Define exposure parameters (for a call option here)
def exposure_increment(undl, k):
    # Exposure increment = max(undl * 1.03 - k, 0) - max(undl - k, 0)
    return max(undl * 1.03 - k, 0) - max(undl - k, 0)

exposure = {o: exposure_increment(undl_price[o], strike[o]) for o in option_ids}

# ------------------------------
# Pyomo Model Setup
# ------------------------------
model = ConcreteModel()

# Sets
model.OPTIONS = Set(initialize=option_ids)
model.DATES = Set(initialize=unique_dates)

# Parameters (we've created dictionaries already)
# (You can also add parameters to the model if needed)

# Decision Variables: Number of contracts to Buy and Sell (non-negative integers)
model.buy = Var(model.OPTIONS, domain=NonNegativeIntegers)
model.sell = Var(model.OPTIONS, domain=NonNegativeIntegers)

# ------------------------------
# Constraints
# ------------------------------

# Upper bound constraints on buys and sells:
def buy_bound_rule(model, o):
    return model.buy[o] <= ask_size[o]
model.buy_bound = Constraint(model.OPTIONS, rule=buy_bound_rule)

def sell_bound_rule(model, o):
    return model.sell[o] <= bid_size[o]
model.sell_bound = Constraint(model.OPTIONS, rule=sell_bound_rule)

# Constraint to prevent selling more than you have purchased (net position >= 0)
def net_position_rule(model, o):
    return model.sell[o] <= model.buy[o]
model.net_position = Constraint(model.OPTIONS, rule=net_position_rule)

# Daily minimum exposure constraint:
min_exposure = 10_000_000  
# For each date, sum the exposure contributions from each option traded that day.
def exposure_rule(model, d):
    expr = sum((model.buy[o] - model.sell[o]) * 100 * exposure[o]
               for o in model.OPTIONS if date_map[o] == d)
    return expr >= min_exposure
model.exposure_constraint = Constraint(model.DATES, rule=exposure_rule)

# ------------------------------
# Objective Function
# ------------------------------
def objective_rule(model):
    # Total cost: cost for buying (pay ask price) minus proceeds from selling (receive bid price)
    return sum(model.buy[o] * ask_price[o] * 100 - model.sell[o] * bid_price[o] * 100
               for o in model.OPTIONS)
model.obj = Objective(rule=objective_rule, sense=minimize)

# ------------------------------
# Solve the Model
# ------------------------------
solver = SolverFactory('cbc')
solver.options['sec'] = 10  # Time limit in seconds
results = solver.solve(model, tee=True)

# ------------------------------
# Save Results
# ------------------------------
results_list = []
for o in model.OPTIONS:
    buy_qty = model.buy[o].value
    sell_qty = model.sell[o].value
    # Only include options with a nonzero trade
    if (buy_qty is not None and buy_qty > 0) or (sell_qty is not None and sell_qty > 0):
        # Calculate premium cost
        premium_cost = (buy_qty * 100 * ask_price[o]) - (sell_qty * 100 * bid_price[o])

        results_list.append({
            "Date": date_map[o],
            "Option_ID": o,
            "Buy": buy_qty,
            "Sell": sell_qty,
            "Ask_Price": ask_price[o],
            "Bid_Price": bid_price[o],
            "Premium_Cost": premium_cost
        })

# calculate daily exposure
daily_exposure = []
for d in model.DATES:
    exposure_sum = sum((model.buy[o].value - model.sell[o].value) * 100 * exposure[o]
                       for o in model.OPTIONS if date_map[o] == d)
    daily_exposure.append({"Date": d, "Exposure": exposure_sum})

exposure_df = pd.DataFrame(daily_exposure)
results_df = pd.DataFrame(results_list)

# Merge the exposure data with the results
out_df = results_df.merge(exposure_df, on="Date", how="left")
out_df.to_csv("optimized_trades.csv", index=False)
print("Results saved to optimized_trades.csv")
