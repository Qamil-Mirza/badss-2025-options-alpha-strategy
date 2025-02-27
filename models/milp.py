import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory

# ------------------------------
# HYPERPARAMETERS
# ------------------------------
WALL_TIME = 300  # 5-minute time limit
CONTRACT_SIZE = 100  # contract size for options
MIN_EXPOSURE = 10_000_000
SOLVER = 'cbc'
OUTPUT_FILE = "../results/optimized_trades.csv"


# TODO: Update spot move values dynamically each day, sample spot moves rfom some distribution??
spot_move = {
    'SPY': 1.03,
    'QQQ': 1.041,
    'IWM': 1.036,
}

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

# Define exposure parameters for a call option
def exposure_increment(undl, k, sym):
    move = spot_move[sym]
    return max(undl * move - k, 0) - max(undl - k, 0)

exposure = {o: exposure_increment(undl_price[o], strike[o], symbol[o]) for o in option_ids}

# ------------------------------
# Create mapping for next trading day
# ------------------------------
sorted_dates = sorted(unique_dates, key=lambda x: pd.to_datetime(x))
next_date = {d: sorted_dates[i+1] if i+1 < len(sorted_dates) else None 
             for i, d in enumerate(sorted_dates)}

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
# Objective Function
# ------------------------------
def objective_rule(model):
    return sum(model.buy[o] * ask_price[o] * CONTRACT_SIZE - model.sell[o] * bid_price[o] * CONTRACT_SIZE
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
    sell_qty = model.sell[o].value
    maturity_date = maturity_map[o]
    # Set trade quantities to 0 if the option has expired
    if pd.to_datetime(date_map[o]) >= pd.to_datetime(maturity_date):
        buy_qty = 0
        sell_qty = 0
    if (buy_qty is not None and buy_qty > 0) or (sell_qty is not None and sell_qty > 0):
        premium_cost = (buy_qty * CONTRACT_SIZE * ask_price[o]) - (sell_qty * CONTRACT_SIZE * bid_price[o])
        results_list.append({
            "Date": date_map[o],
            "Option_ID": o,
            "Maturity": maturity_map[o],
            "Buy": buy_qty,
            "Sell": sell_qty,
            "Ask_Price": ask_price[o],
            "Bid_Price": bid_price[o],
            "Premium_Cost": premium_cost
        })

daily_exposure = []
for d in model.DATES:
    exposure_sum = sum((model.buy[o].value - model.sell[o].value) * CONTRACT_SIZE * exposure[o]
                       for o in model.OPTIONS if date_map[o] == d)
    daily_exposure.append({"Date": d, "Exposure": exposure_sum})
exposure_df = pd.DataFrame(daily_exposure)
results_df = pd.DataFrame(results_list)
out_df = results_df.merge(exposure_df, on="Date", how="left")
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE}")
