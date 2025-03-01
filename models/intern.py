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
MODEL_NAME = "INTERN_TEST"
SOLVER = 'cbc'
MARKET_DATA_PATH = "../data/BADSS test data.csv"
OUTPUT_FILE = f"../results/{MODEL_NAME}_optimized_trades.csv"
SUMMARY_FILE = f"../results/{MODEL_NAME}_daily_summary.csv"
SPOT_MOVES_DAILY_FILE = f"../results/{MODEL_NAME}_daily_spot_moves.csv"

# ------------------------------
# Data Preparation
# ------------------------------
data = pd.read_csv(MARKET_DATA_PATH)
data.columns = data.columns.str.strip().str.replace(" ", "_")
data['Option_ID'] = data['Symbol'] + "_" + data['Maturity'] + "_" + data['Strike'].astype(str)

# Create mappings for option data (from the CSV, these columns will be used later via filtering)
# For clarity we’ll refer to today's market info from the DataFrame.
unique_dates = sorted(data['Date'].unique(), key=lambda x: pd.to_datetime(x))
unique_symbols = data['Symbol'].unique().tolist()

# Create mapping for next trading day
next_date = {d: unique_dates[i+1] if i+1 < len(unique_dates) else None 
             for i, d in enumerate(unique_dates)}

# ------------------------------
# Deterministic Spot Moves
# ------------------------------
def get_spot_move(sym):
    if sym == 'SPY':
        return 1.03
    elif sym == 'IWM':
        return 1.036
    elif sym == 'QQQ':
        return 1.041
    else:
        return 1.0  # default if unexpected

# Exposure increment: the additional intrinsic value if the move occurs
def exposure_increment(undl, k, sym):
    move = get_spot_move(sym)
    return max(undl * move - k, 0) - max(undl - k, 0)

# ------------------------------
# Portfolio and Tracking Structures
# ------------------------------
# Portfolio will store positions with keys = Option_ID and values = dict(quantity, cost_basis)
portfolio = {}

# For tracking daily PnL: we track previous day's mark-to-market portfolio value and cumulative premium cost
previous_portfolio_value = 0.0
cumulative_premium_cost = 0.0

all_trades = []        # to store new trades (buys)
daily_summary = []     # daily summary with exposure, portfolio value, and PnL

# ------------------------------
# Daily Loop: Process each trading day
# ------------------------------
for d in unique_dates:
    nd = next_date[d]
    if nd is None:
        # Last day, no next-day scenario so skip optimization
        break
    
    # Get today's market data as a DataFrame
    data_today = data[data['Date'] == d].copy()
    # Build a dictionary of market info for quick lookup by Option_ID
    market_info = {row['Option_ID']: row for _, row in data_today.iterrows()}
    
    # ------------------------------
    # Update Portfolio: Remove expired positions and update exposure for valid positions
    # ------------------------------
    current_portfolio_exposure = 0.0
    updated_portfolio = {}
    
    for opt_id, pos in portfolio.items():
        # Check if this option is still traded today and has not expired (Maturity >= today)
        if opt_id in market_info:
            row = market_info[opt_id]
            if pd.to_datetime(row['Maturity']) >= pd.to_datetime(d):
                # Position is still valid: compute its exposure using today's underlying price
                exp = pos['quantity'] * CONTRACT_SIZE * exposure_increment(row['Undl_Price'], row['Strike'], row['Symbol'])
                current_portfolio_exposure += exp
                updated_portfolio[opt_id] = pos
            else:
                # Option has expired. Compute final intrinsic value:
                intrinsic_value = max(row['Undl_Price'] - row['Strike'], 0)
                final_value = pos['quantity'] * CONTRACT_SIZE * intrinsic_value
                pnl = final_value - pos['cost_basis']
                daily_summary.append({
                    "Date": d,
                    "Option_ID": opt_id,
                    "Expired": True,
                    "Final_Value": final_value,
                    "Cost_Basis": pos['cost_basis'],
                    "Pnl": pnl
                })
                # Option is removed from the portfolio.
        # If not present in today's data, we skip (could be illiquid; alternatively, one might carry forward)
    portfolio = updated_portfolio  # update our portfolio

    # ------------------------------
    # Check Exposure and Optimize New Buys if Needed
    # ------------------------------
    shortage = max(0, MIN_EXPOSURE - current_portfolio_exposure)
    new_premium_cost = 0.0  # cost for new trades today
    if shortage > 0:
        # Consider options available today that will still be active on the next day
        valid_options = data_today[pd.to_datetime(data_today['Maturity']) >= pd.to_datetime(nd)]
        options_for_buy = valid_options['Option_ID'].tolist()
        
        if options_for_buy:
            # For each option, calculate exposure per contract and cost per contract
            exp_per_contract = {}
            cost_per_contract = {}
            avail_qty = {}
            for _, row in valid_options.iterrows():
                o = row['Option_ID']
                exp_per_contract[o] = CONTRACT_SIZE * exposure_increment(row['Undl_Price'], row['Strike'], row['Symbol'])
                cost_per_contract[o] = row['Ask_Price'] * CONTRACT_SIZE
                avail_qty[o] = row['Ask_Size']
            
            # Build Pyomo model for additional buys
            model_buy = ConcreteModel()
            model_buy.OPTIONS = Set(initialize=options_for_buy)
            model_buy.buy = Var(model_buy.OPTIONS, domain=NonNegativeIntegers)
            
            def buy_bound_rule(model, o):
                return model.buy[o] <= avail_qty[o]
            model_buy.buy_bound = Constraint(model_buy.OPTIONS, rule=buy_bound_rule)
            
            def exposure_rule(model):
                return sum(model.buy[o] * exp_per_contract[o] for o in model.OPTIONS) >= shortage
            model_buy.exposure_constraint = Constraint(rule=exposure_rule)
            
            def objective_rule(model):
                return sum(model.buy[o] * cost_per_contract[o] for o in model.OPTIONS)
            model_buy.obj = Objective(rule=objective_rule, sense=minimize)
            
            solver = SolverFactory(SOLVER)
            solver.options['sec'] = WALL_TIME
            result = solver.solve(model_buy, tee=False)
            
            # Process new buys
            for o in model_buy.OPTIONS:
                qty = model_buy.buy[o].value
                if qty is None or qty == 0:
                    continue
                cost = qty * cost_per_contract[o]
                new_premium_cost += cost
                trade_record = {
                    "Date": d,
                    "Option_ID": o,
                    "Symbol": market_info[o]['Symbol'],
                    "Maturity": market_info[o]['Maturity'],
                    "Buy": qty,
                    "Ask_Price": market_info[o]['Ask_Price'],
                    "Bid_Price": market_info[o]['Bid_Price'],
                    "Premium_Cost": cost,
                }
                all_trades.append(trade_record)
                # Add or update the portfolio entry for this option
                if o in portfolio:
                    portfolio[o]['quantity'] += qty
                    portfolio[o]['cost_basis'] += cost
                else:
                    portfolio[o] = {"quantity": qty, "cost_basis": cost}
            
            # Recompute exposure from the new buys (we could also add up from the optimization model)
            for o in options_for_buy:
                if o in portfolio:
                    row = market_info[o]
                    current_portfolio_exposure += portfolio[o]['quantity'] * CONTRACT_SIZE * exposure_increment(row['Undl_Price'], row['Strike'], row['Symbol'])
    
    # ------------------------------
    # Mark-to-Market: Compute Today's Portfolio Value
    # ------------------------------
    portfolio_value = 0.0
    for opt_id, pos in portfolio.items():
        if opt_id in market_info:
            row = market_info[opt_id]
            portfolio_value += pos['quantity'] * row['Bid_Price'] * CONTRACT_SIZE
    
    # ------------------------------
    # Daily PnL Calculation
    # ------------------------------
    # Here, daily PnL is the change in mark-to-market value minus any new premium cost incurred today.
    daily_pnl = portfolio_value - previous_portfolio_value - new_premium_cost
    cumulative_premium_cost += new_premium_cost
    
    daily_summary.append({
        "Date": d,
        "Portfolio_Exposure": current_portfolio_exposure,
        "Portfolio_Value": portfolio_value,
        "New_Premium_Cost": new_premium_cost,
        "Daily_PnL": daily_pnl,
        "Cumulative_Premium_Cost": cumulative_premium_cost
    })
    previous_portfolio_value = portfolio_value

# ------------------------------
# Save and Output Results
# ------------------------------
results_df = pd.DataFrame(all_trades)
results_df.to_csv(OUTPUT_FILE, index=False)
summary_df = pd.DataFrame(daily_summary)
summary_df.to_csv(SUMMARY_FILE, index=False)
print(f"Trade results saved to {OUTPUT_FILE}")
print(f"Daily summary saved to {SUMMARY_FILE}")

# ------------------------------
# Save Deterministic Spot Moves for Analysis
# ------------------------------
spot_moves_data = []
for date in unique_dates:
    for sym in unique_symbols:
        if sym in ['SPY', 'IWM', 'QQQ']:
            move = get_spot_move(sym)
            spot_moves_data.append({"Date": date, "Symbol": sym, "Spot_Move": move})
spot_moves_df = pd.DataFrame(spot_moves_data)
spot_moves_df.to_csv(SPOT_MOVES_DAILY_FILE, index=False)
print(f"Daily spot moves saved to {SPOT_MOVES_DAILY_FILE}")
