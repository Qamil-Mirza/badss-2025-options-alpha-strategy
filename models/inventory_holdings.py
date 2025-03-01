import pyomo.environ as pyo
import pandas as pd

# Load dataset
data = pd.read_csv('../data/BADSS training data.csv')

# Clean column names
data.columns = data.columns.str.strip()

# Create a unique option ID
data['Option_ID'] = data['Symbol'] + "_" + data['Maturity'] + "_" + data['Strike'].astype(str)

# Convert dates to numeric indices
data['Date'] = pd.to_datetime(data['Date'])
unique_dates = sorted(data['Date'].unique())
date_index = {date: idx+1 for idx, date in enumerate(unique_dates)}
data['Date_Index'] = data['Date'].map(date_index)

# Define sets
I = data['Option_ID'].unique().tolist()
T = sorted(date_index.values())  # Ensuring numeric and ordered

# Expiration mapping (convert expiration dates to numeric indices)
exp = {option: date_index[pd.to_datetime(data[data['Option_ID'] == option]['Maturity'].iloc[0])]
       for option in I if pd.to_datetime(data[data['Option_ID'] == option]['Maturity'].iloc[0]) in date_index}

# Cost (C), Volume (V), and Exposure (E) dictionaries
C = {(row['Option_ID'], date_index[row['Date']]): row['Ask Price'] for _, row in data.iterrows()}
V = {(row['Option_ID'], date_index[row['Date']]): row['Ask Size'] for _, row in data.iterrows()}

# Compute exposure (E)
E = {}
for option in I:
    option_data = data[data['Option_ID'] == option].sort_values('Date')
    dates = option_data['Date_Index'].tolist()
    strikes = option_data['Strike'].tolist()
    underlier_prices = option_data['Undl Price'].tolist()
    
    for idx in range(len(dates) - 1):
        t = dates[idx]
        S_today = underlier_prices[idx]
        S_next = underlier_prices[idx + 1]
        K = strikes[idx]
        intrinsic_today = max(S_today - K, 0)
        intrinsic_next = max(S_next - K, 0)
        E[(option, t)] = intrinsic_next - intrinsic_today

# -------------------------
# Pyomo Model Definition
# -------------------------
model = pyo.ConcreteModel()

# Define sets
model.I = pyo.Set(initialize=I)
model.T = pyo.Set(initialize=T, ordered=True)

# Define parameters
model.exp = pyo.Param(model.I, initialize=exp, within=pyo.Any)
model.C = pyo.Param(model.I, model.T, initialize=C, within=pyo.Reals)
model.E = pyo.Param(model.I, model.T, initialize=E, within=pyo.Reals, default=0)
model.V = pyo.Param(model.I, model.T, initialize=V, within=pyo.NonNegativeIntegers)

# Decision Variables
model.x = pyo.Var(model.I, model.T, domain=pyo.NonNegativeIntegers)
model.h = pyo.Var(model.I, model.T, domain=pyo.NonNegativeIntegers)

# -------------------------
# Constraints
# -------------------------

# (1) Initial condition: no holdings on first available date
def init_holdings_rule(model, i):
    return model.h[i, min(T)] == 0
model.init_holdings = pyo.Constraint(model.I, rule=init_holdings_rule)

# (2) Inventory dynamics
def inventory_rule(model, i, t):
    if t == min(T):
        return pyo.Constraint.Skip
    if i in model.exp and t - 1 < model.exp[i]:  # Corrected access to model.exp
        return model.h[i, t] == model.h[i, t - 1] + model.x[i, t - 1]
    else:
        return model.h[i, t] == 0
model.inventory = pyo.Constraint(model.I, model.T, rule=inventory_rule)

# (3) Exposure constraints
def exposure_rule(model, t):
    if t == min(T):
        return pyo.Constraint.Skip
    return sum(model.E.get((i, t - 1), 0) * model.h[i, t] for i in model.I) >= 10_000_000
model.exposure_con = pyo.Constraint(model.T, rule=exposure_rule)

# (4) Contract limits
def contract_limit_rule(model, i, t):
    if t >= model.exp.get(i, float('inf')):  # Avoid KeyError
        return model.x[i, t] == 0
    else:
        return model.x[i, t] <= model.V.get((i, t), 0)  # Ensure default value if missing
model.contract_limit = pyo.Constraint(model.I, model.T, rule=contract_limit_rule)

# -------------------------
# Objective Function
# -------------------------
def objective_rule(model):
    return sum(model.C.get((i, t), 0) * model.x[i, t] for i in model.I for t in model.T)
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# -------------------------
# Solve the Model
# -------------------------
WALL_TIME = 10  # seconds
solver = pyo.SolverFactory('glpk')
solver.options['tmlim'] = WALL_TIME

results = solver.solve(model, tee=True)

print("Solver Status:", results.solver.termination_condition)
print("Total Cost: $", pyo.value(model.obj))

# -------------------------
# Save Results to CSV
# -------------------------
results_list = []
for i in model.I:
    for t in T:
        results_list.append({
            "Option": i,
            "Day": t,
            "Purchased": pyo.value(model.x[i, t]),
            "Holdings": pyo.value(model.h[i, t])
        })

results_df = pd.DataFrame(results_list)
results_df.to_csv("kenny_optimized_trades.csv", index=False)
print("Results saved to kenny_optimized_trades.csv")
