# Berkeley IEOR + Wells Fargo BADSS Case Competition
## **Optimizing Equity Derivative Hedging Strategies**

## **Overview**
This project is a submission for the **Berkeley Algorithm Design & Systems Solutions (BADSS) Case Competition**, hosted by the **Berkeley IEOR Department** in collaboration with **Wells Fargo**. 

The competition focuses on **developing an optimization model** for **equity derivative hedging** under different market scenarios. Participants are provided with a dataset containing **call options** for **SPY (S&P 500), QQQ (Nasdaq 100), and IWM (Russell 2000)**. The goal is to design a trading strategy that:

- **Maintains a minimum exposure of +$10,000,000** under a specified market movement scenario.
- **Minimizes the total option premium cost** across the given timeframe.
- **Tracks daily exposure and profitability**, ensuring no negative exposure under any scenario.

---

## **Project Structure**
The code is organized into modular components for flexibility and ease of experimentation:

- `main.py`: Entry point for running the simulation with provided strategy
- `sim.py`: Core simulation engine
- `strategies.py`: Modular trading strategies
- `data/` : Directory for training and test datasets
- `results` : Directory for generated strategy results
- `README.md` : Documentation

### **1️⃣ simulator.py - Simulation Engine**
Handles execution of trading strategies across all dates and symbols.
- **Loads the dataset**
- **Applies the selected strategy**
- **Tracks daily exposure and costs**
- **Outputs results in a structured DataFrame**

### **2️⃣ strategies.py - Trading Strategies**
Defines different trading strategies as modular components. Each strategy follows an abstract base class and must implement the `execute()` method.

**Current strategies available:**
- **ATMStrategy** → Buys at-the-money (ATM) call options.
- **OTMStrategy** → Buys out-of-the-money (OTM) call options.

Each strategy calculates:
- The **number of contracts needed** to reach $10M exposure.
- The **total cost** of acquiring those contracts.
- Whether the **daily exposure requirement is met** (`Exposure Met` flag).

### **3️⃣ main.py - Running the Simulation**
Allows users to **choose and test different strategies** dynamically.
- Loads the dataset.
- Runs the selected strategy on all dates.
- Saves the results to a file **named after the strategy** (e.g., `ATMStrategy_simulation_results.csv`).
- Displays the results in an interactive table.

---

## **Installation & Setup**
Setup is very minimal at the moment as we just need `python 3.x, numpy, matplotlib, pandas`. I have created an environment.yml file to manage dependencies as a "just in case" measure for when the project expands. For now, a simple `pip install numpy matplotlib pandas` will suffice.