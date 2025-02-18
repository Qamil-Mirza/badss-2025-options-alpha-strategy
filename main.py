import pandas as pd
from sim import OptionTradingSimulator
from strategies import ATMStrategy, OTMStrategy

if __name__ == "__main__":
    # Load the data
    file_path = "./data/BADSS training data.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Choose the strategy to test
    selected_strategy = ATMStrategy()  

    # Initialize the simulator with the dataset and run the simulation
    simulator = OptionTradingSimulator(df)
    results = simulator.run_simulation(selected_strategy)

    # Save the results to a CSV file
    strategy_name = selected_strategy.__class__.__name__
    output_file = f"{strategy_name}_simulation_results.csv"
    results_dir = "./results/"
    results.to_csv(results_dir + output_file, index=False)
    print(f"Simulation results saved to {results_dir + output_file}")