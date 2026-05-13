import os
import time

import pandas as pd
import matplotlib.pyplot as plt


from .params import default_heston_params, default_mc_config
from .simulation import HestonModelSimulator
from .interfaces import SimulationResult
from .control_variate import (
    compare_variance_swap_methods, 
    price_variance_option_control_variate,
    estimate_variance_option_beta_from_pilot 
)
from .variance_option import price_variance_option


def setup_results_directory(dir_name="results"):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def run_baseline_experiment():
    print("Running baseline Heston model pricing...")

    params = default_heston_params()
    config = default_mc_config()
    strike_variance = params.theta

    start_time = time.time()
    simulator = HestonModelSimulator(params, config)
    S, v = simulator.simulate()
    sim_runtime = time.time() - start_time
    print(f"Simulation completed in {sim_runtime:.4f}s.\n")

    dt = config.maturity / config.n_steps
    sim_result = SimulationResult(stock_paths=S, variance_paths=v, dt=dt)

    
    swap_results = compare_variance_swap_methods(
        sim_result=sim_result,
        strike=strike_variance,
        rate=params.r,
        maturity=config.maturity,
        params=params
    )
    
    print("Variance Swap:")
    print(f"  Plain MC: {swap_results['plain_price']:.6f} (SE: {swap_results['plain_std_error']:.6f})")
    print(f"  CV MC: {swap_results['control_variate_price']:.6f} (SE: {swap_results['control_variate_std_error']:.6f})")
    print(f"  Ratio: {swap_results['std_error_improvement_ratio']:.2f}x")

    print("\nEstimating optimal beta for Variance Option from independent pilot sample...")
    beta_opt = estimate_variance_option_beta_from_pilot(
        params=params,
        config=config,
        strike=strike_variance,
        rate=params.r,
        maturity=config.maturity
    )
    print(f"Estimated Optimal Beta: {beta_opt:.4f}")

    # Variance Option
    plain_opt_res = price_variance_option(
        sim_result=sim_result,
        strike=strike_variance,
        rate=params.r,
        maturity=config.maturity
    )
    
    cv_opt_res = price_variance_option_control_variate(
        sim_result=sim_result,
        strike=strike_variance,
        rate=params.r,
        maturity=config.maturity,
        params=params,
        beta=beta_opt  
    )

    opt_improvement_ratio = plain_opt_res.std_error / cv_opt_res.std_error
    
    print("\nVariance Call Option:")
    print(f"  Plain MC: {plain_opt_res.price:.6f} (SE: {plain_opt_res.std_error:.6f})")
    print(f"  CV MC:    {cv_opt_res.price:.6f} (SE: {cv_opt_res.std_error:.6f})")
    print(f"  Ratio:    {opt_improvement_ratio:.2f}x\n")


def run_path_sensitivity_experiment():
    print("Running path-count sensitivity analysis...")

    params = default_heston_params()
    paths_to_test = [5000, 10000, 20000, 50000, 100000]
    strike = params.theta
    output_dir = setup_results_directory()
    results_data = []

    for n_paths in paths_to_test:
        config = default_mc_config()
        config.n_paths = n_paths
        dt = config.maturity / config.n_steps
        
        simulator = HestonModelSimulator(params, config)
        S, v = simulator.simulate()
        sim_result = SimulationResult(stock_paths=S, variance_paths=v, dt=dt)
        
       
        beta_opt = estimate_variance_option_beta_from_pilot(
            params=params, config=config, strike=strike, rate=params.r, maturity=config.maturity
        )
        
        plain_res = price_variance_option(
            sim_result=sim_result, strike=strike, rate=params.r, maturity=config.maturity
        )
        
        cv_res = price_variance_option_control_variate(
            sim_result=sim_result, strike=strike, rate=params.r, maturity=config.maturity, 
            params=params, beta=beta_opt
        )
        
        results_data.append({
            "n_paths": n_paths,
            "plain_se": plain_res.std_error,
            "cv_se": cv_res.std_error,
            "se_reduction": plain_res.std_error / cv_res.std_error,
            "plain_runtime": plain_res.runtime_seconds,
            "cv_runtime": cv_res.runtime_seconds
        })

    df = pd.DataFrame(results_data)
    df.to_csv(os.path.join(output_dir, "path_sensitivity.csv"), index=False)

    x_paths = df["n_paths"].tolist()
    y_plain_se = df["plain_se"].tolist()
    y_cv_se = df["cv_se"].tolist()

    plt.figure(figsize=(8, 5))
    plt.plot(x_paths, y_plain_se, marker='o', label="Plain MC")
    plt.plot(x_paths, y_cv_se, marker='s', label="Control Variate MC")
    plt.title("Standard Error vs. Path Count")
    plt.xlabel("Number of Paths")
    plt.ylabel("Standard Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "se_convergence_paths.png"))
    plt.close()


def run_timestep_sensitivity_experiment():
    print("Running time-step sensitivity analysis...")

    params = default_heston_params()
    steps_to_test = [50, 100, 252, 500]
    strike = params.theta
    output_dir = setup_results_directory()
    results_data = []

    for n_steps in steps_to_test:
        config = default_mc_config()
        config.n_paths = 10000 
        config.n_steps = n_steps
        config.periods_per_year = n_steps
        dt = config.maturity / config.n_steps
        
        simulator = HestonModelSimulator(params, config)
        S, v = simulator.simulate()
        sim_result = SimulationResult(stock_paths=S, variance_paths=v, dt=dt)
        
        
        beta_opt = estimate_variance_option_beta_from_pilot(
            params=params, config=config, strike=strike, rate=params.r, maturity=config.maturity
        )
        
        plain_res = price_variance_option(
            sim_result=sim_result, strike=strike, rate=params.r, maturity=config.maturity
        )
        
        cv_res = price_variance_option_control_variate(
            sim_result=sim_result, strike=strike, rate=params.r, maturity=config.maturity, 
            params=params, beta=beta_opt
        )
        
        results_data.append({
            "n_steps": n_steps,
            "plain_price": plain_res.price,
            "cv_price": cv_res.price,
            "plain_se": plain_res.std_error,
            "cv_se": cv_res.std_error,
            "cv_runtime": cv_res.runtime_seconds
        })

    df = pd.DataFrame(results_data)
    df.to_csv(os.path.join(output_dir, "timestep_sensitivity.csv"), index=False)

    x_steps = df["n_steps"].tolist()
    y_plain_price = df["plain_price"].tolist()
    y_cv_price = df["cv_price"].tolist()

    plt.figure(figsize=(8, 5))
    plt.plot(x_steps, y_plain_price, marker='o', label="Plain MC")
    plt.plot(x_steps, y_cv_price, marker='s', label="Control Variate MC")
    plt.title("Estimated Price vs. Time Steps")
    plt.xlabel("Number of Time Steps")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "price_convergence_steps.png"))
    plt.close()


def run_parameter_robustness_experiment():
    print("Running parameter robustness analysis...")

    output_dir = setup_results_directory()
    config = default_mc_config()
    config.n_paths = 20000
    
    scenarios = [{"name": "Baseline", "sigma": 0.50, "rho": -0.70, "maturity": 1.0},
        {"name": "High Vol of Vol", "sigma": 0.90, "rho": -0.70, "maturity": 1.0},
        {"name": "Strong Neg Correlation", "sigma": 0.50, "rho": -0.95, "maturity": 1.0},
        {"name": "Longer Maturity", "sigma": 0.50, "rho": -0.70, "maturity": 3.0}]

    results_data = []

    for scenario in scenarios:
        params = default_heston_params()
        params.sigma = scenario["sigma"]
        params.rho = scenario["rho"]
        
        current_config = default_mc_config()
        current_config.n_paths = config.n_paths
        current_config.maturity = scenario["maturity"]
        current_config.n_steps = int(current_config.periods_per_year * scenario["maturity"])
        
        dt = current_config.maturity / current_config.n_steps
        strike = params.theta

        simulator = HestonModelSimulator(params, current_config)
        S, v = simulator.simulate()
        sim_result = SimulationResult(stock_paths=S, variance_paths=v, dt=dt)
        
        
        beta_opt = estimate_variance_option_beta_from_pilot(
            params=params, config=current_config, strike=strike, rate=params.r, maturity=current_config.maturity
        )
        
        plain_res = price_variance_option(
            sim_result=sim_result, strike=strike, rate=params.r, maturity=current_config.maturity
        )
        
        cv_res = price_variance_option_control_variate(
            sim_result=sim_result, strike=strike, rate=params.r, maturity=current_config.maturity, 
            params=params, beta=beta_opt
        )
        
        reduction_ratio = plain_res.std_error / cv_res.std_error
        
        results_data.append({
            "Scenario": scenario["name"],
            "Plain_SE": plain_res.std_error,
            "CV_SE": cv_res.std_error,
            "SE_Reduction_Ratio": reduction_ratio
        })

    df = pd.DataFrame(results_data)
    df.to_csv(os.path.join(output_dir, "parameter_robustness.csv"), index=False)

    scenario_names = df["Scenario"].tolist()
    ratios = df["SE_Reduction_Ratio"].tolist()

    plt.figure(figsize=(9, 6))
    bars = plt.bar(scenario_names, ratios, color='skyblue', edgecolor='black')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}x", ha='center', va='bottom')

    plt.title("Control Variate Efficiency Under Different Market Scenarios")
    plt.ylabel("Standard Error Reduction Ratio (Higher is Better)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "robustness_efficiency_bar.png"))
    plt.close()