import os
from unittest.mock import patch

import pytest

from heston_mc.experiments import (
    run_baseline_experiment,
    run_parameter_robustness_experiment,
    run_path_sensitivity_experiment,
    run_timestep_sensitivity_experiment,
    setup_results_directory
)


def test_setup_results_directory_new(tmp_path):
    test_dir = tmp_path / "new_dir"
    path = setup_results_directory(str(test_dir))
    assert os.path.exists(path)


def test_setup_results_directory_existing(tmp_path):
    # Edge case: Ensure function doesn't crash if directory already exists
    test_dir = tmp_path / "existing_dir"
    os.makedirs(test_dir)
    path = setup_results_directory(str(test_dir))
    assert os.path.exists(path)


def test_baseline_experiment_execution(capsys):
    run_baseline_experiment()
    captured = capsys.readouterr()
    
    # Verify key outputs are printed to stdout
    assert "Variance Swap:" in captured.out
    assert "Variance Call Option:" in captured.out
    assert "Estimating optimal beta" in captured.out


@patch("heston_mc.experiments.setup_results_directory")
def test_path_sensitivity_file_generation(mock_setup, tmp_path):
    # Redirect output to a temporary directory to avoid polluting the real repo
    mock_setup.return_value = str(tmp_path)
    run_path_sensitivity_experiment()
    
    assert os.path.exists(tmp_path / "path_sensitivity.csv")
    assert os.path.exists(tmp_path / "se_convergence_paths.png")


@patch("heston_mc.experiments.setup_results_directory")
def test_timestep_sensitivity_file_generation(mock_setup, tmp_path):
    mock_setup.return_value = str(tmp_path)
    run_timestep_sensitivity_experiment()
    
    assert os.path.exists(tmp_path / "timestep_sensitivity.csv")
    assert os.path.exists(tmp_path / "price_convergence_steps.png")


@patch("heston_mc.experiments.setup_results_directory")
def test_parameter_robustness_file_generation(mock_setup, tmp_path):
    mock_setup.return_value = str(tmp_path)
    run_parameter_robustness_experiment()
    
    assert os.path.exists(tmp_path / "parameter_robustness.csv")
    assert os.path.exists(tmp_path / "robustness_efficiency_bar.png")