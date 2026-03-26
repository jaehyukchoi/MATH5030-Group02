# Contributing Guide

## Module ownership

- Yixin Feng: repo structure, params, utils, README setup
- Ha Pham: Euler implementation
- Yingyue Lai: Full Truncation implementation
- Jiani Han: QE implementation
- Lin Tian: analytic benchmark and validation
- Xingjian Tian: benchmarking, robustness, figures, and final integration

## Project rules

- Use the shared `HestonParams` object from `src/heston_mc/params.py`
- Keep parameter names consistent across all modules
- Put reusable Python code in `src/heston_mc/`
- Put tests in `tests/`
