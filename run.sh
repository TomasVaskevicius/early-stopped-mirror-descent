#/bin/bash

python -m src.experiments.mirror_descent_complexity_experiment 2>./.err_md_complexity_simulation
python -m src.experiments.range_of_early_stopped_estimator_experiment 2>./.err_md_range_simulation
