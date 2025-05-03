# Gaze-Informed Reinforcement Learning

This project explores how gaze information can help disambiguate subtask goals in a reinforcement learning setting.

## Project Structure

- `environment.py`: Contains the 2D navigation environment with two goals
- `data_collector.py`: Collects trajectories to both goals (multimodal behavior)
- `bc_policy.py`: Behavioral cloning policy model architecture
- `train_bc.py`: Trains BC policy without gaze information
- `gaze_simulator.py`: Simulates human gaze towards the desired goal
- `train_bc_with_gaze.py`: Trains BC policy with gaze information

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Collect multimodal trajectories:
```bash
python data_collector.py
```

2. Train BC policy without gaze:
```bash
python train_bc.py
```

3. Train BC policy with gaze:
```bash
python train_bc_with_gaze.py
```

4. Compare results:
```bash
python compare_results.py
``` 