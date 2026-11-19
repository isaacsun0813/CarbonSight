# Examples

## SkyPilot + training stub

[`skypilot/`](skypilot/) has a minimal **SkyPilot-style YAML** and a tiny **`train_stub.py`** so you can try the flow without a real model.

From the repo root:

```bash
cd carbonsight
pip install -e .
export WATTTIME_USERNAME=...   # your WattTime API login
export WATTTIME_PASSWORD=...
carbonsight advise --yaml ../examples/skypilot/train.yaml --json
# Or skip writing YAML — point at the script:
carbonsight train ../examples/skypilot/train_stub.py --json
```

CarbonSight uses **`resources`** (GPUs, CPUs, memory), optional top-level **`duration`**, and optional **`carbonsight`** (`finish_by`, `carbon_budget_kg`, `carbon_price`) from the YAML—it does not read your Python file. Those CarbonSight fields are removed before SkyPilot sees the task. GPU utilization: `--gpu-util` or `--nvidia-smi` on the CLI. Replace `train_stub.py` with your real `train.py` and adjust resources to match your workload.

To launch in the cloud you need [SkyPilot](https://skypilot.readthedocs.io/) configured for AWS; then see `carbonsight/README.md` for `carbonsight run`.
