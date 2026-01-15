import pandas as pd
import wandb
api = wandb.Api(timeout=60)

SWEEPS = {
    "TriConvUNext-hyperparameter-sweep": "mnispkk9",
    "TriConvUNext-convolution-type-sweep": "31spdvkd",
    "UNetplusplus-hyperparameter-sweep": "l88c2zec",
    "TriConvUNext-smaller-hyperparameter-sweep": "pukeohhv",
    "Segformer-hyperparameter-sweep": "0yu0hy8p"
}

for sweep_name, sweep_id in SWEEPS.items():
    print(f"Fetching {sweep_name}...")
    filters = {"Sweep": sweep_id}
    runs = api.runs("dl-3-mm-jd/AI-in-Medical-Imaging-Diagnostics", filters=filters)
    runs_df = runs.histories(samples=20, keys=["epoch", "val/loss", "val/dice", "val/iou", "val/accuracy"], format="pandas", x_axis="_step")
    grouped_runs_df = runs_df.groupby("run_id")
    highest_val_dice = grouped_runs_df["val/dice"].max()
    highest_val_iou = grouped_runs_df["val/iou"].max()
    highest_val_accuracy = grouped_runs_df["val/accuracy"].max()

    runs_list = []
    for run in runs:
        summary = run.summary._json_dict
        config = {k: v for k,v in run.config.items() if not k.startswith('_')}
        runs_list.append({
            "summary": summary,
            "config": config,
            "run_id": run.id,
            "name": run.name,
            "highest_val_dice": highest_val_dice[run.id],
            "highest_val_iou": highest_val_iou[run.id],
            "highest_val_accuracy": highest_val_accuracy[run.id]
        })

    runs_df = pd.DataFrame(runs_list)
    runs_df.to_csv(f"{sweep_name}.csv")