import pandas as pd
import wandb

SWEEP_IDS = {
    "TriConvUNext-hyperparameter-sweep": "mnispkk9",
    "TriConvUNext-convolution-type-sweep": "31spdvkd",
    "UNetplusplus-hyperparameter-sweep": "l88c2zec",
    "TriConvUNext-smaller-hyperparameter-sweep": "pukeohhv",
    "Segformer-hyperparameter-sweep": "0yu0hy8p"
}

SWEEPS = {
    "TriConvUNext-hyperparameter-sweep": ["config.learning_rate", "config.weight_decay", "config.scheduler_params.gamma"],
    "TriConvUNext-convolution-type-sweep": ["config.use_deformable", "config.use_dilated", "config.use_depthwise", "summary.number_of_parameters"],
    "UNetplusplus-hyperparameter-sweep": ["config.learning_rate", "config.weight_decay"],
    "TriConvUNext-smaller-hyperparameter-sweep": ["config.learning_rate", "config.weight_decay"],
    "Segformer-hyperparameter-sweep": ["config.learning_rate", "config.weight_decay"]
}

NAMES = {
    "config.learning_rate": 'learning rate',
    "config.weight_decay": 'weight decay',
    "config.scheduler_params.gamma": 'scheduler $\gamma$',
    "config.use_deformable": "deformable",
    "config.use_dilated": "dilated",
    "config.use_depthwise": "depthwise",
    "summary.number_of_parameters": "num. params"
}

def get_param(x, param):
    parts = param.split(".")
    x = eval(x[parts[0]])
    result = None
    if len(parts) > 1:
        for part in parts[1:]:
            x = x[part]
        result = x
    else:
        result = x[param]
    if isinstance(result, float):
        if result < 1e-3:
            return "{:.0e}".format(result)
        else:
            return "{:.4g}".format(result)
    elif isinstance(result, bool):
        return "X" if result else ""
    else:
        return result

for sweep_name, table_params in SWEEPS.items():
    print(f"Producing tables for {sweep_name}...")
    runs_df = pd.read_csv(f"{sweep_name}.csv")
    new_rows = []
    for index, row in runs_df.iterrows():
        new_row = {NAMES[param]: get_param(row, param) for param in table_params}
        new_row["max val Dice"] = "{:.4f}".format(row["highest_val_dice"])
        new_row["max val IoU"] = "{:.4f}".format(row["highest_val_iou"])
        new_rows.append(new_row)
    table_df = pd.DataFrame(new_rows)
    print(table_df)

    table_df.to_latex(f"{sweep_name}_table.tex", index=False, header=True, caption=f"Results for {sweep_name}", label=f"tab-{sweep_name}_table")

# now final test dice and iou comparison of the best models from each sweep
TEST_DICE_LABEL = "summary.test/dice"
TEST_IOU_LABEL = "summary.test/iou"
best_models = {}    
predictions = {}
for sweep_name, table_params in SWEEPS.items():
    runs_df = pd.read_csv(f"{sweep_name}.csv")
    best_model = runs_df.sort_values(by="highest_val_dice", ascending=False).iloc[0]
    best_models[sweep_name] = {
        "test Dice": get_param(best_model, TEST_DICE_LABEL),
        "test IoU": get_param(best_model, TEST_IOU_LABEL),
        "params": get_param(best_model, "summary.number_of_parameters")
    }
    predictions[sweep_name] = (get_param(best_model, "summary.final_predictions")['filenames'], best_model["run_id"])

best_models_df = pd.DataFrame(best_models).T
best_models_df.to_latex(f"best_models_table.tex", index=True, header=True, caption=f"Best models from each sweep", label=f"tab-best_models_table")
print(best_models_df)

# its wandb image file path predictions[sweep_name][i]
api = wandb.Api()
for sweep_name, (preds, run_id) in predictions.items():
    filters = {"Sweep": SWEEP_IDS[sweep_name]}
    runs = api.runs("dl-3-mm-jd/AI-in-Medical-Imaging-Diagnostics", filters=filters)
    for run in runs:
        if run.id == run_id:
            files = run.files(names=preds)
            for file in files:
                file.download(root=f"predictions/{sweep_name}", replace=True)
                print(file.name)
                print(file.url)