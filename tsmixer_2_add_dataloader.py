import sys
from os import getcwd
from os.path import basename, dirname
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import os
run_id = int(os.environ.get("RUN_ID", -1))

import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# 匯入您想使用的 Profiler 類別
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from darts import concatenate
from darts.dataprocessing.transformers.scaler import Scaler
from darts.datasets import ETTh1Dataset, ETTh2Dataset
from darts.metrics import mql
from darts.models import TiDEModel, TSMixerModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models.torch import QuantileRegression
cwd = getcwd()
if basename(cwd) == "examples":
    sys.path.insert(0, dirname(cwd))

series = []
for idx, ds in enumerate([ETTh1Dataset, ETTh2Dataset]):
    trafo = ds().load().astype(np.float32)
    trafo = trafo.with_static_covariates(pd.DataFrame({"transformer_id": [idx]}))
    series.append(trafo)
train, val, test = [], [], []
for trafo in series:
    train_, temp = trafo.split_after(0.6)
    val_, test_ = temp.split_after(0.5)
    train.append(train_)
    val.append(val_)
    test.append(test_)
scaler = Scaler()  # default uses sklearn's MinMaxScaler
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)
def create_params(
    input_chunk_length: int,
    output_chunk_length: int,
    full_training=True,
    enable_profiling=True,
):
    # early stopping: this setting stops training once the the validation
    # loss has not decreased by more than 1e-5 for 10 epochs
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-5,
        mode="min",
    )

    # PyTorch Lightning Trainer arguments (you can add any custom callback)
    if full_training:
        limit_train_batches = None
        limit_val_batches = None
        max_epochs = 200
        batch_size = 256
    else:
        limit_train_batches = 20
        limit_val_batches = 10
        max_epochs = 40
        batch_size = 64

    # only show the training and prediction progress bars
    progress_bar = TFMProgressBar(
        enable_sanity_check_bar=False, enable_validation_bar=False
    )
    profiler = None
    if enable_profiling:
        # 選擇 Profiler：SimpleProfiler 輸出到終端機；AdvancedProfiler 輸出檔案
        # 建議從 SimpleProfiler 開始，之後可以嘗試advanced profiler
        profiler = SimpleProfiler(dirpath="./profiler_results", filename="Advanced_Profiler")
        
        # 如果想使用 PyTorch 內建的 Profiler (更詳細但開銷較大):
        # from pytorch_lightning.profilers import PyTorchProfiler
        # profiler = PyTorchProfiler(dirpath="./profiler_results", filename="pytorch_profiler")
    pl_trainer_kwargs = {
        "devices": [0],           # 🔥 關鍵：只用一個 GPU
        "gradient_clip_val": 1,
        "max_epochs": max_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "accelerator": "auto",
        "callbacks": [early_stopper, progress_bar],
        "profiler": profiler,
    }


    # optimizer setup, uses Adam by default
    # optimizer_cls = torch.optim.Adam
    optimizer_kwargs = {
        "lr": 1e-4,
    }

    # learning rate scheduler
    lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    lr_scheduler_kwargs = {"gamma": 0.999}

    # for probabilistic models, we use quantile regression, and set `loss_fn` to `None`
    likelihood = QuantileRegression()
    loss_fn = None

    return {
        "input_chunk_length": input_chunk_length,  # lookback window
        "output_chunk_length": output_chunk_length,  # forecast/lookahead window
        "use_reversible_instance_norm": True,
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "likelihood": likelihood,  # use a `likelihood` for probabilistic forecasts
        "loss_fn": loss_fn,  # use a `loss_fn` for determinsitic model
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "batch_size": batch_size,
        "random_state": 42,
        "add_encoders": {
            "cyclic": {
                "future": ["hour", "dayofweek", "month"]
            }  # add cyclic time axis encodings as future covariates
        },
    }
# train the models and load the model from its best state/checkpoint

input_chunk_length = 7 * 24
output_chunk_length = 24
use_static_covariates = True
full_training = True
enable_profiling = False 
# create the models
# create the models
model_tsm = TSMixerModel(
    **create_params(
        input_chunk_length,
        output_chunk_length,
        full_training=full_training,
        enable_profiling=enable_profiling, # 🌟 傳入新參數
    ),
    use_static_covariates=use_static_covariates,
    model_name="tsm",
)
model_tide = TiDEModel(
    **create_params(
        input_chunk_length,
        output_chunk_length,
        full_training=full_training,
        enable_profiling=enable_profiling, # 🌟 傳入新參數
    ),
    use_static_covariates=use_static_covariates,
    model_name="tide",
)

models = {
    "TSM": model_tsm,
}


import time
for model_name, model in models.items():
    print(f"正在訓練{model_name}")
    start_t = time.time()
    #CHANGE2: OPTIMIZE DATA LOADER
    model.fit(
        series=train,
        val_series=val,
        dataloader_kwargs={
            "num_workers": 8,     # 可試 4、8，看你機器 CPU 核心數
            "pin_memory": True,   # 如果有 GPU，建議打開
        },
    )
    end_t = time.time()
    print(f"{model_name} fit time: {end_t - start_t:.2f} 秒")

    start_t_load = time.time()
    models[model_name] = model.load_from_checkpoint(
        model_name=model.model_name, best=True
    )
    end_t_load = time.time()
    print(f"{model_name} load checkpoint time: {end_t_load - start_t_load:.2f} 秒")
import csv
import os
from datetime import datetime

# ---- 設定這次實驗的版本名稱 ----
# 例如：base, precision, dataloader, batch
EXPERIMENT_NAME = "fp+dataloader"  # <-- 請依照版本改掉

RESULT_CSV = "results.csv"

# ---- 確保 csv 存在且有欄位名稱 ----
file_exists = os.path.isfile(RESULT_CSV)

with open(RESULT_CSV, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    if not file_exists:
        writer.writerow([
            "timestamp",
            "experiment",
            "run_id",
            "fit_time",
            "load_time"
        ])

    writer.writerow([
        datetime.now().isoformat(),
        EXPERIMENT_NAME,
        run_id,            # 這個變數由 bash 注入（下面會示範）
        round(end_t - start_t, 4),
        round(end_t_load - start_t_load, 4)
    ])

# # configure the probabilistic prediction
# num_samples = 500
# forecast_horizon = output_chunk_length

# # compute the Mean Quantile Loss over these quantiles
# evaluate_quantiles = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]


# def historical_forecasts(model):
#     """Generates probabilistic historical forecasts for each transformer
#     and returns the inverse transformed results.

#     Each forecast covers 24h (forecast_horizon). The time between two forecasts
#     (stride) is also 24 hours.
#     """
#     hfc = model.historical_forecasts(
#         series=test,
#         forecast_horizon=forecast_horizon,
#         stride=forecast_horizon,
#         last_points_only=False,
#         retrain=False,
#         num_samples=num_samples,
#         verbose=True,
#     )
#     return scaler.inverse_transform(hfc)


# def backtest(model, hfc, name):
#     """Evaluates probabilistic historical forecasts using the Mean Quantile
#     Loss (MQL) over a set of quantiles."""
#     # add metric specific kwargs
#     metric_kwargs = [{"q": q} for q in evaluate_quantiles]
#     metrics = [mql for _ in range(len(evaluate_quantiles))]
#     bt = model.backtest(
#         series=series,
#         historical_forecasts=hfc,
#         last_points_only=False,
#         metric=metrics,
#         metric_kwargs=metric_kwargs,
#         verbose=True,
#     )
#     bt = pd.DataFrame(
#         bt,
#         columns=[f"q_{q}" for q in evaluate_quantiles],
#         index=[f"{trafo}_{name}" for trafo in ["ETTh1", "ETTh2"]],
#     )
#     return bt


# def generate_plots(n_days, hfcs):
#     """Plot the probabilistic forecasts for each model, transformer and transformer
#     feature against the ground truth."""
#     # concatenate historical forecasts into contiguous time series
#     # (works because forecast_horizon=stride)
#     hfcs_plot = {}
#     for model_name, hfc_model in hfcs.items():
#         hfcs_plot[model_name] = [
#             concatenate(hfc_series[-n_days:], axis=0) for hfc_series in hfc_model
#         ]

#     # remember start and end points for plotting the target series
#     hfc_ = hfcs_plot[model_name][0]
#     start, end = hfc_.start_time(), hfc_.end_time()

#     # for each target column...
#     for col in series[0].columns:
#         fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
#         # ... and for each transformer...
#         for trafo_idx, trafo in enumerate(series):
#             trafo[col][start:end].plot(label="ground truth", ax=axes[trafo_idx])
#             # ... plot the historical forecasts for each model
#             for model_name, hfc in hfcs_plot.items():
#                 hfc[trafo_idx][col].plot(
#                     label=model_name + "_q0.05-q0.95", ax=axes[trafo_idx]
#                 )
#             axes[trafo_idx].set_title(f"ETTh{trafo_idx + 1}: {col}")
#         plt.show()
# bts = {}
# hfcs = {}
# for model_name, model in models.items():
#     print(f"Model: {model_name}")
#     print("Generating historical forecasts..")
#     hfcs[model_name] = historical_forecasts(models[model_name])

#     print("Evaluating historical forecasts..")
#     bts[model_name] = backtest(models[model_name], hfcs[model_name], model_name)
# bt_df = pd.concat(bts.values(), axis=0).sort_index()
# print(bt_df.mean(axis=1))
