"""
Time-Series Mixer (TSMixer)
---------------------------
"""

# The inner layers (``nn.Modules``) and the ``TimeBatchNorm2d`` were provided by a PyTorch implementation
# of TSMixer: https://github.com/ditschuk/pytorch-tsmixer
#
# The License of pytorch-tsmixer v0.2.0 from https://github.com/ditschuk/pytorch-tsmixer/blob/main/LICENSE,
# accessed Thursday, March 21st, 2024:
# 'The MIT License
#
# Copyright 2023 Konstantin Ditschuneit
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the “Software”), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# '

from typing import Callable, Optional, Union

import torch
from torch import nn

from darts.logging import get_logger, raise_log
from darts.models.components import layer_norm_variants
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
from darts.utils.torch import MonteCarloDropout

logger = get_logger(__name__)

ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]

NORMS = [
    "LayerNorm",
    "LayerNormNoBias",
    "TimeBatchNorm2d",
    "RMSNorm"
]

def _time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series Tensor to a feature Tensor."""
    return x.permute(0, 2, 1)


class TimeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        """A batch normalization layer that normalizes over the last two dimensions of a Tensor."""
        super().__init__(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` has shape (batch_size, time, features)
        if x.ndim != 3:
            raise_log(
                ValueError(
                    f"Expected 3D input Tensor, but got {x.ndim}D Tensor instead."
                ),
                logger=logger,
            )
        # apply 2D batch norm over reshape input_data `(batch_size, 1, timepoints, features)`
        output = super().forward(x.unsqueeze(1))
        # reshape back to (batch_size, timepoints, features)
        return output.squeeze(1)


"""
HPC Optimized Time-Series Mixer (TSMixer)
-----------------------------------------
Optimized with Channel-First memory layout, RMSNorm, and 1x1 Convolutions
for PyTorch 2.0+ and modern GPUs.
"""

from typing import Callable, Optional, Union, Tuple

import torch
from torch import nn

from darts.logging import get_logger
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample

logger = get_logger(__name__)

# ==========================================
# 1. HPC Components (RMSNorm & Fast Blocks)
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Root Mean Square Layer Normalization.
        Optimized for speed by removing mean calculation.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channel, Time) or (Batch, ..., Dim)
        # We apply norm over the Channel dimension (dim=1) for (B, C, T) layout
        # Or last dim if (B, T, C). TSMixer logic below ensures correct usage.
        
        # Here we assume input is (B, C, T) based on the FastBlock logic
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** -0.5)
        return self.scale.view(1, -1, 1) * x / (rms_x + self.eps)

class _FastTimeMixing(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, dropout: float, activation: nn.Module):
        super().__init__()
        # Input: (B, C, T)
        self.norm = RMSNorm(input_dim)
        # Linear acts on the last dimension (Time) by default in PyTorch
        # so for (B, C, T), Linear(T, T) mixes time.
        self.fc = nn.Sequential(
            nn.Linear(sequence_length, sequence_length),
            activation,
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = x
        x = self.norm(x)
        x = self.fc(x) # Mixes Time (last dim)
        return res + x

class _FastFeatureMixing(nn.Module):
    def __init__(self, input_dim: int, ff_size: int, dropout: float, activation: nn.Module):
        super().__init__()
        # Input: (B, C, T)
        # We use 1x1 Conv to mix Channels (C) without permuting
        self.norm = RMSNorm(input_dim)
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_dim, ff_size, kernel_size=1),
            activation,
            nn.Dropout(dropout),
            nn.Conv1d(ff_size, input_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = x
        x = self.norm(x)
        x = self.conv_net(x) # Mixes Features (dim 1)
        return res + x

class _FastTSMixerBlock(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, ff_size: int, 
                 activation: nn.Module, dropout: float):
        super().__init__()
        self.time_mixing = _FastTimeMixing(
            sequence_length=sequence_length,
            input_dim=input_dim,
            dropout=dropout,
            activation=activation
        )
        self.feature_mixing = _FastFeatureMixing(
            input_dim=input_dim,
            ff_size=ff_size,
            dropout=dropout,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x

class _FastConditionalMixerLayer(nn.Module):
    """
    Handles Static Covariates mixing if present.
    Adapts the conditional mixing logic to the Channel-First layout.
    """
    def __init__(self, sequence_length: int, input_dim: int, static_cov_dim: int, 
                 ff_size: int, activation: nn.Module, dropout: float):
        super().__init__()
        
        # In Channel-First, input_dim is C.
        
        # If static covariates exist, we project them and add to input
        if static_cov_dim > 0:
            self.static_mixing = nn.Sequential(
                nn.Linear(static_cov_dim, input_dim), # Project to Hidden
                activation,
                nn.Dropout(dropout)
            )
        else:
            self.static_mixing = None

        self.block = _FastTSMixerBlock(
            sequence_length=sequence_length,
            input_dim=input_dim,
            ff_size=ff_size,
            activation=activation,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, x_static: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, C, T)
        # x_static: (B, S) or (B, 1, S)
        
        if self.static_mixing is not None and x_static is not None:
            # Prepare static: (B, S) -> (B, C) -> (B, C, 1) -> (B, C, T)
            if x_static.ndim == 3: x_static = x_static.squeeze(1)
            
            s_emb = self.static_mixing(x_static) # (B, C)
            s_emb = s_emb.unsqueeze(-1) # (B, C, 1)
            
            # We simply add the projected static embedding to the dynamic features (common conditioning technique)
            x = x + s_emb 

        x = self.block(x)
        return x


# ==========================================
# 2. Main Module (Fixes Dimension Logic)
# ==========================================

class _TSMixerModule(PLForecastingModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        past_cov_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        hidden_size: int,
        ff_size: int,
        num_blocks: int,
        activation: str,
        dropout: float,
        **kwargs,
    ) -> None:
        """
        PLForecastingModule already handles input_chunk_length and output_chunk_length via **kwargs.
        We do NOT set self.output_chunk_length manually here to avoid AttributeError.
        """
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        
        # Note: self.input_chunk_length and self.output_chunk_length are available 
        # because they are set in PLForecastingModule.__init__

        # Define Activation
        act_cls = getattr(nn, activation)
        act_obj = act_cls()

        # --- 1. Projections to Hidden Space ---
        # FIX: Calculate total input dimension expected from Darts
        # Darts sends x_past = [Target, PastCov, HistFutureCov]
        total_hist_dim = input_dim + past_cov_dim + future_cov_dim
        
        # Project History (B, T_in, Total_Hist) -> (B, T_in, Hidden)
        self.hist_proj = nn.Linear(total_hist_dim, hidden_size)
        
        # Project Future Covariates (B, T_out, FutureCov) -> (B, T_out, Hidden)
        if future_cov_dim > 0:
            self.future_proj = nn.Linear(future_cov_dim, hidden_size)
        else:
            self.future_proj = None

        # Transform Time Axis: (B, T_in, Hidden) -> (B, T_out, Hidden)
        self.fc_time_transform = nn.Linear(self.input_chunk_length, self.output_chunk_length)

        # --- 2. Fast Mixing Blocks ---
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                _FastConditionalMixerLayer(
                    sequence_length=self.output_chunk_length,
                    input_dim=hidden_size, # Everything is in Hidden Space now
                    static_cov_dim=static_cov_dim,
                    ff_size=ff_size,
                    activation=act_obj,
                    dropout=dropout
                )
            )

        # --- 3. Output Head ---
        # (B, Hidden, T_out) -> (B, T_out, OutputDim * Params)
        self.head = nn.Linear(hidden_size, output_dim * nr_params)

    @io_processor
    def forward(self, x_in: PLModuleInput) -> torch.Tensor:
        # x_hist: (B, T_in, Total_Hist_Dim)
        # x_future: (B, T_out, Future_Cov_Dim)
        # x_static: (B, Static_Dim)
        x_hist, x_future, x_static = x_in

        # 1. Project History to Hidden Size
        # x_hist: (B, T_in, Total) -> (B, T_in, 64)
        x = self.hist_proj(x_hist) 

        # 2. Time Transformation (T_in -> T_out)
        # Transpose to (B, Hidden, T_in) to apply Linear on Time
        x = x.transpose(1, 2) # (B, H, T_in)
        x = self.fc_time_transform(x) # (B, H, T_out)
        
        # 3. Handle Future Covariates
        if self.future_proj is not None and x_future is not None:
            # x_future: (B, T_out, F_cov) -> (B, T_out, H)
            x_f = self.future_proj(x_future)
            x_f = x_f.transpose(1, 2) # (B, H, T_out)
            # Add to history representation
            x = x + x_f

        # 4. HPC Blocks (Channel-First Processing)
        # Current state: x is (B, H, T_out) aka (B, C, T)
        # Perfect for our _FastTSMixerBlock
        
        for block in self.blocks:
            x = block(x, x_static=x_static)

        # 5. Output Head
        # Transpose back to (B, T_out, H) for final projection
        x = x.transpose(1, 2) 
        x = self.head(x)

        # Reshape for Darts: (B, T, C, Params)
        x = x.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
        
        return x

# ==========================================
# 3. Wrapper Model (Darts Interface)
# ==========================================

class TSMixerModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hidden_size: int = 64,
        ff_size: int = 64,
        num_blocks: int = 2,
        activation: str = "ReLU",
        dropout: float = 0.1,
        norm_type: str = "RMSNorm", # Placeholder, actually hardcoded to RMS in HPC blocks
        normalize_before: bool = False,
        use_static_covariates: bool = True,
        **kwargs,
    ) -> None:
        """
        HPC Optimized TSMixer Model.
        """
        model_kwargs = {key: val for key, val in self.model_params.items()}
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
        self.ff_size = ff_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.activation = activation
        self.hidden_size = hidden_size
        self.use_static_covariates = use_static_covariates

    def _create_model(self, train_sample: TorchTrainingSample) -> nn.Module:
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        input_dim = past_target.shape[1]
        output_dim = future_target.shape[1]

        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1]
            if static_covariates is not None
            else 0
        )
        future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        past_cov_dim = past_covariates.shape[1] if past_covariates is not None else 0
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        # We pass the separate dimensions to the module, 
        # allowing it to construct the correct input_proj size.
        return _TSMixerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            past_cov_dim=past_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            num_blocks=self.num_blocks,
            activation=self.activation,
            dropout=self.dropout,
            **self.pl_module_params,
        )

    @property
    def supports_static_covariates(self) -> bool:
        return True