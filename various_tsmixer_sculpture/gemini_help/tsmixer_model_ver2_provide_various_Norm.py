"""
HPC Optimized Time-Series Mixer (TSMixer) - Ablation Ready
----------------------------------------------------------
Optimized with Channel-First memory layout and 1x1 Convolutions.
Supports multiple Normalization types for ablation studies.
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
# 1. Normalization Components
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
        # x: (B, C, T)
        # Normalize over Channel dimension (dim=1)
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** -0.5)
        return self.scale.view(1, -1, 1) * x / (rms_x + self.eps)

class ChannelLayerNorm(nn.Module):
    """
    Adapts standard LayerNorm to work on (B, C, T) layout.
    Standard LayerNorm expects (B, ..., C).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Transpose to (B, T, C) -> Apply Norm -> Transpose back
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)

def get_norm_layer(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "RMSNorm":
        return RMSNorm(dim)
    elif norm_type == "LayerNorm":
        return ChannelLayerNorm(dim)
    elif norm_type == "BatchNorm":
        # BatchNorm1d naturally works on (B, C, T)
        return nn.BatchNorm1d(dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

# ==========================================
# 2. Fast Mixing Blocks (With Configurable Norm)
# ==========================================

class _FastTimeMixing(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, dropout: float, activation: nn.Module, norm_type: str):
        super().__init__()
        # Input: (B, C, T)
        self.norm = get_norm_layer(norm_type, input_dim)
        
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
    def __init__(self, input_dim: int, ff_size: int, dropout: float, activation: nn.Module, norm_type: str):
        super().__init__()
        # Input: (B, C, T)
        # We use 1x1 Conv to mix Channels (C) without permuting
        self.norm = get_norm_layer(norm_type, input_dim)
        
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
                 activation: nn.Module, dropout: float, norm_type: str):
        super().__init__()
        self.time_mixing = _FastTimeMixing(
            sequence_length=sequence_length,
            input_dim=input_dim,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type
        )
        self.feature_mixing = _FastFeatureMixing(
            input_dim=input_dim,
            ff_size=ff_size,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type
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
                 ff_size: int, activation: nn.Module, dropout: float, norm_type: str):
        super().__init__()
        
        # In Channel-First, input_dim is C.
        
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
            dropout=dropout,
            norm_type=norm_type
        )

    def forward(self, x: torch.Tensor, x_static: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, C, T)
        # x_static: (B, S) or (B, 1, S)
        
        if self.static_mixing is not None and x_static is not None:
            # Prepare static: (B, S) -> (B, C) -> (B, C, 1) -> (B, C, T)
            if x_static.ndim == 3: x_static = x_static.squeeze(1)
            
            s_emb = self.static_mixing(x_static) # (B, C)
            s_emb = s_emb.unsqueeze(-1) # (B, C, 1)
            
            x = x + s_emb 

        x = self.block(x)
        return x


# ==========================================
# 3. Main Module
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
        norm_type: str,
        **kwargs,
    ) -> None:
        """
        PLForecastingModule already handles input_chunk_length and output_chunk_length via **kwargs.
        """
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        
        # Define Activation
        act_cls = getattr(nn, activation)
        act_obj = act_cls()

        # --- 1. Projections to Hidden Space ---
        total_hist_dim = input_dim + past_cov_dim + future_cov_dim
        self.hist_proj = nn.Linear(total_hist_dim, hidden_size)
        
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
                    dropout=dropout,
                    norm_type=norm_type
                )
            )

        # --- 3. Output Head ---
        self.head = nn.Linear(hidden_size, output_dim * nr_params)
        
        logger.info(f"TSMixer initialized with norm_type: {norm_type}")

    @io_processor
    def forward(self, x_in: PLModuleInput) -> torch.Tensor:
        # x_hist: (B, T_in, Total_Hist_Dim)
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
# 4. Wrapper Model (Darts Interface)
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
        norm_type: str = "RMSNorm", # Options: "RMSNorm", "LayerNorm", "BatchNorm"
        normalize_before: bool = False,
        use_static_covariates: bool = True,
        **kwargs,
    ) -> None:
        """
        HPC Optimized TSMixer Model with Selectable Norm.
        
        Parameters
        ----------
        norm_type
            Type of normalization to use. 
            Options: "RMSNorm" (Fastest), "LayerNorm" (Standard), "BatchNorm" (Fast convergence)
        """
        model_kwargs = {key: val for key, val in self.model_params.items()}
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
        self.ff_size = ff_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.activation = activation
        self.hidden_size = hidden_size
        self.norm_type = norm_type
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
            norm_type=self.norm_type,
            **self.pl_module_params,
        )

    @property
    def supports_static_covariates(self) -> bool:
        return True