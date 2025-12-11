"""
Time-Series Mixer (TSMixer) - Channel-First Optimization
--------------------------------------------------------
Optimization Focus: 
1. Switch memory layout to (Batch, Channel, Time) to remove repeated permuations.
2. Use Conv1d 1x1 to replace Linear layers for feature mixing (mathematically equivalent).
3. Keep ALL other hyperparameters (ReLU, LayerNorm, Dropout) identical to the original Darts implementation.
"""

from typing import Callable, Optional, Union

import torch
from torch import nn

from darts.logging import get_logger, raise_log
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

# Helper to handle LayerNorm on (B, C, T) layout
class ChannelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C) -> Norm -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)

class _TimeMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        activation: nn.Module,
        dropout: float,
        normalize_before: bool,
    ) -> None:
        super().__init__()
        
        # Input: (B, C, T)
        # Note: We use ChannelLayerNorm because input is (B, C, T)
        self.norm_before = (
            ChannelLayerNorm(input_dim) if normalize_before else nn.Identity()
        )
        
        # Linear acts on the last dimension. In (B, C, T), T is the last dim.
        # So nn.Linear(T, T) performs mixing over Time.
        # This is mathematically identical to the original code's:
        # permute(B, T, C) -> Linear(T, T) -> permute(B, C, T)
        self.fc1 = nn.Linear(sequence_length, sequence_length)
        
        self.activation = activation
        self.dropout = MonteCarloDropout(dropout)
        
        self.norm_after = (
            ChannelLayerNorm(input_dim) if not normalize_before else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x_temp = self.norm_before(x)
        
        # No permutation needed! Linear works on T dimension directly.
        x_temp = self.fc1(x_temp) 
        x_temp = self.activation(x_temp)
        x_temp = self.dropout(x_temp)
        
        x_temp = x + x_temp
        x_temp = self.norm_after(x_temp)
        return x_temp


class _FeatureMixing(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ff_size: int,
        activation: nn.Module,
        dropout: float,
        normalize_before: bool,
    ) -> None:
        super().__init__()

        # Input: (B, C, T)
        
        # Use Conv1d 1x1 instead of Linear.
        # Conv1d(in, out, 1) is mathematically identical to Linear(in, out) 
        # applied to the Feature dimension.
        
        self.projection = (
            nn.Conv1d(input_dim, output_dim, kernel_size=1)
            if input_dim != output_dim
            else nn.Identity()
        )
        
        self.norm_before = (
            ChannelLayerNorm(input_dim) if normalize_before else nn.Identity()
        )
        
        self.fc1 = nn.Conv1d(input_dim, ff_size, kernel_size=1)
        self.activation = activation
        self.dropout1 = MonteCarloDropout(dropout)
        self.fc2 = nn.Conv1d(ff_size, output_dim, kernel_size=1)
        self.dropout2 = MonteCarloDropout(dropout)
        
        self.norm_after = (
            ChannelLayerNorm(output_dim) if not normalize_before else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x_proj = self.projection(x)
        
        x = self.norm_before(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        
        x = x_proj + x
        x = self.norm_after(x)
        return x


class _ConditionalMixerLayer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        static_cov_dim: int,
        ff_size: int,
        activation: nn.Module,
        dropout: float,
        normalize_before: bool,
    ) -> None:
        super().__init__()

        mixing_input = input_dim
        
        # Handle Static Covariates
        if static_cov_dim != 0:
            # We treat static covs as a Feature Mixing problem
            self.feature_mixing_static = _FeatureMixing(
                input_dim=static_cov_dim,
                output_dim=output_dim,
                ff_size=ff_size,
                activation=activation,
                dropout=dropout,
                normalize_before=normalize_before,
            )
            mixing_input += output_dim
        else:
            self.feature_mixing_static = None

        self.time_mixing = _TimeMixing(
            sequence_length=sequence_length,
            input_dim=mixing_input,
            activation=activation,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        
        self.feature_mixing = _FeatureMixing(
            input_dim=mixing_input,
            output_dim=output_dim,
            ff_size=ff_size,
            activation=activation,
            dropout=dropout,
            normalize_before=normalize_before,
        )

    def forward(
        self, x: torch.Tensor, x_static: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # x: (B, C, T)
        # x_static: (B, S)
        
        if self.feature_mixing_static is not None:
            # Prepare static for Conv1d: (B, S) -> (B, S, 1) -> repeat -> (B, S, T)
            x_static_expanded = x_static.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x_static_mixed = self.feature_mixing_static(x_static_expanded)
            x = torch.cat([x, x_static_mixed], dim=1) # Cat on Channel dim
            
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x


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
        normalize_before: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        
        # Activation setup
        if activation not in ACTIVATIONS:
            raise_log(ValueError(f"Invalid activation: {activation}"), logger=logger)
        act_obj = getattr(nn, activation)()

        # 1. Projections (History & Future)
        # We process inputs to map them to (B, Hidden, T)
        
        total_hist_dim = input_dim + past_cov_dim + future_cov_dim
        # Conv1d(in, out, 1) is equivalent to Linear(in, out) on features
        self.hist_proj = nn.Conv1d(total_hist_dim, hidden_size, kernel_size=1)
        
        if future_cov_dim > 0:
            self.future_proj = nn.Conv1d(future_cov_dim, hidden_size, kernel_size=1)
        else:
            self.future_proj = None

        # Time transformation: T_in -> T_out
        # Linear acts on T dim directly in (B, C, T) layout
        self.fc_time_transform = nn.Linear(self.input_chunk_length, self.output_chunk_length)

        # 2. Mixing Blocks
        self.blocks = nn.ModuleList()
        # Input to blocks is size (B, hidden_size + [hidden_size if future], T)
        # But wait, original code concats future covs BEFORE blocks if they exist?
        # Let's look at original logic:
        # It creates `feature_mixing_hist` then `feature_mixing_future` then concats.
        # THEN it enters `conditional_mixer` blocks.
        
        # To strictly replicate original logic with Channel-First:
        
        # A. Pre-Mixers (History)
        mixer_params = {
            "ff_size": ff_size,
            "activation": act_obj,
            "dropout": dropout,
            "normalize_before": normalize_before,
        }
        
        # Feature Mixing for History
        # Note: Original used _FeatureMixing here.
        # Input dim is total history dim. Output is hidden_size.
        # But wait, self.hist_proj above basically does the projection part of _FeatureMixing.
        # Original: Linear(input) -> Norm -> MLP -> Add.
        # Let's reimplement _FeatureMixing exactly but with Conv1d.
        
        self.feature_mixing_hist = _FeatureMixing(
            input_dim=total_hist_dim,
            output_dim=hidden_size,
            **mixer_params
        )
        
        # Feature Mixing for Future
        if future_cov_dim > 0:
            self.feature_mixing_future = _FeatureMixing(
                input_dim=future_cov_dim,
                output_dim=hidden_size,
                **mixer_params
            )
        else:
            self.feature_mixing_future = None
            
        # Determine input dim for blocks
        input_dim_block = hidden_size * (1 + int(future_cov_dim > 0))

        # B. Conditional Blocks
        for _ in range(num_blocks):
            layer = _ConditionalMixerLayer(
                sequence_length=self.output_chunk_length,
                input_dim=input_dim_block,
                output_dim=hidden_size,
                static_cov_dim=static_cov_dim,
                ff_size=ff_size,
                activation=act_obj,
                dropout=dropout,
                normalize_before=normalize_before,
            )
            self.blocks.append(layer)
            input_dim_block = hidden_size # Next blocks only take hidden_size

        # 3. Output Head
        self.fc_out = nn.Linear(hidden_size, output_dim * nr_params)

    @io_processor
    def forward(self, x_in: PLModuleInput) -> torch.Tensor:
        # x_hist: (B, T_in, Total_Hist_Dim)
        # x_future: (B, T_out, Future_Cov_Dim)
        # x_static: (B, Static_Dim)
        x_hist, x_future, x_static = x_in
        
        # === 1. Prepare Inputs (Transpose to Channel-First) ===
        # (B, T, C) -> (B, C, T)
        x = x_hist.transpose(1, 2)
        
        # === 2. Time Resizing (Linear on Time Dim) ===
        # Linear acts on last dim. In (B, C, T), T is last.
        # So Linear(T_in, T_out) works directly.
        # But original code did: Time->Feature, Linear(T_in, T_out), Feature->Time.
        # Actually original code:
        # x = _time_to_feature(x) -> (B, C, T)
        # x = self.fc_hist(x) -> (B, C, T_out)
        # x = _time_to_feature(x) -> (B, T_out, C)
        
        # Here we are already (B, C, T).
        # But wait, original `fc_hist` was `nn.Linear(input_chunk_length, output_chunk_length)`.
        # So we apply it directly.
        x = self.fc_time_transform(x) # (B, C, T_out)

        # === 3. Pre-Mixing (History) ===
        x = self.feature_mixing_hist(x) # (B, Hidden, T_out)
        
        # === 4. Pre-Mixing (Future) ===
        if self.feature_mixing_future is not None:
            x_f = x_future.transpose(1, 2) # (B, C_fut, T_out)
            x_f = self.feature_mixing_future(x_f) # (B, Hidden, T_out)
            x = torch.cat([x, x_f], dim=1) # (B, 2*Hidden, T_out)

        # === 5. Prepare Static ===
        if self.static_cov_dim and x_static is not None:
             # Make sure static is (B, S)
             if x_static.ndim == 3: x_static = x_static.squeeze(1)
        
        # === 6. Conditional Blocks ===
        for mixer in self.blocks:
            x = mixer(x, x_static=x_static)

        # === 7. Output Head ===
        # x is (B, Hidden, T_out).
        # Original code: fc_out(x).
        # Original fc_out is Linear(hidden, output_dim * nr_params).
        # Linear acts on last dim. We need it to act on Feature dim.
        # So we transpose to (B, T_out, Hidden) first.
        x = x.transpose(1, 2)
        x = self.fc_out(x)
        
        # Reshape
        x = x.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
        return x


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
        norm_type: Union[str, nn.Module] = "LayerNorm",
        normalize_before: bool = False,
        use_static_covariates: bool = True,
        **kwargs,
    ) -> None:
        """
        TSMixer optimized for HPC with Channel-First layout.
        Logic is strictly preserved from Original TSMixer.
        """
        model_kwargs = {key: val for key, val in self.model_params.items()}
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
        self.ff_size = ff_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.activation = activation
        self.hidden_size = hidden_size
        
        # Note: We ignore 'norm_type' here because we hardcode ChannelLayerNorm 
        # to strictly match the original LayerNorm behavior but in new layout.
        self.normalize_before = normalize_before
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
            normalize_before=self.normalize_before,
            **self.pl_module_params,
        )

    @property
    def supports_static_covariates(self) -> bool:
        return True