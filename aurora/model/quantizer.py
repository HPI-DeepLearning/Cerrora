"""Optimized uniform quantization implementation with improved performance.

This module provides efficient quantization classes for neural network weights
and activations, with vectorized operations and reduced memory overhead.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIPMIN = 1e-8


def round_ste(x: torch.Tensor, qmax: int, qmin: int) -> torch.Tensor:
    """Implement Straight-Through Estimator for rounding operation.
    
    Args:
        x: Input tensor to be rounded
        qmax: Maximum quantization value
        qmin: Minimum quantization value
        
    Returns:
        Rounded tensor with gradient flow preserved
    """
    return (x.round().clamp(qmin, qmax) - x).detach() + x

def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Implement Straight-Through Estimator for signing operation.
    
    Args:
        x: Input tensor to be signed
        
    Returns:
        Signed tensor with gradient flow preserved
    """
    return (x.sign() - x).detach() + x

class UniformAffineQuantizer(nn.Module):
    """Optimized uniform affine quantizer for neural network weights.
    Args:
        qparams: Quantization parameters object
        symmetric: Whether to use symmetric quantization
        shape: Weight tensor shape (height, width)
        weight_updates: Enable learnable weight corrections
        shift: Enable learnable scale/zero-point adjustments
        name: Layer name for debugging
    """

    def __init__(
        self,
        qparams: Optional[object] = None,
        symmetric: bool = False,
        shape: Optional[Tuple[int, int]] = None,
        weight_updates: bool = True,
        shift: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Validate inputs
        if qparams is None or shape is None:
            msg = "qparams and shape must be provided"
            raise ValueError(msg)

        # Core quantization parameters
        self.qparams = qparams
        self.n_bits: List[float] = qparams.bits
        self.group_size: Dict[str, int] = qparams.group_size
        self.bits_prop: List[float] = qparams.bits_prop
        self.name = name

        # Validation
        self._validate_parameters()

        # Quantization configuration
        self.symmetric = symmetric
        self.qmin = 0
        self.qmax = 2 ** int(max(self.n_bits)) - 1
        self.qmax_list: List[int] = []
        self.bits_column_end_index: List[int] = []

        # Validate group size divisibility
        if self.group_size is not None:
            min_group_size = min(self.group_size.values())
            if shape[1] % min_group_size != 0:
                msg = "The group size should be divisible!"
                raise AssertionError(msg)

        # Setup bit-width specific parameters
        self._setup_bit_parameters(shape)

        # Training configuration
        self.weight_updates = weight_updates

        self.shift = shift
        self.shape = shape
        self.scale: Optional[List[torch.Tensor]] = None
        self.zeros: Optional[List[torch.Tensor]] = None

        # Initialize learnable parameters
        self._initialize_learnable_parameters(shape)

        # Initialize weight corrections if enabled
        self._initialize_weight_corrections(shape)

        # Precompute constants for efficiency
        self.bpw = self._compute_bits_per_weight()
        self.n = 0
        self._precompute_constants()
        
        # Initialize attributes that may be accessed externally
        self.scale_post: Optional[List[torch.Tensor]] = None
        self.zeros_post: Optional[List[torch.Tensor]] = None

    def bits_count(self) -> float:
        """Backward compatibility method for bits_count calculation.
        
        Returns:
            Average bits per weight (same as bpw)
        """
        return self._compute_bits_per_weight()

    def _validate_parameters(self) -> None:
        """Validate quantization parameters."""
        if len(self.n_bits) != len(self.bits_prop):
            msg = "Bits num is unaligned to bits prop num!"
            raise AssertionError(msg)
        
        if min(self.n_bits) < 1 or max(self.n_bits) > 16:
            msg = "Bits range not supported!"
            raise AssertionError(msg)
        
        if min(self.bits_prop) <= 0 or max(self.bits_prop) > 1:
            msg = "Bits prop range not supported!"
            raise AssertionError(msg)

    def _setup_bit_parameters(self, shape: Tuple[int, int]) -> None:
        """Setup bit-width specific quantization parameters."""
        if len(self.n_bits) > 1:
            for idx in range(len(self.bits_prop)):
                # Calculate qmax for current bit width
                if self.n_bits[idx] != 1.5:
                    qmax = 2 ** math.ceil(self.n_bits[idx]) - 1
                else:
                    qmax = 2 ** math.ceil(self.n_bits[idx]) - 2

                # Calculate column indices for bit allocation
                if idx < len(self.bits_prop) - 1:
                    minimal_columns = 32
                    columns = int(shape[1] * self.bits_prop[idx])
                    columns_index = max(1, columns // minimal_columns) * minimal_columns
                    if idx > 0:
                        columns_index += self.bits_column_end_index[-1]
                    self.bits_column_end_index.append(columns_index)
                else:
                    self.bits_column_end_index.append(shape[1])
                
                self.qmax_list.append(qmax)
        else:
            self.qmax_list.append(self.qmax)
            self.bits_column_end_index.append(shape[1])

    def _initialize_learnable_parameters(self, shape: Tuple[int, int]) -> None:
        """Initialize learnable quantization parameters."""
        group_sizes = list(self.group_size.values())

        if self.shift:
            self._initialize_shift_parameters(shape, group_sizes)
        else:
            group_size = group_sizes[0]
            num_groups = shape[0] * (shape[1] // group_size)
            self.scale_dequant_post = torch.zeros(num_groups, 1)
            if not self.symmetric:
                self.zeros_dequant_post = torch.zeros(num_groups, 1)
            else:
                self.zeros_dequant_post = None

    def _initialize_shift_parameters(
        self, 
        shape: Tuple[int, int], 
        group_sizes: List[int]
    ) -> None:
        """Initialize shift-related learnable parameters."""
        # Initialize parameter lists
        self.scale_dequant_post = nn.ParameterList()
        self.max_scale = nn.ParameterList()
        self.min_scale = nn.ParameterList()

        if not self.symmetric:
            self.zeros_dequant_post = nn.ParameterList()
        else:
            self.zeros_dequant_post = None

        # Initialize parameters for each bit group
        for i in range(len(group_sizes)):
            num_cols = self._get_column_count(i)
            num_groups = num_cols // group_sizes[i]
            
            # Scale parameters
            scale_param = torch.randn(shape[0], num_groups) * 1e-5
            self.scale_dequant_post.append(
                nn.Parameter(scale_param.reshape(-1, 1).half())
            )
            
            max_scale_param = torch.ones(shape[0], num_groups) * 4
            self.max_scale.append(
                nn.Parameter(max_scale_param.reshape(-1, 1).half())
            )
            
            # Zero-point parameters (asymmetric only)
            if not self.symmetric:
                zeros_param = torch.randn(shape[0], num_groups) * 1e-5
                self.zeros_dequant_post.append(
                    nn.Parameter(zeros_param.reshape(-1, 1).half())
                )
                
                min_scale_param = torch.ones(shape[0], num_groups) * 4
                self.min_scale.append(
                    nn.Parameter(min_scale_param.reshape(-1, 1).half())
                )
                
    def _get_column_count(self, bit_index: int) -> int:
        """Get number of columns for a specific bit group."""
        if bit_index == 0:
            return self.bits_column_end_index[0]
        return (
            self.bits_column_end_index[bit_index] 
            - self.bits_column_end_index[bit_index - 1]
        )

    def _initialize_weight_corrections(self, shape: Tuple[int, int]) -> None:
        """Initialize learnable weight correction parameters."""
        if self.weight_updates:
            self.weight = nn.ParameterDict()
            for i in range(len(self.bits_column_end_index)):
                weight_shape = self._get_weight_shape(i, shape)
                weight_key = f"{self.n_bits[i]}_bit"
                weight_param = torch.zeros(weight_shape).half() * 1e-4
                self.weight[weight_key] = nn.Parameter(weight_param)
        else:
            self.weight = None

    def _get_weight_shape(self, bit_index: int, shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get weight shape for a specific bit group."""
        if bit_index == 0:
            return (shape[0], self.bits_column_end_index[0])
        return (
            shape[0], 
            self.bits_column_end_index[bit_index] - self.bits_column_end_index[bit_index - 1]
        )

    def _compute_bits_per_weight(self) -> float:
        """Calculate average bits per weight and set total_bits attribute."""
        bits_counts = []

        for i, end_idx in enumerate(self.bits_column_end_index):
            # Weight bits
            if i == 0:
                num_cols = end_idx
            else:
                num_cols = end_idx - self.bits_column_end_index[i - 1]
            
            weight_bits = num_cols * math.ceil(self.n_bits[i])
            bits_counts.append(weight_bits)

            # Scale bits
            group_size = list(self.group_size.values())[i]
            num_scale_groups = num_cols // group_size
            scale_bits = num_scale_groups * 16
            bits_counts.append(scale_bits)

            # Zero-point bits (asymmetric only)
            if not self.symmetric:
                zero_bits = num_scale_groups * 16
                bits_counts.append(zero_bits)

        # Set total_bits attribute for external access
        self.total_bits = sum(bits_counts) * self.shape[0]
        return self.total_bits / (self.shape[0] * self.shape[1])

    def _precompute_constants(self) -> None:
        """Precompute constants to reduce runtime overhead."""
        self.group_sizes = list(self.group_size.values())
        
        # Precompute symmetric quantization ranges
        self.qmax_symmetric = []
        self.qmin_symmetric = []
        for n_bit in self.n_bits:
            if n_bit > 1:
                self.qmax_symmetric.append(2 ** (n_bit - 1) - 1)
                self.qmin_symmetric.append(-(2 ** (n_bit - 1)))
            else:
                self.qmax_symmetric.append(0)
                self.qmin_symmetric.append(0)
        
        # Precompute tensor slice indices
        self.slice_indices = []
        for i in range(len(self.bits_column_end_index)):
            start_idx = 0 if i == 0 else self.bits_column_end_index[i - 1]
            end_idx = self.bits_column_end_index[i]
            self.slice_indices.append((start_idx, end_idx))

    def _fast_dynamic_quantization_and_fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized combined dynamic quantization and fake quantization.
        
        Merges the computation of quantization parameters and fake quantization
        to reduce memory allocations and improve performance through vectorization.
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Fake-quantized tensor
        """
        self.scale_post = []
        self.zeros_post = []
        result_parts = []
        
        # Process each bit-width group
        for i in range(len(self.bits_column_end_index)):
            start_idx, end_idx = self.slice_indices[i]
            tensor_per_bits = x[:, start_idx:end_idx]
            tensor_shape = tensor_per_bits.size()
            group_size = self.group_sizes[i]
            
            # Reshape for group-wise processing
            tensor_reshaped = tensor_per_bits.view(-1, group_size)

            # Compute quantization parameters
            scale_per_bits, zeros_per_bits = self._compute_quantization_params(
                tensor_reshaped, i
            )

            # Apply quantization
            tensor_quant = self._apply_fake_quantization(
                tensor_reshaped, scale_per_bits, zeros_per_bits, i
            )

            # Apply learnable adjustments
            if self.shift:
                scale_per_bits = scale_per_bits + self.scale_dequant_post[i].clamp(-1, 1)
                if not self.symmetric:
                    zeros_per_bits = zeros_per_bits + self.zeros_dequant_post[i].clamp(-1, 1)

            # Store processed parameters
            self.scale_post.append(scale_per_bits)
            self.zeros_post.append(zeros_per_bits)

            # Dequantize
            tensor_dequant = self._apply_dequantization(
                tensor_quant, scale_per_bits, zeros_per_bits, i
            )

            # Restore original shape and collect results
            result_parts.append(tensor_dequant.view(tensor_shape))

        # Combine results and update counter
        result = torch.cat(result_parts, dim=1)
        self.n += 1
        
        return result

    def _compute_quantization_params(
        self, 
        tensor: torch.Tensor, 
        bit_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute quantization scale and zero-point parameters."""
        if self.n_bits[bit_index] > 1:
            # Multi-bit quantization
            xmin = tensor.amin(-1, keepdim=True)
            xmax = tensor.amax(-1, keepdim=True)

            if not self.symmetric:
                # Asymmetric quantization
                qmax = self.qmax_list[bit_index]
                scale = ((xmax - xmin) / qmax).clamp_(CLIPMIN, 1e8)
                zeros = -xmin
            else:
                # Symmetric quantization
                qmax = self.qmax_symmetric[bit_index]
                scale = torch.max(xmin.abs(), xmax.abs()) / qmax
                if self.shift:
                    scale *= torch.sigmoid(self.max_scale[bit_index])
                zeros = torch.full_like(scale, qmax)
        else:
            # 1-bit quantization
            scale = tensor.norm(p=1, dim=-1, keepdim=True) / tensor.size(-1)
            zeros = -tensor.mean(dim=-1, keepdim=True)

        return scale, zeros

    def _apply_fake_quantization(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zeros: torch.Tensor,
        bit_index: int,
    ) -> torch.Tensor:
        """Apply fake quantization to tensor."""
        if self.n_bits[bit_index] > 1:
            # Multi-bit quantization
            if not self.symmetric:
                tensor_q = tensor + zeros
                qmax_val = self.qmax_list[bit_index]
                qmin_val = 0
            else:
                tensor_q = tensor
                qmax_val = self.qmax_symmetric[bit_index]
                qmin_val = self.qmin_symmetric[bit_index]

            tensor_q = round_ste(tensor_q / scale, qmax_val, qmin_val)

            if self.symmetric:
                tensor_q = (tensor_q + zeros).clamp(0, self.qmax_list[bit_index])
        else:
            # 1-bit quantization
            tensor_q = sign_ste(tensor + zeros)

        return tensor_q

    def _apply_dequantization(
        self,
        tensor_q: torch.Tensor,
        scale: torch.Tensor,
        zeros: torch.Tensor,
        bit_index: int,
    ) -> torch.Tensor:
        """Apply dequantization to recover tensor values."""
        if self.n_bits[bit_index] > 1:
            if self.symmetric:
                tensor_dq = tensor_q - zeros
            else:
                tensor_dq = tensor_q
            
            tensor_dq = tensor_dq * scale
            
            if not self.symmetric:
                tensor_dq = tensor_dq - zeros
        else:
            # 1-bit dequantization
            tensor_dq = tensor_q * scale - zeros

        return tensor_dq

    def forward(self, x: torch.Tensor, enable_fake_quant: bool = False) -> torch.Tensor:
        """Forward pass with optimized quantization.
        
        Args:
            x: Input tensor to quantize
            enable_fake_quant: Whether to apply fake quantization
            
        Returns:
            Quantized or original tensor based on configuration
        """
        # Early exit for high precision or disabled quantization
        if min(self.n_bits) >= 32 or not enable_fake_quant:
            return x

        # Apply weight corrections if enabled
        if self.weight is not None:
            weight_correction = torch.cat(
                [weight for weight in self.weight.values()], 
                dim=-1
            ).to(x.dtype)
            x = x + weight_correction

        # Apply optimized quantization
        return self._fast_dynamic_quantization_and_fake_quant(x)

    def register_scales_and_zeros(self) -> None:
        """Register final scales and zeros, cleanup training parameters."""
        if self.scale_post is None or self.zeros_post is None:
            msg = "scale_post and zeros_post must be computed before registration"
            raise AssertionError(msg)

        # Cleanup training parameters to save memory
        if hasattr(self, 'weight'):
            del self.weight
        if hasattr(self, 'scale_dequant_post'):
            del self.scale_dequant_post
        if hasattr(self, 'zeros_dequant_post') and self.zeros_dequant_post is not None:
            del self.zeros_dequant_post
        if hasattr(self, 'min_scale'):
            del self.min_scale
        if hasattr(self, 'max_scale'):
            del self.max_scale

        torch.cuda.empty_cache()
