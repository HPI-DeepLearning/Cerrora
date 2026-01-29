import gc

import torch
import torch.nn as nn
from tqdm import tqdm
from aurora.model.qparam import QParams

# Import quantization packing utilities
try:
    import aurora.model.quant_linear as qlinear_pack
    HAS_BIT_PACKING = True
except ImportError:
    HAS_BIT_PACKING = False
    print("Warning: aurora.model.quant_linear not found. Bit packing will be disabled.")


class CalibrationArgs:
    """Configuration for Aurora quantization calibration."""

    def __init__(self):
        # Quantization parameters
        self.weight_bits = 4
        self.group_size = 32
        self.scale_groups = 32
        self.output_bits = 4
        self.symmetric = True
        self.use_shift = True

        # Optimizer parameters
        self.four_bit_weight_lr = 1e-5
        self.scale_lr = 4e-5
        self.shift_lr = 4e-5
        self.post_lr = 4e-5
        self.norm_lr = 4e-5
        self.wd = 0.0

        # Training parameters
        self.clip_grad = 1.0
        self.kl_loss = False

        # Bit packing
        self.real_quant = True
        self.qqq_format = "torch"

def get_linear_layers_to_replace(module: nn.Module) -> dict:
    """Get a dictionary of all Linear layers in the module that we want to replace.
    This includes all nn.Linear layers except those related to time embeddings.

    Args:
        module: The PyTorch module to search through.

    Returns:
        A dictionary mapping names to nn.Linear instances.
    """
    excluded_keys = ["time", "norm", "downsample", "upsample"]
    return {
        name: m for name, m in module.named_modules()
        if isinstance(m, nn.Linear) and all(ek not in name for ek in excluded_keys)
    }


def replace_linear_layers(backbone, args):
    """Pack quantized weights into actual low-bit format (OmniQuant style).

    This function replaces nn.Linear with QuantLinearTorch (actual quantized format)

    Args:
        backbone: Backbone module
        args: Configuration with packing options

    Returns:
        Backbone with packed weights
    """
    if not args.real_quant:
        print("  Skipping bit packing (real_quant=False)")
        return backbone

    if not HAS_BIT_PACKING:
        print("  Warning: Bit packing not available. Skipping.")
        return backbone

    print("\n" + "=" * 80)
    print("Packing Quantized Weights")
    print("=" * 80)

    # Move to CPU for packing
    backbone = backbone.to("cpu")

    # Get all quantized linear layers
    named_linears = get_linear_layers_to_replace(backbone)

    print(f"Found {len(named_linears)} nn.Linear layers to replace with quantized versions.")

    for name, module in tqdm(named_linears.items(), desc="Replacing linear layers"):
        symmetric = args.symmetric

        qparams = QParams(
                group_size=args.group_size,
                bits=[args.weight_bits],
                bits_prop=[1.0],
                scale_groups=args.scale_groups
        )

        # Create packed quantized linear layer
        q_linear = qlinear_pack.QuantLinearTorch(
            qparams=qparams,
            aparams=None,
            infeatures=module.in_features,
            outfeatures=module.out_features,
            bias=not module.bias is None,
            symmetric=symmetric,
            # qqq_format=args.qqq_format
        )

        q_linear = q_linear.to('cpu')

        # Replace the module with packed version
        levels = name.split('.')
        if len(levels) > 1:
            mod_ = backbone
            for l_idx in range(len(levels) - 1):
                if levels[l_idx].isdigit():
                    mod_ = mod_[int(levels[l_idx])]
                else:
                    mod_ = getattr(mod_, levels[l_idx])
            setattr(mod_, levels[-1], q_linear)
        else:
            setattr(backbone, name, q_linear)

        # Clean up original module
        del module

    # Clean up
    del named_linears
    torch.cuda.empty_cache()
    gc.collect()

    print("✓ Layer replacement complete")

    # Calculate compression statistics
    total_params = sum(p.numel() for p in backbone.parameters())
    total_size_mb = sum(p.numel() * p.element_size() for p in backbone.parameters()) / 1024 / 1024

    print(f"\nCompressed Backbone Statistics:")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Memory size: {total_size_mb:.2f} MB")
    print(f"  Bits per weight: {args.weight_bits}")
    print(f"  Theoretical compression: ~{16 / args.weight_bits:.1f}x from FP16")

    return backbone
