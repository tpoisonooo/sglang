"""
ModelOpt related constants
"""

import modelopt.torch.quantization as mtq

QUANT_CFG_CHOICES = {
    "fp8": "FP8_DEFAULT_CFG",
    "int4_awq": "INT4_AWQ_CFG",  # TODO: add support for int4_awq
    "w4a8_awq": "W4A8_AWQ_BETA_CFG",  # TODO: add support for w4a8_awq
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "nvfp4_awq": "NVFP4_AWQ_LITE_CFG",  # TODO: add support for nvfp4_awq
}


def get_quant_config_with_layer_exclusion(
    base_config_name: str,
    num_layers: int,
    skip_first_n: int = 1,
    skip_last_n: int = 1,
    layer_prefix: str = "model.layers",
):
    """
    Create a quantization config that excludes first N and last N layers.
    
    According to BRECQ and other quantization research, skipping the first and last
    layers can significantly improve accuracy with minimal impact on speed.
    
    Args:
        base_config_name: Base quantization config name (e.g., "NVFP4_DEFAULT_CFG")
        num_layers: Total number of layers in the model
        skip_first_n: Number of first layers to skip (not quantize)
        skip_last_n: Number of last layers to skip (not quantize)
        layer_prefix: Prefix for layer names (e.g., "model.layers" or "layers")
    
    Returns:
        Modified quantization config dict
    """
    import copy
    
    base_cfg = getattr(mtq, base_config_name)
    if callable(base_cfg):
        base_cfg = base_cfg()
    cfg = copy.deepcopy(base_cfg)
    
    # Add exclusion rules for first N layers
    for i in range(skip_first_n):
        pattern = f"{layer_prefix}.{i}.*"
        cfg['quant_cfg'][pattern] = {'enable': False}
    
    # Add exclusion rules for last N layers
    for i in range(num_layers - skip_last_n, num_layers):
        pattern = f"{layer_prefix}.{i}.*"
        cfg['quant_cfg'][pattern] = {'enable': False}
    
    skipped_layers = list(range(skip_first_n)) + list(range(num_layers - skip_last_n, num_layers))
    print(f"📋 Quantization config: Skipping layers {skipped_layers} (first {skip_first_n} + last {skip_last_n})")
    
    return cfg


def create_layer_excluded_config(
    base_config: dict,
    num_layers: int,
    skip_first_n: int = 1,
    skip_last_n: int = 1,
    layer_prefix: str = "model.layers",
):
    """
    Modify an existing quantization config to exclude first/last layers.
    
    Args:
        base_config: Base quantization config dict
        num_layers: Total number of layers in the model
        skip_first_n: Number of first layers to skip
        skip_last_n: Number of last layers to skip
        layer_prefix: Prefix for layer names
    
    Returns:
        Modified quantization config dict
    """
    import copy
    
    cfg = copy.deepcopy(base_config)
    
    # Add exclusion rules for first N layers
    for i in range(skip_first_n):
        pattern = f"{layer_prefix}.{i}.*"
        cfg['quant_cfg'][pattern] = {'enable': False}
    
    # Add exclusion rules for last N layers
    for i in range(num_layers - skip_last_n, num_layers):
        pattern = f"{layer_prefix}.{i}.*"
        cfg['quant_cfg'][pattern] = {'enable': False}
    
    skipped_layers = list(range(skip_first_n)) + list(range(num_layers - skip_last_n, num_layers))
    print(f"📋 Excluded layers from quantization: {skipped_layers}")
    
    return cfg
