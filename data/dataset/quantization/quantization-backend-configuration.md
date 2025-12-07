Quantization Backend Configuration 
========================================================================================================

FX Graph Mode Quantization allows the user to configure various
quantization behaviors of an op in order to match the expectation
of their backend. 

In the future, this document will contain a detailed spec of
these configurations. 

Default values for native configurations 
--------------------------------------------------------------------------------------------------------------------

Below is the output of the configuration for quantization of ops
in x86 and qnnpack (PyTorch’s default quantized backends). 

Results: 

```
{
  'pattern': <class 'torch.nn.modules.pooling.AdaptiveAvgPool1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method adaptive_avg_pool1d of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <function adaptive_avg_pool2d at 0x7f0264154220>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pooling.AdaptiveAvgPool3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <function adaptive_avg_pool3d at 0x7f02641542c0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in function add>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <built-in method add of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pooling.AvgPool1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method avg_pool1d of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pooling.AvgPool2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in function avg_pool2d>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pooling.AvgPool3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in function avg_pool3d>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv1d'>, <class 'torch.nn.modules.batchnorm.BatchNorm1d'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBn1d'>,
  'fuser_method': <function fuse_conv_bn at 0x7f026392ac00>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.ConvTranspose1d'>, <class 'torch.nn.modules.batchnorm.BatchNorm1d'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.ConvTranspose1d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.ConvTranspose1d'>,
  'fuser_method': <function fuse_convtranspose_bn at 0x7f0263976200>,
},
{
  'pattern': (<class 'torch.nn.modules.linear.Linear'>, <class 'torch.nn.modules.batchnorm.BatchNorm1d'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.LinearBn1d'>,
  'fuser_method': <function fuse_linear_bn at 0x7f0263976160>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv2d'>, <class 'torch.nn.modules.batchnorm.BatchNorm2d'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBn2d'>,
  'fuser_method': <function fuse_conv_bn at 0x7f026392ac00>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.ConvTranspose2d'>, <class 'torch.nn.modules.batchnorm.BatchNorm2d'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.ConvTranspose2d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.ConvTranspose2d'>,
  'fuser_method': <function fuse_convtranspose_bn at 0x7f0263976200>,
},
{
  'pattern': <class 'torch.nn.modules.batchnorm.BatchNorm2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv3d'>, <class 'torch.nn.modules.batchnorm.BatchNorm3d'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBn3d'>,
  'fuser_method': <function fuse_conv_bn at 0x7f026392ac00>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.ConvTranspose3d'>, <class 'torch.nn.modules.batchnorm.BatchNorm3d'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.ConvTranspose3d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.ConvTranspose3d'>,
  'fuser_method': <function fuse_convtranspose_bn at 0x7f0263976200>,
},
{
  'pattern': <class 'torch.nn.modules.batchnorm.BatchNorm3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.BNReLU2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.BNReLU3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <built-in method cat of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method clamp of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': clamp,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': contiguous,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.conv.Conv1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv1d'>,
  'qat_module': <class 'torch.ao.nn.qat.modules.conv.Conv1d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv1d'>,
},
{
  'pattern': <class 'torch.ao.nn.qat.modules.conv.Conv1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv1d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv1d'>,
},
{
  'pattern': <built-in method conv1d of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 1, 'bias': 2},
},
{
  'pattern': <class 'torch.nn.modules.conv.Conv2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv2d'>,
  'qat_module': <class 'torch.ao.nn.qat.modules.conv.Conv2d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv2d'>,
},
{
  'pattern': <class 'torch.ao.nn.qat.modules.conv.Conv2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv2d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv2d'>,
},
{
  'pattern': <built-in method conv2d of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 1, 'bias': 2},
},
{
  'pattern': <class 'torch.nn.modules.conv.Conv3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv3d'>,
  'qat_module': <class 'torch.ao.nn.qat.modules.conv.Conv3d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv3d'>,
},
{
  'pattern': <class 'torch.ao.nn.qat.modules.conv.Conv3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv3d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv3d'>,
},
{
  'pattern': <built-in method conv3d of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 1, 'bias': 2},
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBn1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn1d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv1d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv1d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBn2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn2d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv2d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv2d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBn3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn3d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv3d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv3d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU1d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv1d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv1d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv2d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv2d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU3d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv3d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv3d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU1d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv1d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv1d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv2d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv2d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU3d'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv3d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.Conv3d'>,
},
{
  'pattern': <class 'torch.nn.modules.conv.ConvTranspose1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.ConvTranspose1d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.ConvTranspose1d'>,
},
{
  'pattern': <built-in method conv_transpose1d of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 1, 'bias': 2},
},
{
  'pattern': <class 'torch.nn.modules.conv.ConvTranspose2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.ConvTranspose2d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.ConvTranspose2d'>,
},
{
  'pattern': <built-in method conv_transpose2d of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 1, 'bias': 2},
},
{
  'pattern': <class 'torch.nn.modules.conv.ConvTranspose3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.ConvTranspose3d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.conv.ConvTranspose3d'>,
},
{
  'pattern': <built-in method conv_transpose3d of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 1, 'bias': 2},
},
{
  'pattern': detach,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': detach_,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.dropout.Dropout'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <function dropout at 0x7f0264154360>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.activation.ELU'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <function elu at 0x7f0264154a40>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.sparse.Embedding'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.quint4x2, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.sparse.Embedding'>,
  'qat_module': <class 'torch.ao.nn.qat.modules.embedding_ops.Embedding'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.sparse.Embedding'>,
},
{
  'pattern': <class 'torch.ao.nn.qat.modules.embedding_ops.Embedding'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.quint4x2, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.sparse.Embedding'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.sparse.Embedding'>,
},
{
  'pattern': <class 'torch.nn.modules.sparse.EmbeddingBag'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.quint4x2, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.sparse.EmbeddingBag'>,
  'qat_module': <class 'torch.ao.nn.qat.modules.embedding_ops.EmbeddingBag'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.sparse.EmbeddingBag'>,
},
{
  'pattern': <class 'torch.ao.nn.qat.modules.embedding_ops.EmbeddingBag'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.quint4x2, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.sparse.EmbeddingBag'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.sparse.EmbeddingBag'>,
},
{
  'pattern': <built-in method flatten of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in function floordiv>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <function group_norm at 0x7f0264155b20>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 2, 'bias': 3},
},
{
  'pattern': <class 'torch.nn.modules.rnn.GRU'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.rnn.GRU'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.rnn.GRU'>,
},
{
  'pattern': <class 'torch.nn.modules.rnn.GRUCell'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.rnn.GRUCell'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.rnn.GRUCell'>,
},
{
  'pattern': <class 'torch.nn.modules.activation.Hardsigmoid'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <function hardsigmoid at 0x7f0264155300>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': hardsigmoid,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': hardsigmoid_,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.activation.Hardswish'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <function hardswish at 0x7f02641554e0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.activation.Hardtanh'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <function hardtanh at 0x7f0264154900>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in function hardtanh_>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.linear.Identity'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <function instance_norm at 0x7f0264155940>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 3, 'bias': 4},
},
{
  'pattern': <class 'torch.nn.modules.instancenorm.InstanceNorm1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.instancenorm.InstanceNorm2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.instancenorm.InstanceNorm3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <function interpolate at 0x7f0264183e20>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.normalization.LayerNorm'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <function layer_norm at 0x7f02641559e0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 2, 'bias': 3},
},
{
  'pattern': <class 'torch.nn.modules.activation.LeakyReLU'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <function leaky_relu at 0x7f0264154c20>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.linear.Linear'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.linear.Linear'>,
  'qat_module': <class 'torch.ao.nn.qat.modules.linear.Linear'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.linear.Linear'>,
},
{
  'pattern': <class 'torch.ao.nn.qat.modules.linear.Linear'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.linear.Linear'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.linear.Linear'>,
},
{
  'pattern': <built-in function linear>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'input_type_to_index': {'weight': 1, 'bias': 2},
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.LinearBn1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.linear.Linear'>,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.linear_fused.LinearBn1d'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.linear.Linear'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.linear_fused.LinearBn1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.linear.Linear'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.linear.Linear'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.modules.fused.LinearReLU'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.linear.Linear'>,
  'qat_module': <class 'torch.ao.nn.intrinsic.qat.modules.linear_relu.LinearReLU'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.linear.Linear'>,
},
{
  'pattern': <class 'torch.ao.nn.intrinsic.qat.modules.linear_relu.LinearReLU'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.linear.Linear'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.linear.Linear'>,
},
{
  'pattern': <class 'torch.nn.modules.rnn.LSTM'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.rnn.LSTM'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.rnn.LSTM'>,
},
{
  'pattern': <class 'torch.nn.modules.rnn.LSTMCell'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.rnn.LSTMCell'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.rnn.LSTMCell'>,
},
{
  'pattern': <built-in method matmul of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pooling.MaxPool1d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': torch.nn.functional.max_pool1d,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pooling.MaxPool2d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': torch.nn.functional.max_pool2d,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pooling.MaxPool3d'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': torch.nn.functional.max_pool3d,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method mean of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': mean,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in function mul>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <built-in method mul of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <built-in method narrow of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': permute,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pixelshuffle.PixelShuffle'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method pixel_shuffle of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.pixelshuffle.PixelUnshuffle'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method pixel_unshuffle of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.activation.PReLU'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv1d'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU1d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f039d11d940>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv1d'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU1d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262906200>,
},
{
  'pattern': (<built-in method conv1d of type object at 0x7f038ff8b4a0>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method conv1d of type object at 0x7f038ff8b4a0>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv1d'>, <class 'torch.nn.modules.batchnorm.BatchNorm1d'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU1d'>,
  'fuser_method': <function fuse_conv_bn_relu at 0x7f02639760c0>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv1d'>, <class 'torch.nn.modules.batchnorm.BatchNorm1d'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv1d'>,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU1d'>,
  'fuser_method': <function fuse_conv_bn_relu at 0x7f02639760c0>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv2d'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU2d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940900>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv2d'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU2d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f02629409a0>,
},
{
  'pattern': (<built-in method conv2d of type object at 0x7f038ff8b4a0>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method conv2d of type object at 0x7f038ff8b4a0>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv2d'>, <class 'torch.nn.modules.batchnorm.BatchNorm2d'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU2d'>,
  'fuser_method': <function fuse_conv_bn_relu at 0x7f02639760c0>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv2d'>, <class 'torch.nn.modules.batchnorm.BatchNorm2d'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv2d'>,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU2d'>,
  'fuser_method': <function fuse_conv_bn_relu at 0x7f02639760c0>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv3d'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU3d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940a40>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv3d'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvReLU3d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940ae0>,
},
{
  'pattern': (<built-in method conv3d of type object at 0x7f038ff8b4a0>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method conv3d of type object at 0x7f038ff8b4a0>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv3d'>, <class 'torch.nn.modules.batchnorm.BatchNorm3d'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU3d'>,
  'fuser_method': <function fuse_conv_bn_relu at 0x7f02639760c0>,
},
{
  'pattern': (<class 'torch.nn.modules.conv.Conv3d'>, <class 'torch.nn.modules.batchnorm.BatchNorm3d'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.conv.Conv3d'>,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU3d'>,
  'fuser_method': <function fuse_conv_bn_relu at 0x7f02639760c0>,
},
{
  'pattern': (<class 'torch.nn.modules.linear.Linear'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.LinearReLU'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940b80>,
},
{
  'pattern': (<class 'torch.nn.modules.linear.Linear'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.LinearReLU'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940c20>,
},
{
  'pattern': (<built-in function linear>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in function linear>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in function add>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in function add>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in function add>, <built-in method relu of type object at 0x7f038ff8b4a0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method add of type object at 0x7f038ff8b4a0>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method add of type object at 0x7f038ff8b4a0>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method add of type object at 0x7f038ff8b4a0>, <built-in method relu of type object at 0x7f038ff8b4a0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in function mul>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in function mul>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in function mul>, <built-in method relu of type object at 0x7f038ff8b4a0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method mul of type object at 0x7f038ff8b4a0>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method mul of type object at 0x7f038ff8b4a0>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': (<built-in method mul of type object at 0x7f038ff8b4a0>, <built-in method relu of type object at 0x7f038ff8b4a0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'num_tensor_args_to_observation_type': {
    0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  },
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.activation.ReLU'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <function relu at 0x7f02641547c0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': relu,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': relu_,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': (<class 'torch.nn.modules.batchnorm.BatchNorm2d'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.BNReLU2d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940e00>,
},
{
  'pattern': (<class 'torch.nn.modules.batchnorm.BatchNorm2d'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.BNReLU2d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940ea0>,
},
{
  'pattern': (<class 'torch.nn.modules.batchnorm.BatchNorm3d'>, <class 'torch.nn.modules.activation.ReLU'>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.BNReLU3d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940f40>,
},
{
  'pattern': (<class 'torch.nn.modules.batchnorm.BatchNorm3d'>, <function relu at 0x7f02641547c0>),
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'fused_module': <class 'torch.ao.nn.intrinsic.modules.fused.BNReLU3d'>,
  'fuser_method': <function _sequential_wrapper2.<locals>.fuser_method at 0x7f0262940fe0>,
},
{
  'pattern': <class 'torch.nn.modules.activation.ReLU6'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <function relu6 at 0x7f02641549a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': repeat,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method repeat_interleave of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': repeat_interleave,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': reshape,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': resize_,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.rnn.RNNCell'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.qint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.float32, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'weight_dtype': DTypeWithConstraints(dtype=torch.float16, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'bias_dtype': torch.float32,
      'is_dynamic': True,
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
  'root_module': <class 'torch.nn.modules.rnn.RNNCell'>,
  'reference_quantized_module_for_root': <class 'torch.ao.nn.quantized.reference.modules.rnn.RNNCell'>,
},
{
  'pattern': shape,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.INPUT_OUTPUT_NOT_OBSERVED,
},
{
  'pattern': <class 'torch.nn.modules.activation.Sigmoid'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <built-in method sigmoid of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': sigmoid,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': sigmoid_,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': size,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.INPUT_OUTPUT_NOT_OBSERVED,
},
{
  'pattern': <class 'torch.nn.modules.activation.Softmax'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.00390625, zero_point_exact_match=0),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <built-in method squeeze of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': squeeze,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': squeeze_,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method stack of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <class 'torch.nn.modules.activation.Tanh'>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.0078125, zero_point_exact_match=128),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.0078125, zero_point_exact_match=128),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <built-in method tanh of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.0078125, zero_point_exact_match=128),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.0078125, zero_point_exact_match=128),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': tanh,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.0078125, zero_point_exact_match=128),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.0078125, zero_point_exact_match=128),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': tanh_,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.0078125, zero_point_exact_match=128),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=0.0078125, zero_point_exact_match=128),
    },
  ],
  'observation_type': ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
},
{
  'pattern': <built-in method transpose of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': transpose,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': <built-in method unsqueeze of type object at 0x7f038ff8b4a0>,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': unsqueeze,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': unsqueeze_,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
},
{
  'pattern': view,
  'dtype_configs': [
    {
      'input_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
      'output_dtype': DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=None, quant_max_upper_bound=None, scale_min_lower_bound=None, scale_max_upper_bound=None, scale_exact_match=None, zero_point_exact_match=None),
    },
  ],
  'observation_type': ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
}

```

