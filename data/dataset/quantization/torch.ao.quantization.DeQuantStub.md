DeQuantStub 
==========================================================

*class* torch.ao.quantization. DeQuantStub ( *qconfig = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/ao/quantization/stubs.py#L29) 
:   Dequantize stub module, before calibration, this is same as identity,
this will be swapped as *nnq.DeQuantize* in *convert* . 

Parameters
: **qconfig** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]*  ) – quantization configuration for the tensor,
if qconfig is not provided, we will get qconfig from parent modules

