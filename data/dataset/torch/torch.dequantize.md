torch.dequantize 
====================================================================

torch. dequantize ( *tensor* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns an fp32 Tensor by dequantizing a quantized Tensor 

Parameters
: **tensor** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – A quantized Tensor

torch. dequantize ( *tensors* ) → sequence of Tensors
:

Given a list of quantized Tensors, dequantize them and return a list of fp32 Tensors 

Parameters
: **tensors** ( *sequence* *of* *Tensors*  ) – A list of quantized Tensors

