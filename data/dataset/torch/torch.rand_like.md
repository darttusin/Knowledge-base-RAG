torch.rand_like 
===================================================================

torch. rand_like ( *input*  , *** , *dtype = None*  , *layout = None*  , *device = None*  , *requires_grad = False*  , *memory_format = torch.preserve_format* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a tensor with the same size as `input`  that is filled with
random numbers from a uniform distribution on the interval <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            [
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           [0, 1)
          </annotation>
</semantics>
</math> -->[ 0 , 1 ) [0, 1)[ 0 , 1 )  . `torch.rand_like(input)`  is equivalent to `torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`  . 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the size of `input`  will determine size of the output tensor.

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned Tensor.
Default: if `None`  , defaults to the dtype of `input`  .
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned tensor.
Default: if `None`  , defaults to the layout of `input`  .
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , defaults to the device of `input`  .
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .
* **memory_format** ( [`torch.memory_format`](../tensor_attributes.html#torch.memory_format "torch.memory_format")  , optional) – the desired memory format of
returned Tensor. Default: `torch.preserve_format`  .

