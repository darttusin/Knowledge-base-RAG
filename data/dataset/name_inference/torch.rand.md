torch.rand 
========================================================

torch. rand ( **size*  , *** , *generator=None*  , *out=None*  , *dtype=None*  , *layout=torch.strided*  , *device=None*  , *requires_grad=False*  , *pin_memory=False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a tensor filled with random numbers from a uniform distribution
on the interval <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->[ 0 , 1 ) [0, 1)[ 0 , 1 ) 

The shape of the tensor is defined by the variable argument `size`  . 

Parameters
: **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – a sequence of integers defining the shape of the output tensor.
Can be a variable number of arguments or a collection like a list or tuple.

Keyword Arguments
:   * **generator** ( [`torch.Generator`](torch.Generator.html#torch.Generator "torch.Generator")  , optional) – a pseudorandom number generator for sampling
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.
* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned Tensor.
Default: `torch.strided`  .
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , uses the current device for the default tensor type
(see [`torch.set_default_device()`](torch.set_default_device.html#torch.set_default_device "torch.set_default_device")  ). [`device`](../tensor_attributes.html#torch.device "torch.device")  will be the CPU
for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .
* **pin_memory** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If set, returned tensor would be allocated in
the pinned memory. Works only for CPU tensors. Default: `False`  .

Example: 

```
>>> torch.rand(4)
tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
>>> torch.rand(2, 3)
tensor([[ 0.8237,  0.5781,  0.6879],
        [ 0.3816,  0.7249,  0.0998]])

```

