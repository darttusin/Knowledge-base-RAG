torch.range 
==========================================================

torch. range ( *start=0*  , *end*  , *step=1*  , *** , *out=None*  , *dtype=None*  , *layout=torch.strided*  , *device=None*  , *requires_grad=False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a 1-D tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mo fence="true">
             ⌊
            </mo>
<mfrac>
<mrow>
<mtext>
               end
              </mtext>
<mo>
               −
              </mo>
<mtext>
               start
              </mtext>
</mrow>
<mtext>
              step
             </mtext>
</mfrac>
<mo fence="true">
             ⌋
            </mo>
</mrow>
<mo>
            +
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           leftlfloor frac{text{end} - text{start}}{text{step}} rightrfloor + 1
          </annotation>
</semantics>
</math> -->⌊ end − start step ⌋ + 1 leftlfloor frac{text{end} - text{start}}{text{step}} rightrfloor + 1⌊ step end − start ​ ⌋ + 1  with values from `start`  to `end`  with step `step`  . Step is
the gap between two values in the tensor. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mrow>
<mi>
              i
             </mi>
<mo>
              +
             </mo>
<mn>
              1
             </mn>
</mrow>
</msub>
<mo>
            =
           </mo>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            +
           </mo>
<mtext>
            step
           </mtext>
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i+1} = text{out}_i + text{step}.
          </annotation>
</semantics>
</math> -->
out i + 1 = out i + step . text{out}_{i+1} = text{out}_i + text{step}.

out i + 1 ​ = out i ​ + step .

Warning 

This function is deprecated and will be removed in a future release because its behavior is inconsistent with
Python’s range builtin. Instead, use [`torch.arange()`](torch.arange.html#torch.arange "torch.arange")  , which produces values in [start, end).

Parameters
:   * **start** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – the starting value for the set of points. Default: `0`  .
* **end** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the ending value for the set of points
* **step** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – the gap between each pair of adjacent points. Default: `1`  .

Keyword Arguments
:   * **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.
* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ). If *dtype* is not given, infer the data type from the other input
arguments. If any of *start* , *end* , or *step* are floating-point, the *dtype* is inferred to be the default dtype, see [`get_default_dtype()`](torch.get_default_dtype.html#torch.get_default_dtype "torch.get_default_dtype")  . Otherwise, the *dtype* is inferred to
be *torch.int64* .
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned Tensor.
Default: `torch.strided`  .
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , uses the current device for the default tensor type
(see [`torch.set_default_device()`](torch.set_default_device.html#torch.set_default_device "torch.set_default_device")  ). [`device`](../tensor_attributes.html#torch.device "torch.device")  will be the CPU
for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .

Example: 

```
>>> torch.range(1, 4)
tensor([ 1.,  2.,  3.,  4.])
>>> torch.range(1, 4, 0.5)
tensor([ 1.0000,  1.5000,  2.0000,  2.5000,  3.0000,  3.5000,  4.0000])

```

