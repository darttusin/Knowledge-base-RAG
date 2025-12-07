torch.randn 
==========================================================

torch. randn ( **size*  , *** , *generator=None*  , *out=None*  , *dtype=None*  , *layout=torch.strided*  , *device=None*  , *requires_grad=False*  , *pin_memory=False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a tensor filled with random numbers from a normal distribution
with mean *0* and variance *1* (also called the standard normal
distribution). 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            ∼
           </mo>
<mi mathvariant="script">
            N
           </mi>
<mo stretchy="false">
            (
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
           text{out}_{i} sim mathcal{N}(0, 1)
          </annotation>
</semantics>
</math> -->
out i ∼ N ( 0 , 1 ) text{out}_{i} sim mathcal{N}(0, 1)

out i ​ ∼ N ( 0 , 1 )

For complex dtypes, the tensor is i.i.d. sampled from a [complex normal distribution](https://en.wikipedia.org/wiki/Complex_normal_distribution)  with zero mean and
unit variance as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            ∼
           </mo>
<mrow>
<mi mathvariant="script">
             C
            </mi>
<mi mathvariant="script">
             N
            </mi>
</mrow>
<mo stretchy="false">
            (
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
           text{out}_{i} sim mathcal{CN}(0, 1)
          </annotation>
</semantics>
</math> -->
out i ∼ C N ( 0 , 1 ) text{out}_{i} sim mathcal{CN}(0, 1)

out i ​ ∼ C N ( 0 , 1 )

This is equivalent to separately sampling the real <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi mathvariant="normal">
            Re
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (operatorname{Re})
          </annotation>
</semantics>
</math> -->( Re ⁡ ) (operatorname{Re})( Re )  and imaginary <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi mathvariant="normal">
            Im
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (operatorname{Im})
          </annotation>
</semantics>
</math> -->( Im ⁡ ) (operatorname{Im})( Im )  part of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i
          </annotation>
</semantics>
</math> -->out i text{out}_iout i ​  as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
            Re
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            ∼
           </mo>
<mi mathvariant="script">
            N
           </mi>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mfrac>
<mn>
             1
            </mn>
<mn>
             2
            </mn>
</mfrac>
<mo stretchy="false">
            )
           </mo>
<mo separator="true">
            ,
           </mo>
<mspace width="1em">
</mspace>
<mi mathvariant="normal">
            Im
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            ∼
           </mo>
<mi mathvariant="script">
            N
           </mi>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mfrac>
<mn>
             1
            </mn>
<mn>
             2
            </mn>
</mfrac>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           operatorname{Re}(text{out}_{i}) sim mathcal{N}(0, frac{1}{2}),quad
operatorname{Im}(text{out}_{i}) sim mathcal{N}(0, frac{1}{2})
          </annotation>
</semantics>
</math> -->
Re ⁡ ( out i ) ∼ N ( 0 , 1 2 ) , Im ⁡ ( out i ) ∼ N ( 0 , 1 2 ) operatorname{Re}(text{out}_{i}) sim mathcal{N}(0, frac{1}{2}),quad
operatorname{Im}(text{out}_{i}) sim mathcal{N}(0, frac{1}{2})

Re ( out i ​ ) ∼ N ( 0 , 2 1 ​ ) , Im ( out i ​ ) ∼ N ( 0 , 2 1 ​ )

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
>>> torch.randn(4)
tensor([-2.1436,  0.9966,  2.3426, -0.6366])
>>> torch.randn(2, 3)
tensor([[ 1.5954,  2.8929, -1.0923],
        [ 1.1719, -0.4709, -0.1996]])

```

