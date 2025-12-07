torch.nn.init 
==============================================================

Warning 

All the functions in this module are intended to be used to initialize neural network
parameters, so they all run in [`torch.no_grad()`](generated/torch.no_grad.html#torch.no_grad "torch.no_grad")  mode and will not be taken into
account by autograd.

torch.nn.init. calculate_gain ( *nonlinearity*  , *param = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L139) 
:   Return the recommended gain value for the given nonlinearity function. 

The values are as follows: 

| nonlinearity | gain |
| --- | --- |
| Linear / Identity | <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML"> <semantics> <mrow> <mn> 1 </mn> </mrow> <annotation encoding="application/x-tex"> 1 </annotation> </semantics> </math> -->1 11 |
| Conv{1,2,3}D | <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML"> <semantics> <mrow> <mn> 1 </mn> </mrow> <annotation encoding="application/x-tex"> 1 </annotation> </semantics> </math> -->1 11 |
| Sigmoid | <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML"> <semantics> <mrow> <mn> 1 </mn> </mrow> <annotation encoding="application/x-tex"> 1 </annotation> </semantics> </math> -->1 11 |
| Tanh | <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML"> <semantics> <mrow> <mfrac> <mn> 5 </mn> <mn> 3 </mn> </mfrac> </mrow> <annotation encoding="application/x-tex"> frac{5}{3} </annotation> </semantics> </math> -->5 3 frac{5}{3}3 5 ​ |
| ReLU | <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML"> <semantics> <mrow> <msqrt> <mn> 2 </mn> </msqrt> </mrow> <annotation encoding="application/x-tex"> sqrt{2} </annotation> </semantics> </math> -->2 sqrt{2}2 ​ |
| Leaky Relu | <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML"> <semantics> <mrow> <msqrt> <mfrac> <mn> 2 </mn> <mrow> <mn> 1 </mn> <mo> + </mo> <msup> <mtext> negative_slope </mtext> <mn> 2 </mn> </msup> </mrow> </mfrac> </msqrt> </mrow> <annotation encoding="application/x-tex"> sqrt{frac{2}{1 + text{negative_slope}^2}} </annotation> </semantics> </math> -->2 1 + negative_slope 2 sqrt{frac{2}{1 + text{negative_slope}^2}}1 + negative_slope 2 2 ​ ​ |
| SELU | <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML"> <semantics> <mrow> <mfrac> <mn> 3 </mn> <mn> 4 </mn> </mfrac> </mrow> <annotation encoding="application/x-tex"> frac{3}{4} </annotation> </semantics> </math> -->3 4 frac{3}{4}4 3 ​ |

Warning 

In order to implement [Self-Normalizing Neural Networks](https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html)  ,
you should use `nonlinearity='linear'`  instead of `nonlinearity='selu'`  .
This gives the initial weights a variance of `1 / N`  ,
which is necessary to induce a stable fixed point in the forward pass.
In contrast, the default gain for `SELU`  sacrifices the normalization
effect for more stable gradient flow in rectangular layers.

Parameters
:   * **nonlinearity** ( [*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)") *[* *'linear'* *,* *'conv1d'* *,* *'conv2d'* *,* *'conv3d'* *,* *'conv_transpose1d'* *,* *'conv_transpose2d'* *,* *'conv_transpose3d'* *,* *'sigmoid'* *,* *'tanh'* *,* *'relu'* *,* *'leaky_relu'* *,* *'selu'* *]*  ) – the non-linear function ( *nn.functional* name)
* **param** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]* *]*  ) – optional parameter for the non-linear function

Return type
:   [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")

Examples 

```
>>> gain = nn.init.calculate_gain(
...     "leaky_relu", 0.2
... )  # leaky_relu with negative_slope=0.2

```

torch.nn.init. uniform_ ( *tensor*  , *a = 0.0*  , *b = 1.0*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L213) 
:   Fill the input Tensor with values drawn from the uniform distribution. 

<!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
            U
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            a
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            b
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{U}(a, b)
          </annotation>
</semantics>
</math> -->U ( a , b ) mathcal{U}(a, b)U ( a , b )  . 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **a** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the lower bound of the uniform distribution
* **b** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the upper bound of the uniform distribution
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.uniform_(w)

```

torch.nn.init. normal_ ( *tensor*  , *mean = 0.0*  , *std = 1.0*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L240) 
:   Fill the input Tensor with values drawn from the normal distribution. 

<!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
            N
           </mi>
<mo stretchy="false">
            (
           </mo>
<mtext>
            mean
           </mtext>
<mo separator="true">
            ,
           </mo>
<msup>
<mtext>
             std
            </mtext>
<mn>
             2
            </mn>
</msup>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{N}(text{mean}, text{std}^2)
          </annotation>
</semantics>
</math> -->N ( mean , std 2 ) mathcal{N}(text{mean}, text{std}^2)N ( mean , std 2 )  . 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **mean** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the mean of the normal distribution
* **std** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the standard deviation of the normal distribution
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.normal_(w)

```

torch.nn.init. constant_ ( *tensor*  , *val* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L298) 
:   Fill the input Tensor with the value <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            val
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{val}
          </annotation>
</semantics>
</math> -->val text{val}val  . 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **val** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the value to fill the tensor with

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.constant_(w, 0.3)

```

torch.nn.init. ones_ ( *tensor* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L316) 
:   Fill the input Tensor with the scalar value *1* . 

Parameters
: **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.ones_(w)

```

torch.nn.init. zeros_ ( *tensor* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L329) 
:   Fill the input Tensor with the scalar value *0* . 

Parameters
: **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.zeros_(w)

```

torch.nn.init. eye_ ( *tensor* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L342) 
:   Fill the 2-dimensional input *Tensor* with the identity matrix. 

Preserves the identity of the inputs in *Linear* layers, where as
many inputs are preserved as possible. 

Parameters
: **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – a 2-dimensional *torch.Tensor*

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.eye_(w)

```

torch.nn.init. dirac_ ( *tensor*  , *groups = 1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L363) 
:   Fill the {3, 4, 5}-dimensional input *Tensor* with the Dirac delta function. 

Preserves the identity of the inputs in *Convolutional* layers, where as many input channels are preserved as possible. In case
of groups>1, each group of channels preserves identity 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – a {3, 4, 5}-dimensional *torch.Tensor*
* **groups** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – number of groups in the conv layer (default: 1)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 16, 5, 5)
>>> nn.init.dirac_(w)
>>> w = torch.empty(3, 24, 5, 5)
>>> nn.init.dirac_(w, 3)

```

torch.nn.init. xavier_uniform_ ( *tensor*  , *gain = 1.0*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L437) 
:   Fill the input *Tensor* with values using a Xavier uniform distribution. 

The method is described in *Understanding the difficulty of training
deep feedforward neural networks* - Glorot, X. & Bengio, Y. (2010).
The resulting tensor will have values sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
            U
           </mi>
<mo stretchy="false">
            (
           </mo>
<mo>
            −
           </mo>
<mi>
            a
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            a
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{U}(-a, a)
          </annotation>
</semantics>
</math> -->U ( − a , a ) mathcal{U}(-a, a)U ( − a , a )  where 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
<mo>
            =
           </mo>
<mtext>
            gain
           </mtext>
<mo>
            ×
           </mo>
<msqrt>
<mfrac>
<mn>
              6
             </mn>
<mrow>
<mtext>
               fan_in
              </mtext>
<mo>
               +
              </mo>
<mtext>
               fan_out
              </mtext>
</mrow>
</mfrac>
</msqrt>
</mrow>
<annotation encoding="application/x-tex">
           a = text{gain} times sqrt{frac{6}{text{fan_in} + text{fan_out}}}
          </annotation>
</semantics>
</math> -->
a = gain × 6 fan_in + fan_out a = text{gain} times sqrt{frac{6}{text{fan_in} + text{fan_out}}}

a = gain × fan_in + fan_out 6 ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMzI0MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNDczLDI3OTMKYzMzOS4zLC0xNzk5LjMsNTA5LjMsLTI3MDAsNTEwLC0yNzAyIGwwIC0wCmMzLjMsLTcuMyw5LjMsLTExLDE4LC0xMSBINDAwMDAwdjQwSDEwMTcuNwpzLTkwLjUsNDc4LC0yNzYuMiwxNDY2Yy0xODUuNyw5ODgsLTI3OS41LDE0ODMsLTI4MS41LDE0ODVjLTIsNiwtMTAsOSwtMjQsOQpjLTgsMCwtMTIsLTAuNywtMTIsLTJjMCwtMS4zLC01LjMsLTMyLC0xNiwtOTJjLTUwLjcsLTI5My4zLC0xMTkuNywtNjkzLjMsLTIwNywtMTIwMApjMCwtMS4zLC01LjMsOC43LC0xNiwzMGMtMTAuNywyMS4zLC0yMS4zLDQyLjcsLTMyLDY0cy0xNiwzMywtMTYsMzNzLTI2LC0yNiwtMjYsLTI2CnM3NiwtMTUzLDc2LC0xNTNzNzcsLTE1MSw3NywtMTUxYzAuNywwLjcsMzUuNywyMDIsMTA1LDYwNGM2Ny4zLDQwMC43LDEwMiw2MDIuNywxMDQsCjYwNnpNMTAwMSA4MGg0MDAwMDB2NDBIMTAxNy43eiI+CjwvcGF0aD4KPC9zdmc+)​

Also known as Glorot initialization. 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **gain** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – an optional scaling factor
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))

```

Note 

Be aware that `fan_in`  and `fan_out`  are calculated assuming
that the weight matrix is used in a transposed manner,
(i.e., `x @ w.T`  in `Linear`  layers, where `w.shape = [fan_out, fan_in]`  ).
This is important for correct initialization.
If you plan to use `x @ w`  , where `w.shape = [fan_in, fan_out]`  ,
pass in a transposed weight matrix, i.e. `nn.init.xavier_uniform_(w.T, ...)`  .

torch.nn.init. xavier_normal_ ( *tensor*  , *gain = 1.0*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L478) 
:   Fill the input *Tensor* with values using a Xavier normal distribution. 

The method is described in *Understanding the difficulty of training deep feedforward
neural networks* - Glorot, X. & Bengio, Y. (2010). The resulting tensor
will have values sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
<msup>
<mtext>
             std
            </mtext>
<mn>
             2
            </mn>
</msup>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{N}(0, text{std}^2)
          </annotation>
</semantics>
</math> -->N ( 0 , std 2 ) mathcal{N}(0, text{std}^2)N ( 0 , std 2 )  where 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            std
           </mtext>
<mo>
            =
           </mo>
<mtext>
            gain
           </mtext>
<mo>
            ×
           </mo>
<msqrt>
<mfrac>
<mn>
              2
             </mn>
<mrow>
<mtext>
               fan_in
              </mtext>
<mo>
               +
              </mo>
<mtext>
               fan_out
              </mtext>
</mrow>
</mfrac>
</msqrt>
</mrow>
<annotation encoding="application/x-tex">
           text{std} = text{gain} times sqrt{frac{2}{text{fan_in} + text{fan_out}}}
          </annotation>
</semantics>
</math> -->
std = gain × 2 fan_in + fan_out text{std} = text{gain} times sqrt{frac{2}{text{fan_in} + text{fan_out}}}

std = gain × fan_in + fan_out 2 ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMzI0MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNDczLDI3OTMKYzMzOS4zLC0xNzk5LjMsNTA5LjMsLTI3MDAsNTEwLC0yNzAyIGwwIC0wCmMzLjMsLTcuMyw5LjMsLTExLDE4LC0xMSBINDAwMDAwdjQwSDEwMTcuNwpzLTkwLjUsNDc4LC0yNzYuMiwxNDY2Yy0xODUuNyw5ODgsLTI3OS41LDE0ODMsLTI4MS41LDE0ODVjLTIsNiwtMTAsOSwtMjQsOQpjLTgsMCwtMTIsLTAuNywtMTIsLTJjMCwtMS4zLC01LjMsLTMyLC0xNiwtOTJjLTUwLjcsLTI5My4zLC0xMTkuNywtNjkzLjMsLTIwNywtMTIwMApjMCwtMS4zLC01LjMsOC43LC0xNiwzMGMtMTAuNywyMS4zLC0yMS4zLDQyLjcsLTMyLDY0cy0xNiwzMywtMTYsMzNzLTI2LC0yNiwtMjYsLTI2CnM3NiwtMTUzLDc2LC0xNTNzNzcsLTE1MSw3NywtMTUxYzAuNywwLjcsMzUuNywyMDIsMTA1LDYwNGM2Ny4zLDQwMC43LDEwMiw2MDIuNywxMDQsCjYwNnpNMTAwMSA4MGg0MDAwMDB2NDBIMTAxNy43eiI+CjwvcGF0aD4KPC9zdmc+)​

Also known as Glorot initialization. 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **gain** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – an optional scaling factor
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.xavier_normal_(w)

```

Note 

Be aware that `fan_in`  and `fan_out`  are calculated assuming
that the weight matrix is used in a transposed manner,
(i.e., `x @ w.T`  in `Linear`  layers, where `w.shape = [fan_out, fan_in]`  ).
This is important for correct initialization.
If you plan to use `x @ w`  , where `w.shape = [fan_in, fan_out]`  ,
pass in a transposed weight matrix, i.e. `nn.init.xavier_normal_(w.T, ...)`  .

torch.nn.init. kaiming_uniform_ ( *tensor*  , *a = 0*  , *mode = 'fan_in'*  , *nonlinearity = 'leaky_relu'*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L527) 
:   Fill the input *Tensor* with values using a Kaiming uniform distribution. 

The method is described in *Delving deep into rectifiers: Surpassing
human-level performance on ImageNet classification* - He, K. et al. (2015).
The resulting tensor will have values sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
            U
           </mi>
<mo stretchy="false">
            (
           </mo>
<mo>
            −
           </mo>
<mtext>
            bound
           </mtext>
<mo separator="true">
            ,
           </mo>
<mtext>
            bound
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{U}(-text{bound}, text{bound})
          </annotation>
</semantics>
</math> -->U ( − bound , bound ) mathcal{U}(-text{bound}, text{bound})U ( − bound , bound )  where 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            bound
           </mtext>
<mo>
            =
           </mo>
<mtext>
            gain
           </mtext>
<mo>
            ×
           </mo>
<msqrt>
<mfrac>
<mn>
              3
             </mn>
<mtext>
              fan_mode
             </mtext>
</mfrac>
</msqrt>
</mrow>
<annotation encoding="application/x-tex">
           text{bound} = text{gain} times sqrt{frac{3}{text{fan_mode}}}
          </annotation>
</semantics>
</math> -->
bound = gain × 3 fan_mode text{bound} = text{gain} times sqrt{frac{3}{text{fan_mode}}}

bound = gain × fan_mode 3 ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMzI0MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNDczLDI3OTMKYzMzOS4zLC0xNzk5LjMsNTA5LjMsLTI3MDAsNTEwLC0yNzAyIGwwIC0wCmMzLjMsLTcuMyw5LjMsLTExLDE4LC0xMSBINDAwMDAwdjQwSDEwMTcuNwpzLTkwLjUsNDc4LC0yNzYuMiwxNDY2Yy0xODUuNyw5ODgsLTI3OS41LDE0ODMsLTI4MS41LDE0ODVjLTIsNiwtMTAsOSwtMjQsOQpjLTgsMCwtMTIsLTAuNywtMTIsLTJjMCwtMS4zLC01LjMsLTMyLC0xNiwtOTJjLTUwLjcsLTI5My4zLC0xMTkuNywtNjkzLjMsLTIwNywtMTIwMApjMCwtMS4zLC01LjMsOC43LC0xNiwzMGMtMTAuNywyMS4zLC0yMS4zLDQyLjcsLTMyLDY0cy0xNiwzMywtMTYsMzNzLTI2LC0yNiwtMjYsLTI2CnM3NiwtMTUzLDc2LC0xNTNzNzcsLTE1MSw3NywtMTUxYzAuNywwLjcsMzUuNywyMDIsMTA1LDYwNGM2Ny4zLDQwMC43LDEwMiw2MDIuNywxMDQsCjYwNnpNMTAwMSA4MGg0MDAwMDB2NDBIMTAxNy43eiI+CjwvcGF0aD4KPC9zdmc+)​

Also known as He initialization. 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **a** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the negative slope of the rectifier used after this layer (only
used with `'leaky_relu'`  )
* **mode** ( [*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)") *[* *'fan_in'* *,* *'fan_out'* *]*  ) – either `'fan_in'`  (default) or `'fan_out'`  . Choosing `'fan_in'`  preserves the magnitude of the variance of the weights in the
forward pass. Choosing `'fan_out'`  preserves the magnitudes in the
backwards pass.
* **nonlinearity** ( [*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)") *[* *'linear'* *,* *'conv1d'* *,* *'conv2d'* *,* *'conv3d'* *,* *'conv_transpose1d'* *,* *'conv_transpose2d'* *,* *'conv_transpose3d'* *,* *'sigmoid'* *,* *'tanh'* *,* *'relu'* *,* *'leaky_relu'* *,* *'selu'* *]*  ) – the non-linear function ( *nn.functional* name),
recommended to use only with `'relu'`  or `'leaky_relu'`  (default).
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.kaiming_uniform_(w, mode="fan_in", nonlinearity="relu")

```

Note 

Be aware that `fan_in`  and `fan_out`  are calculated assuming
that the weight matrix is used in a transposed manner,
(i.e., `x @ w.T`  in `Linear`  layers, where `w.shape = [fan_out, fan_in]`  ).
This is important for correct initialization.
If you plan to use `x @ w`  , where `w.shape = [fan_in, fan_out]`  ,
pass in a transposed weight matrix, i.e. `nn.init.kaiming_uniform_(w.T, ...)`  .

torch.nn.init. kaiming_normal_ ( *tensor*  , *a = 0*  , *mode = 'fan_in'*  , *nonlinearity = 'leaky_relu'*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L592) 
:   Fill the input *Tensor* with values using a Kaiming normal distribution. 

The method is described in *Delving deep into rectifiers: Surpassing
human-level performance on ImageNet classification* - He, K. et al. (2015).
The resulting tensor will have values sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
<msup>
<mtext>
             std
            </mtext>
<mn>
             2
            </mn>
</msup>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{N}(0, text{std}^2)
          </annotation>
</semantics>
</math> -->N ( 0 , std 2 ) mathcal{N}(0, text{std}^2)N ( 0 , std 2 )  where 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            std
           </mtext>
<mo>
            =
           </mo>
<mfrac>
<mtext>
             gain
            </mtext>
<msqrt>
<mtext>
              fan_mode
             </mtext>
</msqrt>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{std} = frac{text{gain}}{sqrt{text{fan_mode}}}
          </annotation>
</semantics>
</math> -->
std = gain fan_mode text{std} = frac{text{gain}}{sqrt{text{fan_mode}}}

std = fan_mode ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ gain ​

Also known as He initialization. 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **a** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the negative slope of the rectifier used after this layer (only
used with `'leaky_relu'`  )
* **mode** ( [*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)") *[* *'fan_in'* *,* *'fan_out'* *]*  ) – either `'fan_in'`  (default) or `'fan_out'`  . Choosing `'fan_in'`  preserves the magnitude of the variance of the weights in the
forward pass. Choosing `'fan_out'`  preserves the magnitudes in the
backwards pass.
* **nonlinearity** ( [*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)") *[* *'linear'* *,* *'conv1d'* *,* *'conv2d'* *,* *'conv3d'* *,* *'conv_transpose1d'* *,* *'conv_transpose2d'* *,* *'conv_transpose3d'* *,* *'sigmoid'* *,* *'tanh'* *,* *'relu'* *,* *'leaky_relu'* *,* *'selu'* *]*  ) – the non-linear function ( *nn.functional* name),
recommended to use only with `'relu'`  or `'leaky_relu'`  (default).
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")

```

Note 

Be aware that `fan_in`  and `fan_out`  are calculated assuming
that the weight matrix is used in a transposed manner,
(i.e., `x @ w.T`  in `Linear`  layers, where `w.shape = [fan_out, fan_in]`  ).
This is important for correct initialization.
If you plan to use `x @ w`  , where `w.shape = [fan_in, fan_out]`  ,
pass in a transposed weight matrix, i.e. `nn.init.kaiming_normal_(w.T, ...)`  .

torch.nn.init. trunc_normal_ ( *tensor*  , *mean = 0.0*  , *std = 1.0*  , *a = -2.0*  , *b = 2.0*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L267) 
:   Fill the input Tensor with values drawn from a truncated normal distribution. 

The values are effectively drawn from the
normal distribution <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
            N
           </mi>
<mo stretchy="false">
            (
           </mo>
<mtext>
            mean
           </mtext>
<mo separator="true">
            ,
           </mo>
<msup>
<mtext>
             std
            </mtext>
<mn>
             2
            </mn>
</msup>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{N}(text{mean}, text{std}^2)
          </annotation>
</semantics>
</math> -->N ( mean , std 2 ) mathcal{N}(text{mean}, text{std}^2)N ( mean , std 2 )  with values outside <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            [
           </mo>
<mi>
            a
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            b
           </mi>
<mo stretchy="false">
            ]
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           [a, b]
          </annotation>
</semantics>
</math> -->[ a , b ] [a, b][ a , b ]  redrawn until they are within
the bounds. The method used for generating the random values works
best when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
<mo>
            ≤
           </mo>
<mtext>
            mean
           </mtext>
<mo>
            ≤
           </mo>
<mi>
            b
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           a leq text{mean} leq b
          </annotation>
</semantics>
</math> -->a ≤ mean ≤ b a leq text{mean} leq ba ≤ mean ≤ b  . 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **mean** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the mean of the normal distribution
* **std** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the standard deviation of the normal distribution
* **a** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the minimum cutoff value
* **b** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the maximum cutoff value
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.trunc_normal_(w)

```

torch.nn.init. orthogonal_ ( *tensor*  , *gain = 1*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L645) 
:   Fill the input *Tensor* with a (semi) orthogonal matrix. 

Described in *Exact solutions to the nonlinear dynamics of learning in deep
linear neural networks* - Saxe, A. et al. (2013). The input tensor must have
at least 2 dimensions, and for tensors with more than 2 dimensions the
trailing dimensions are flattened. 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor* , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                n
               </mi>
<mo>
                ≥
               </mo>
<mn>
                2
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               n geq 2
              </annotation>
</semantics>
</math> -->n ≥ 2 n geq 2n ≥ 2

* **gain** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – optional scaling factor
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.orthogonal_(w)

```

torch.nn.init. sparse_ ( *tensor*  , *sparsity*  , *std = 0.01*  , *generator = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L696) 
:   Fill the 2D input *Tensor* as a sparse matrix. 

The non-zero elements will be drawn from the normal distribution <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
            0.01
           </mn>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{N}(0, 0.01)
          </annotation>
</semantics>
</math> -->N ( 0 , 0.01 ) mathcal{N}(0, 0.01)N ( 0 , 0.01 )  , as described in *Deep learning via
Hessian-free optimization* - Martens, J. (2010). 

Parameters
:   * **tensor** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – an n-dimensional *torch.Tensor*
* **sparsity** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – The fraction of elements in each column to be set to zero
* **std** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the standard deviation of the normal distribution used to generate
the non-zero values
* **generator** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Generator*](generated/torch.Generator.html#torch.Generator "torch._C.Generator") *]*  ) – the torch Generator to sample from (default: None)

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Examples 

```
>>> w = torch.empty(3, 5)
>>> nn.init.sparse_(w, sparsity=0.1)

```

