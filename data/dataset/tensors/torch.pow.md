torch.pow 
======================================================

torch. pow ( *input*  , *exponent*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Takes the power of each element in `input`  with `exponent`  and
returns a tensor with the result. 

`exponent`  can be either a single `float`  number or a *Tensor* with the same number of elements as `input`  . 

When `exponent`  is a scalar value, the operation applied is: 

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
            =
           </mo>
<msubsup>
<mi>
             x
            </mi>
<mi>
             i
            </mi>
<mtext>
             exponent
            </mtext>
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i = x_i ^ text{exponent}
          </annotation>
</semantics>
</math> -->
out i = x i exponent text{out}_i = x_i ^ text{exponent}

out i ​ = x i exponent ​

When `exponent`  is a tensor, the operation applied is: 

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
            =
           </mo>
<msubsup>
<mi>
             x
            </mi>
<mi>
             i
            </mi>
<msub>
<mtext>
              exponent
             </mtext>
<mi>
              i
             </mi>
</msub>
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i = x_i ^ {text{exponent}_i}
          </annotation>
</semantics>
</math> -->
out i = x i exponent i text{out}_i = x_i ^ {text{exponent}_i}

out i ​ = x i exponent i ​ ​

When `exponent`  is a tensor, the shapes of `input`  and `exponent`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **exponent** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* *tensor*  ) – the exponent value

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4)
>>> a
tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
>>> torch.pow(a, 2)
tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
>>> exp = torch.arange(1., 5.)

>>> a = torch.arange(1., 5.)
>>> a
tensor([ 1.,  2.,  3.,  4.])
>>> exp
tensor([ 1.,  2.,  3.,  4.])
>>> torch.pow(a, exp)
tensor([   1.,    4.,   27.,  256.])

```

torch. pow ( *self*  , *exponent*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor")
:

`self`  is a scalar `float`  value, and `exponent`  is a tensor.
The returned tensor `out`  is of the same shape as `exponent` 

The operation applied is: 

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
            =
           </mo>
<msup>
<mtext>
             self
            </mtext>
<msub>
<mtext>
              exponent
             </mtext>
<mi>
              i
             </mi>
</msub>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i = text{self} ^ {text{exponent}_i}
          </annotation>
</semantics>
</math> -->
out i = self exponent i text{out}_i = text{self} ^ {text{exponent}_i}

out i ​ = self exponent i ​

Parameters
:   * **self** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the scalar base value for the power operation
* **exponent** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the exponent tensor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> exp = torch.arange(1., 5.)
>>> base = 2
>>> torch.pow(base, exp)
tensor([  2.,   4.,   8.,  16.])

```

