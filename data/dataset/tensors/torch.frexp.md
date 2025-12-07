torch.frexp 
==========================================================

torch. frexp ( *input*  , *** , *out=None) -> (Tensor mantissa*  , *Tensor exponent* ) 
:   Decomposes `input`  into mantissa and exponent tensors
such that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            input
           </mtext>
<mo>
            =
           </mo>
<mtext>
            mantissa
           </mtext>
<mo>
            ×
           </mo>
<msup>
<mn>
             2
            </mn>
<mtext>
             exponent
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           text{input} = text{mantissa} times 2^{text{exponent}}
          </annotation>
</semantics>
</math> -->input = mantissa × 2 exponent text{input} = text{mantissa} times 2^{text{exponent}}input = mantissa × 2 exponent  . 

The range of mantissa is the open interval (-1, 1). 

Supports float inputs. 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor

Keyword Arguments
: **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – the output tensors

Example: 

```
>>> x = torch.arange(9.)
>>> mantissa, exponent = torch.frexp(x)
>>> mantissa
tensor([0.0000, 0.5000, 0.5000, 0.7500, 0.5000, 0.6250, 0.7500, 0.8750, 0.5000])
>>> exponent
tensor([0, 1, 2, 2, 3, 3, 3, 3, 4], dtype=torch.int32)
>>> torch.ldexp(mantissa, exponent)
tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])

```

