GELU 
============================================

*class* torch.nn. GELU ( *approximate = 'none'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L702) 
:   Applies the Gaussian Error Linear Units function. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            GELU
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi>
            x
           </mi>
<mo>
            ∗
           </mo>
<mi mathvariant="normal">
            Φ
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{GELU}(x) = x * Phi(x)
          </annotation>
</semantics>
</math> -->
GELU ( x ) = x ∗ Φ ( x ) text{GELU}(x) = x * Phi(x)

GELU ( x ) = x ∗ Φ ( x )

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
            Φ
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           Phi(x)
          </annotation>
</semantics>
</math> -->Φ ( x ) Phi(x)Φ ( x )  is the Cumulative Distribution Function for Gaussian Distribution. 

When the approximate argument is ‘tanh’, Gelu is estimated with: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            GELU
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mn>
            0.5
           </mn>
<mo>
            ∗
           </mo>
<mi>
            x
           </mi>
<mo>
            ∗
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            1
           </mn>
<mo>
            +
           </mo>
<mtext>
            Tanh
           </mtext>
<mo stretchy="false">
            (
           </mo>
<msqrt>
<mrow>
<mn>
              2
             </mn>
<mi mathvariant="normal">
              /
             </mi>
<mi>
              π
             </mi>
</mrow>
</msqrt>
<mo>
            ∗
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo>
            +
           </mo>
<mn>
            0.044715
           </mn>
<mo>
            ∗
           </mo>
<msup>
<mi>
             x
            </mi>
<mn>
             3
            </mn>
</msup>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{GELU}(x) = 0.5 * x * (1 + text{Tanh}(sqrt{2 / pi} * (x + 0.044715 * x^3)))
          </annotation>
</semantics>
</math> -->
GELU ( x ) = 0.5 ∗ x ∗ ( 1 + Tanh ( 2 / π ∗ ( x + 0.044715 ∗ x 3 ) ) ) text{GELU}(x) = 0.5 * x * (1 + text{Tanh}(sqrt{2 / pi} * (x + 0.044715 * x^3)))

GELU ( x ) = 0.5 ∗ x ∗ ( 1 + Tanh ( 2/ π ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ ∗ ( x + 0.044715 ∗ x 3 )))

Parameters
: **approximate** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – the gelu approximation algorithm to use: `'none'`  | `'tanh'`  . Default: `'none'`

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*)
              </annotation>
</semantics>
</math> -->( ∗ ) (*)( ∗ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
                ∗
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               *
              </annotation>
</semantics>
</math> -->∗ *∗  means any number of dimensions.

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*)
              </annotation>
</semantics>
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input.

![../_images/GELU.png](../_images/GELU.png)

Examples: 

```
>>> m = nn.GELU()
>>> input = torch.randn(2)
>>> output = m(input)

```

