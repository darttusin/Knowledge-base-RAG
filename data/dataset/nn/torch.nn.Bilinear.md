Bilinear 
====================================================

*class* torch.nn. Bilinear ( *in1_features*  , *in2_features*  , *out_features*  , *bias = True*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/linear.py#L150) 
:   Applies a bilinear transformation to the incoming data: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            =
           </mo>
<msubsup>
<mi>
             x
            </mi>
<mn>
             1
            </mn>
<mi>
             T
            </mi>
</msubsup>
<mi>
            A
           </mi>
<msub>
<mi>
             x
            </mi>
<mn>
             2
            </mn>
</msub>
<mo>
            +
           </mo>
<mi>
            b
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y = x_1^T A x_2 + b
          </annotation>
</semantics>
</math> -->y = x 1 T A x 2 + b y = x_1^T A x_2 + by = x 1 T ​ A x 2 ​ + b  . 

Parameters
:   * **in1_features** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – size of each first input sample, must be > 0
* **in2_features** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – size of each second input sample, must be > 0
* **out_features** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – size of each output sample, must be > 0
* **bias** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to `False`  , the layer will not learn an additive bias.
Default: `True`

Shape:
:   * Input1: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 H
                </mi>
<mtext>
                 in1
                </mtext>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, H_text{in1})
              </annotation>
</semantics>
</math> -->( ∗ , H in1 ) (*, H_text{in1})( ∗ , H in1 ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 H
                </mi>
<mtext>
                 in1
                </mtext>
</msub>
<mo>
                =
               </mo>
<mtext>
                in1_features
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               H_text{in1}=text{in1_features}
              </annotation>
</semantics>
</math> -->H in1 = in1_features H_text{in1}=text{in1_features}H in1 ​ = in1_features  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  means any number of additional dimensions including none. All but the last dimension
of the inputs should be the same.

* Input2: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 H
                </mi>
<mtext>
                 in2
                </mtext>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, H_text{in2})
              </annotation>
</semantics>
</math> -->( ∗ , H in2 ) (*, H_text{in2})( ∗ , H in2 ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 H
                </mi>
<mtext>
                 in2
                </mtext>
</msub>
<mo>
                =
               </mo>
<mtext>
                in2_features
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               H_text{in2}=text{in2_features}
              </annotation>
</semantics>
</math> -->H in2 = in2_features H_text{in2}=text{in2_features}H in2 ​ = in2_features  .

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 H
                </mi>
<mtext>
                 out
                </mtext>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, H_text{out})
              </annotation>
</semantics>
</math> -->( ∗ , H out ) (*, H_text{out})( ∗ , H out ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 H
                </mi>
<mtext>
                 out
                </mtext>
</msub>
<mo>
                =
               </mo>
<mtext>
                out_features
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               H_text{out}=text{out_features}
              </annotation>
</semantics>
</math> -->H out = out_features H_text{out}=text{out_features}H out ​ = out_features  and all but the last dimension are the same shape as the input.

Variables
:   * **weight** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable weights of the module of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mtext>
                out_features
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
                in1_features
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
                in2_features
               </mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (text{out_features}, text{in1_features}, text{in2_features})
              </annotation>
</semantics>
</math> -->( out_features , in1_features , in2_features ) (text{out_features}, text{in1_features}, text{in2_features})( out_features , in1_features , in2_features )  .
The values are initialized from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msqrt>
<mi>
                 k
                </mi>
</msqrt>
<mo separator="true">
                ,
               </mo>
<msqrt>
<mi>
                 k
                </mi>
</msqrt>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               mathcal{U}(-sqrt{k}, sqrt{k})
              </annotation>
</semantics>
</math> -->U ( − k , k ) mathcal{U}(-sqrt{k}, sqrt{k})U ( − k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ , k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mo>
                =
               </mo>
<mfrac>
<mn>
                 1
                </mn>
<mtext>
                 in1_features
                </mtext>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
               k = frac{1}{text{in1_features}}
              </annotation>
</semantics>
</math> -->k = 1 in1_features k = frac{1}{text{in1_features}}k = in1_features 1 ​

* **bias** – the learnable bias of the module of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mtext>
                out_features
               </mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (text{out_features})
              </annotation>
</semantics>
</math> -->( out_features ) (text{out_features})( out_features )  .
If `bias`  is `True`  , the values are initialized from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msqrt>
<mi>
                 k
                </mi>
</msqrt>
<mo separator="true">
                ,
               </mo>
<msqrt>
<mi>
                 k
                </mi>
</msqrt>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               mathcal{U}(-sqrt{k}, sqrt{k})
              </annotation>
</semantics>
</math> -->U ( − k , k ) mathcal{U}(-sqrt{k}, sqrt{k})U ( − k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ , k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mo>
                =
               </mo>
<mfrac>
<mn>
                 1
                </mn>
<mtext>
                 in1_features
                </mtext>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
               k = frac{1}{text{in1_features}}
              </annotation>
</semantics>
</math> -->k = 1 in1_features k = frac{1}{text{in1_features}}k = in1_features 1 ​

Examples: 

```
>>> m = nn.Bilinear(20, 30, 40)
>>> input1 = torch.randn(128, 20)
>>> input2 = torch.randn(128, 30)
>>> output = m(input1, input2)
>>> print(output.size())
torch.Size([128, 40])

```

