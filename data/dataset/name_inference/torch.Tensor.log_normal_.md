torch.Tensor.log_normal_ 
=====================================================================================

Tensor. log_normal_ ( *mean = 1*  , *std = 2*  , *** , *generator = None* ) 
:   Fills `self`  tensor with numbers samples from the log-normal distribution
parameterized by the given mean <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            μ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           mu
          </annotation>
</semantics>
</math> -->μ muμ  and standard deviation <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            σ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           sigma
          </annotation>
</semantics>
</math> -->σ sigmaσ  . Note that [`mean`](torch.mean.html#torch.mean "torch.mean")  and [`std`](torch.std.html#torch.std "torch.std")  are the mean and
standard deviation of the underlying normal distribution, and not of the
returned distribution: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            f
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
<mo>
            =
           </mo>
<mfrac>
<mn>
             1
            </mn>
<mrow>
<mi>
              x
             </mi>
<mi>
              σ
             </mi>
<msqrt>
<mrow>
<mn>
                2
               </mn>
<mi>
                π
               </mi>
</mrow>
</msqrt>
</mrow>
</mfrac>
<mtext>
</mtext>
<msup>
<mi>
             e
            </mi>
<mrow>
<mo>
              −
             </mo>
<mfrac>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                ln
               </mi>
<mo>
                ⁡
               </mo>
<mi>
                x
               </mi>
<mo>
                −
               </mo>
<mi>
                μ
               </mi>
<msup>
<mo stretchy="false">
                 )
                </mo>
<mn>
                 2
                </mn>
</msup>
</mrow>
<mrow>
<mn>
                2
               </mn>
<msup>
<mi>
                 σ
                </mi>
<mn>
                 2
                </mn>
</msup>
</mrow>
</mfrac>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           f(x) = dfrac{1}{x sigma sqrt{2pi}} e^{-frac{(ln x - mu)^2}{2sigma^2}}
          </annotation>
</semantics>
</math> -->
f ( x ) = 1 x σ 2 π e − ( ln ⁡ x − μ ) 2 2 σ 2 f(x) = dfrac{1}{x sigma sqrt{2pi}} e^{-frac{(ln x - mu)^2}{2sigma^2}}

f ( x ) = x σ 2 π ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 1 ​ e − 2 σ 2 ( l n x − μ ) 2 ​

