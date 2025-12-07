torch.special 
==============================================================

The torch.special module, modeled after SciPy’s [special](https://docs.scipy.org/doc/scipy/reference/special.html)  module. 

Functions 
------------------------------------------------------

torch.special. airy_ai ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Airy function <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             Ai
            </mtext>
<mrow>
<mo fence="true">
              (
             </mo>
<mtext>
              input
             </mtext>
<mo fence="true">
              )
             </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
            text{Ai}left(text{input}right)
           </annotation>
</semantics>
</math> -->Ai ( input ) text{Ai}left(text{input}right)Ai ( input )  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. bessel_j0 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Bessel function of the first kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            0
           </annotation>
</semantics>
</math> -->0 00  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. bessel_j1 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Bessel function of the first kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. bessel_y0 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Bessel function of the second kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            0
           </annotation>
</semantics>
</math> -->0 00  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. bessel_y1 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Bessel function of the second kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. chebyshev_polynomial_t ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Chebyshev polynomial of the first kind <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              T
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            T_{n}(text{input})
           </annotation>
</semantics>
</math> -->T n ( input ) T_{n}(text{input})T n ​ ( input )  . 

If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 0
           </annotation>
</semantics>
</math> -->n = 0 n = 0n = 0  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  is returned. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 1
           </annotation>
</semantics>
</math> -->n = 1 n = 1n = 1  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             input
            </mtext>
</mrow>
<annotation encoding="application/x-tex">
            text{input}
           </annotation>
</semantics>
</math> -->input text{input}input  is returned. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             &lt;
            </mo>
<mn>
             6
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n &lt; 6
           </annotation>
</semantics>
</math> -->n < 6 n < 6n < 6  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             ∣
            </mi>
<mtext>
             input
            </mtext>
<mi mathvariant="normal">
             ∣
            </mi>
<mo>
             &gt;
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            |text{input}| &gt; 1
           </annotation>
</semantics>
</math> -->∣ input ∣ > 1 |text{input}| > 1∣ input ∣ > 1  the recursion: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              T
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mn>
             2
            </mn>
<mo>
             ×
            </mo>
<mtext>
             input
            </mtext>
<mo>
             ×
            </mo>
<msub>
<mi>
              T
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             −
            </mo>
<msub>
<mi>
              T
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            T_{n + 1}(text{input}) = 2 times text{input} times T_{n}(text{input}) - T_{n - 1}(text{input})
           </annotation>
</semantics>
</math> -->
T n + 1 ( input ) = 2 × input × T n ( input ) − T n − 1 ( input ) T_{n + 1}(text{input}) = 2 times text{input} times T_{n}(text{input}) - T_{n - 1}(text{input})

T n + 1 ​ ( input ) = 2 × input × T n ​ ( input ) − T n − 1 ​ ( input )

is evaluated. Otherwise, the explicit trigonometric formula: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              T
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mtext>
             cos
            </mtext>
<mo stretchy="false">
             (
            </mo>
<mi>
             n
            </mi>
<mo>
             ×
            </mo>
<mtext>
             arccos
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
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            T_{n}(text{input}) = text{cos}(n times text{arccos}(x))
           </annotation>
</semantics>
</math> -->
T n ( input ) = cos ( n × arccos ( x ) ) T_{n}(text{input}) = text{cos}(n times text{arccos}(x))

T n ​ ( input ) = cos ( n × arccos ( x ))

is evaluated. 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. chebyshev_polynomial_u ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Chebyshev polynomial of the second kind <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              U
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            U_{n}(text{input})
           </annotation>
</semantics>
</math> -->U n ( input ) U_{n}(text{input})U n ​ ( input )  . 

If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 0
           </annotation>
</semantics>
</math> -->n = 0 n = 0n = 0  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  is returned. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 1
           </annotation>
</semantics>
</math> -->n = 1 n = 1n = 1  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             2
            </mn>
<mo>
             ×
            </mo>
<mtext>
             input
            </mtext>
</mrow>
<annotation encoding="application/x-tex">
            2 times text{input}
           </annotation>
</semantics>
</math> -->2 × input 2 times text{input}2 × input  is returned. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             &lt;
            </mo>
<mn>
             6
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n &lt; 6
           </annotation>
</semantics>
</math> -->n < 6 n < 6n < 6  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             ∣
            </mi>
<mtext>
             input
            </mtext>
<mi mathvariant="normal">
             ∣
            </mi>
<mo>
             &gt;
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            |text{input}| &gt; 1
           </annotation>
</semantics>
</math> -->∣ input ∣ > 1 |text{input}| > 1∣ input ∣ > 1  , the recursion: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              U
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mn>
             2
            </mn>
<mo>
             ×
            </mo>
<mtext>
             input
            </mtext>
<mo>
             ×
            </mo>
<msub>
<mi>
              U
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             −
            </mo>
<msub>
<mi>
              U
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            U_{n + 1}(text{input}) = 2 times text{input} times U_{n}(text{input}) - U_{n - 1}(text{input})
           </annotation>
</semantics>
</math> -->
U n + 1 ( input ) = 2 × input × U n ( input ) − U n − 1 ( input ) U_{n + 1}(text{input}) = 2 times text{input} times U_{n}(text{input}) - U_{n - 1}(text{input})

U n + 1 ​ ( input ) = 2 × input × U n ​ ( input ) − U n − 1 ​ ( input )

is evaluated. Otherwise, the explicit trigonometric formula: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mrow>
<mtext>
               sin
              </mtext>
<mo stretchy="false">
               (
              </mo>
<mo stretchy="false">
               (
              </mo>
<mi>
               n
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
<mo>
               ×
              </mo>
<mtext>
               arccos
              </mtext>
<mo stretchy="false">
               (
              </mo>
<mtext>
               input
              </mtext>
<mo stretchy="false">
               )
              </mo>
<mo stretchy="false">
               )
              </mo>
</mrow>
<mrow>
<mtext>
               sin
              </mtext>
<mo stretchy="false">
               (
              </mo>
<mtext>
               arccos
              </mtext>
<mo stretchy="false">
               (
              </mo>
<mtext>
               input
              </mtext>
<mo stretchy="false">
               )
              </mo>
<mo stretchy="false">
               )
              </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            frac{text{sin}((n + 1) times text{arccos}(text{input}))}{text{sin}(text{arccos}(text{input}))}
           </annotation>
</semantics>
</math> -->
sin ( ( n + 1 ) × arccos ( input ) ) sin ( arccos ( input ) ) frac{text{sin}((n + 1) times text{arccos}(text{input}))}{text{sin}(text{arccos}(text{input}))}

sin ( arccos ( input )) sin (( n + 1 ) × arccos ( input )) ​

is evaluated. 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. chebyshev_polynomial_v ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Chebyshev polynomial of the third kind <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
              V
             </mi>
<mi>
              n
             </mi>
<mo lspace="0em" rspace="0em">
              ∗
             </mo>
</msubsup>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            V_{n}^{ast}(text{input})
           </annotation>
</semantics>
</math> -->V n ∗ ( input ) V_{n}^{ast}(text{input})V n ∗ ​ ( input )  . 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. chebyshev_polynomial_w ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Chebyshev polynomial of the fourth kind <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
              W
             </mi>
<mi>
              n
             </mi>
<mo lspace="0em" rspace="0em">
              ∗
             </mo>
</msubsup>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            W_{n}^{ast}(text{input})
           </annotation>
</semantics>
</math> -->W n ∗ ( input ) W_{n}^{ast}(text{input})W n ∗ ​ ( input )  . 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. digamma ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the logarithmic derivative of the gamma function on *input* . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             ϝ
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
<mi>
              d
             </mi>
<mrow>
<mi>
               d
              </mi>
<mi>
               x
              </mi>
</mrow>
</mfrac>
<mi>
             ln
            </mi>
<mo>
             ⁡
            </mo>
<mrow>
<mo fence="true">
              (
             </mo>
<mi mathvariant="normal">
              Γ
             </mi>
<mrow>
<mo fence="true">
               (
              </mo>
<mi>
               x
              </mi>
<mo fence="true">
               )
              </mo>
</mrow>
<mo fence="true">
              )
             </mo>
</mrow>
<mo>
             =
            </mo>
<mfrac>
<mrow>
<msup>
<mi mathvariant="normal">
                Γ
               </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                ′
               </mo>
</msup>
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
<mrow>
<mi mathvariant="normal">
               Γ
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
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            digamma(x) = frac{d}{dx} lnleft(Gammaleft(xright)right) = frac{Gamma'(x)}{Gamma(x)}
           </annotation>
</semantics>
</math> -->
ϝ ( x ) = d d x ln ⁡ ( Γ ( x ) ) = Γ ′ ( x ) Γ ( x ) digamma(x) = frac{d}{dx} lnleft(Gammaleft(xright)right) = frac{Gamma'(x)}{Gamma(x)}

ϝ ( x ) = d x d ​ ln ( Γ ( x ) ) = Γ ( x ) Γ ′ ( x ) ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to compute the digamma function on

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Note 

This function is similar to SciPy’s *scipy.special.digamma* .

Note 

From PyTorch 1.8 onwards, the digamma function returns *-Inf* for *0* .
Previously it returned *NaN* for *0* .

Example: 

```
>>> a = torch.tensor([1, 0.5])
>>> torch.special.digamma(a)
tensor([-0.5772, -1.9635])

```

torch.special. entr ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the entropy on `input`  (as defined below), elementwise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right" columnspacing="" rowspacing="0.25em">
<mtr>
<mtd class="mtr-glue">
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mtext>
                 entr(x)
                </mtext>
<mo>
                 =
                </mo>
<mrow>
<mo fence="true">
                  {
                 </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mo>
                       −
                      </mo>
<mi>
                       x
                      </mi>
<mo>
                       ∗
                      </mo>
<mi>
                       ln
                      </mi>
<mo>
                       ⁡
                      </mo>
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
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                       x
                      </mi>
<mo>
                       &gt;
                      </mo>
<mn>
                       0
                      </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mn>
                      0
                     </mn>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                       x
                      </mi>
<mo>
                       =
                      </mo>
<mn>
                       0.0
                      </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mo>
                       −
                      </mo>
<mi mathvariant="normal">
                       ∞
                      </mi>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                       x
                      </mi>
<mo>
                       &lt;
                      </mo>
<mn>
                       0
                      </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
</mstyle>
</mtd>
<mtd class="mtr-glue">
</mtd>
<mtd class="mml-eqn-num">
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{align}
text{entr(x)} = begin{cases}
    -x * ln(x)  &amp; x &gt; 0 
    0 &amp;  x = 0.0 
    -infty &amp; x &lt; 0
end{cases}
end{align}
           </annotation>
</semantics>
</math> -->
entr(x) = { − x ∗ ln ⁡ ( x ) x > 0 0 x = 0.0 − ∞ x < 0 begin{align}
text{entr(x)} = begin{cases}
 -x * ln(x) & x > 0 
 0 & x = 0.0 
 -infty & x < 0
end{cases}
end{align}

entr(x) = ⎩ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎨ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎧ ​ − x ∗ ln ( x ) 0 − ∞ ​ x > 0 x = 0.0 x < 0 ​ ​ ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.arange(-0.5, 1, 0.5)
>>> a
tensor([-0.5000,  0.0000,  0.5000])
>>> torch.special.entr(a)
tensor([  -inf, 0.0000, 0.3466])

```

torch.special. erf ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the error function of `input`  . The error function is defined as follows: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mi mathvariant="normal">
              e
             </mi>
<mi mathvariant="normal">
              r
             </mi>
<mi mathvariant="normal">
              f
             </mi>
</mrow>
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
              2
             </mn>
<msqrt>
<mi>
               π
              </mi>
</msqrt>
</mfrac>
<msubsup>
<mo>
              ∫
             </mo>
<mn>
              0
             </mn>
<mi>
              x
             </mi>
</msubsup>
<msup>
<mi>
              e
             </mi>
<mrow>
<mo>
               −
              </mo>
<msup>
<mi>
                t
               </mi>
<mn>
                2
               </mn>
</msup>
</mrow>
</msup>
<mi>
             d
            </mi>
<mi>
             t
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            mathrm{erf}(x) = frac{2}{sqrt{pi}} int_{0}^{x} e^{-t^2} dt
           </annotation>
</semantics>
</math> -->
e r f ( x ) = 2 π ∫ 0 x e − t 2 d t mathrm{erf}(x) = frac{2}{sqrt{pi}} int_{0}^{x} e^{-t^2} dt

erf ( x ) = π ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 2 ​ ∫ 0 x ​ e − t 2 d t

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.erf(torch.tensor([0, -1., 10.]))
tensor([ 0.0000, -0.8427,  1.0000])

```

torch.special. erfc ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the complementary error function of `input`  .
The complementary error function is defined as follows: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mi mathvariant="normal">
              e
             </mi>
<mi mathvariant="normal">
              r
             </mi>
<mi mathvariant="normal">
              f
             </mi>
<mi mathvariant="normal">
              c
             </mi>
</mrow>
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
             1
            </mn>
<mo>
             −
            </mo>
<mfrac>
<mn>
              2
             </mn>
<msqrt>
<mi>
               π
              </mi>
</msqrt>
</mfrac>
<msubsup>
<mo>
              ∫
             </mo>
<mn>
              0
             </mn>
<mi>
              x
             </mi>
</msubsup>
<msup>
<mi>
              e
             </mi>
<mrow>
<mo>
               −
              </mo>
<msup>
<mi>
                t
               </mi>
<mn>
                2
               </mn>
</msup>
</mrow>
</msup>
<mi>
             d
            </mi>
<mi>
             t
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            mathrm{erfc}(x) = 1 - frac{2}{sqrt{pi}} int_{0}^{x} e^{-t^2} dt
           </annotation>
</semantics>
</math> -->
e r f c ( x ) = 1 − 2 π ∫ 0 x e − t 2 d t mathrm{erfc}(x) = 1 - frac{2}{sqrt{pi}} int_{0}^{x} e^{-t^2} dt

erfc ( x ) = 1 − π ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 2 ​ ∫ 0 x ​ e − t 2 d t

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.erfc(torch.tensor([0, -1., 10.]))
tensor([ 1.0000, 1.8427,  0.0000])

```

torch.special. erfcx ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the scaled complementary error function for each element of `input`  .
The scaled complementary error function is defined as follows: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mi mathvariant="normal">
              e
             </mi>
<mi mathvariant="normal">
              r
             </mi>
<mi mathvariant="normal">
              f
             </mi>
<mi mathvariant="normal">
              c
             </mi>
<mi mathvariant="normal">
              x
             </mi>
</mrow>
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
<msup>
<mi>
              e
             </mi>
<msup>
<mi>
               x
              </mi>
<mn>
               2
              </mn>
</msup>
</msup>
<mrow>
<mi mathvariant="normal">
              e
             </mi>
<mi mathvariant="normal">
              r
             </mi>
<mi mathvariant="normal">
              f
             </mi>
<mi mathvariant="normal">
              c
             </mi>
</mrow>
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
            mathrm{erfcx}(x) = e^{x^2} mathrm{erfc}(x)
           </annotation>
</semantics>
</math> -->
e r f c x ( x ) = e x 2 e r f c ( x ) mathrm{erfcx}(x) = e^{x^2} mathrm{erfc}(x)

erfcx ( x ) = e x 2 erfc ( x )

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.erfcx(torch.tensor([0, -1., 10.]))
tensor([ 1.0000, 5.0090, 0.0561])

```

torch.special. erfinv ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the inverse error function of `input`  .
The inverse error function is defined in the range <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<mn>
             1
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
            (-1, 1)
           </annotation>
</semantics>
</math> -->( − 1 , 1 ) (-1, 1)( − 1 , 1 )  as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mi mathvariant="normal">
              e
             </mi>
<mi mathvariant="normal">
              r
             </mi>
<mi mathvariant="normal">
              f
             </mi>
<mi mathvariant="normal">
              i
             </mi>
<mi mathvariant="normal">
              n
             </mi>
<mi mathvariant="normal">
              v
             </mi>
</mrow>
<mo stretchy="false">
             (
            </mo>
<mrow>
<mi mathvariant="normal">
              e
             </mi>
<mi mathvariant="normal">
              r
             </mi>
<mi mathvariant="normal">
              f
             </mi>
</mrow>
<mo stretchy="false">
             (
            </mo>
<mi>
             x
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mi>
             x
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            mathrm{erfinv}(mathrm{erf}(x)) = x
           </annotation>
</semantics>
</math> -->
e r f i n v ( e r f ( x ) ) = x mathrm{erfinv}(mathrm{erf}(x)) = x

erfinv ( erf ( x )) = x

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.erfinv(torch.tensor([0, 0.5, -1.]))
tensor([ 0.0000,  0.4769,    -inf])

```

torch.special. exp2 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the base two exponential function of `input`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              y
             </mi>
<mi>
              i
             </mi>
</msub>
<mo>
             =
            </mo>
<msup>
<mn>
              2
             </mn>
<msub>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
</msub>
</msup>
</mrow>
<annotation encoding="application/x-tex">
            y_{i} = 2^{x_{i}}
           </annotation>
</semantics>
</math> -->
y i = 2 x i y_{i} = 2^{x_{i}}

y i ​ = 2 x i ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.exp2(torch.tensor([0, math.log2(2.), 3, 4]))
tensor([ 1.,  2.,  8., 16.])

```

torch.special. expit ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the expit (also known as the logistic sigmoid function) of the elements of `input`  . 

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
<mfrac>
<mn>
              1
             </mn>
<mrow>
<mn>
               1
              </mn>
<mo>
               +
              </mo>
<msup>
<mi>
                e
               </mi>
<mrow>
<mo>
                 −
                </mo>
<msub>
<mtext>
                  input
                 </mtext>
<mi>
                  i
                 </mi>
</msub>
</mrow>
</msup>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = frac{1}{1 + e^{-text{input}_{i}}}
           </annotation>
</semantics>
</math> -->
out i = 1 1 + e − input i text{out}_{i} = frac{1}{1 + e^{-text{input}_{i}}}

out i ​ = 1 + e − input i ​ 1 ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> t = torch.randn(4)
>>> t
tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
>>> torch.special.expit(t)
tensor([ 0.7153,  0.7481,  0.2920,  0.1458])

```

torch.special. expm1 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the exponential of the elements minus 1
of `input`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              y
             </mi>
<mi>
              i
             </mi>
</msub>
<mo>
             =
            </mo>
<msup>
<mi>
              e
             </mi>
<msub>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
</msub>
</msup>
<mo>
             −
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            y_{i} = e^{x_{i}} - 1
           </annotation>
</semantics>
</math> -->
y i = e x i − 1 y_{i} = e^{x_{i}} - 1

y i ​ = e x i ​ − 1

Note 

This function provides greater precision than exp(x) - 1 for small values of x.

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.expm1(torch.tensor([0, math.log(2.)]))
tensor([ 0.,  1.])

```

torch.special. gammainc ( *input*  , *other*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the regularized lower incomplete gamma function: 

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
<mfrac>
<mn>
              1
             </mn>
<mrow>
<mi mathvariant="normal">
               Γ
              </mi>
<mo stretchy="false">
               (
              </mo>
<msub>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
</msub>
<mo stretchy="false">
               )
              </mo>
</mrow>
</mfrac>
<msubsup>
<mo>
              ∫
             </mo>
<mn>
              0
             </mn>
<msub>
<mtext>
               other
              </mtext>
<mi>
               i
              </mi>
</msub>
</msubsup>
<msup>
<mi>
              t
             </mi>
<mrow>
<msub>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
</msub>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msup>
<msup>
<mi>
              e
             </mi>
<mrow>
<mo>
               −
              </mo>
<mi>
               t
              </mi>
</mrow>
</msup>
<mi>
             d
            </mi>
<mi>
             t
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = frac{1}{Gamma(text{input}_i)} int_0^{text{other}_i} t^{text{input}_i-1} e^{-t} dt
           </annotation>
</semantics>
</math> -->
out i = 1 Γ ( input i ) ∫ 0 other i t input i − 1 e − t d t text{out}_{i} = frac{1}{Gamma(text{input}_i)} int_0^{text{other}_i} t^{text{input}_i-1} e^{-t} dt

out i ​ = Γ ( input i ​ ) 1 ​ ∫ 0 other i ​ ​ t input i ​ − 1 e − t d t

where both <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            text{input}_i
           </annotation>
</semantics>
</math> -->input i text{input}_iinput i ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
              other
             </mtext>
<mi>
              i
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            text{other}_i
           </annotation>
</semantics>
</math> -->other i text{other}_iother i ​  are weakly positive
and at least one is strictly positive.
If both are zero or either is negative then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mtext>
             nan
            </mtext>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_i=text{nan}
           </annotation>
</semantics>
</math> -->out i = nan text{out}_i=text{nan}out i ​ = nan  . <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             Γ
            </mi>
<mo stretchy="false">
             (
            </mo>
<mo>
             ⋅
            </mo>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            Gamma(cdot)
           </annotation>
</semantics>
</math> -->Γ ( ⋅ ) Gamma(cdot)Γ ( ⋅ )  in the equation above is the gamma function, 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             Γ
            </mi>
<mo stretchy="false">
             (
            </mo>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<msubsup>
<mo>
              ∫
             </mo>
<mn>
              0
             </mn>
<mi mathvariant="normal">
              ∞
             </mi>
</msubsup>
<msup>
<mi>
              t
             </mi>
<mrow>
<mo stretchy="false">
               (
              </mo>
<msub>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
</msub>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
</mrow>
</msup>
<msup>
<mi>
              e
             </mi>
<mrow>
<mo>
               −
              </mo>
<mi>
               t
              </mi>
</mrow>
</msup>
<mi>
             d
            </mi>
<mi>
             t
            </mi>
<mi mathvariant="normal">
             .
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            Gamma(text{input}_i) = int_0^infty t^{(text{input}_i-1)} e^{-t} dt.
           </annotation>
</semantics>
</math> -->
Γ ( input i ) = ∫ 0 ∞ t ( input i − 1 ) e − t d t . Gamma(text{input}_i) = int_0^infty t^{(text{input}_i-1)} e^{-t} dt.

Γ ( input i ​ ) = ∫ 0 ∞ ​ t ( input i ​ − 1 ) e − t d t .

See [`torch.special.gammaincc()`](#torch.special.gammaincc "torch.special.gammaincc")  and [`torch.special.gammaln()`](#torch.special.gammaln "torch.special.gammaln")  for related functions. 

Supports [broadcasting to a common shape](notes/broadcasting.html#broadcasting-semantics)  and float inputs. 

Note 

The backward pass with respect to `input`  is not yet supported.
Please open an issue on PyTorch’s Github to request it.

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the first non-negative input tensor
* **other** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the second non-negative input tensor

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a1 = torch.tensor([4.0])
>>> a2 = torch.tensor([3.0, 4.0, 5.0])
>>> a = torch.special.gammaincc(a1, a2)
tensor([0.3528, 0.5665, 0.7350])
tensor([0.3528, 0.5665, 0.7350])
>>> b = torch.special.gammainc(a1, a2) + torch.special.gammaincc(a1, a2)
tensor([1., 1., 1.])

```

torch.special. gammaincc ( *input*  , *other*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the regularized upper incomplete gamma function: 

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
<mfrac>
<mn>
              1
             </mn>
<mrow>
<mi mathvariant="normal">
               Γ
              </mi>
<mo stretchy="false">
               (
              </mo>
<msub>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
</msub>
<mo stretchy="false">
               )
              </mo>
</mrow>
</mfrac>
<msubsup>
<mo>
              ∫
             </mo>
<msub>
<mtext>
               other
              </mtext>
<mi>
               i
              </mi>
</msub>
<mi mathvariant="normal">
              ∞
             </mi>
</msubsup>
<msup>
<mi>
              t
             </mi>
<mrow>
<msub>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
</msub>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msup>
<msup>
<mi>
              e
             </mi>
<mrow>
<mo>
               −
              </mo>
<mi>
               t
              </mi>
</mrow>
</msup>
<mi>
             d
            </mi>
<mi>
             t
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = frac{1}{Gamma(text{input}_i)} int_{text{other}_i}^{infty} t^{text{input}_i-1} e^{-t} dt
           </annotation>
</semantics>
</math> -->
out i = 1 Γ ( input i ) ∫ other i ∞ t input i − 1 e − t d t text{out}_{i} = frac{1}{Gamma(text{input}_i)} int_{text{other}_i}^{infty} t^{text{input}_i-1} e^{-t} dt

out i ​ = Γ ( input i ​ ) 1 ​ ∫ other i ​ ∞ ​ t input i ​ − 1 e − t d t

where both <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            text{input}_i
           </annotation>
</semantics>
</math> -->input i text{input}_iinput i ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
              other
             </mtext>
<mi>
              i
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            text{other}_i
           </annotation>
</semantics>
</math> -->other i text{other}_iother i ​  are weakly positive
and at least one is strictly positive.
If both are zero or either is negative then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mtext>
             nan
            </mtext>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_i=text{nan}
           </annotation>
</semantics>
</math> -->out i = nan text{out}_i=text{nan}out i ​ = nan  . <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             Γ
            </mi>
<mo stretchy="false">
             (
            </mo>
<mo>
             ⋅
            </mo>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            Gamma(cdot)
           </annotation>
</semantics>
</math> -->Γ ( ⋅ ) Gamma(cdot)Γ ( ⋅ )  in the equation above is the gamma function, 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             Γ
            </mi>
<mo stretchy="false">
             (
            </mo>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<msubsup>
<mo>
              ∫
             </mo>
<mn>
              0
             </mn>
<mi mathvariant="normal">
              ∞
             </mi>
</msubsup>
<msup>
<mi>
              t
             </mi>
<mrow>
<mo stretchy="false">
               (
              </mo>
<msub>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
</msub>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
</mrow>
</msup>
<msup>
<mi>
              e
             </mi>
<mrow>
<mo>
               −
              </mo>
<mi>
               t
              </mi>
</mrow>
</msup>
<mi>
             d
            </mi>
<mi>
             t
            </mi>
<mi mathvariant="normal">
             .
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            Gamma(text{input}_i) = int_0^infty t^{(text{input}_i-1)} e^{-t} dt.
           </annotation>
</semantics>
</math> -->
Γ ( input i ) = ∫ 0 ∞ t ( input i − 1 ) e − t d t . Gamma(text{input}_i) = int_0^infty t^{(text{input}_i-1)} e^{-t} dt.

Γ ( input i ​ ) = ∫ 0 ∞ ​ t ( input i ​ − 1 ) e − t d t .

See [`torch.special.gammainc()`](#torch.special.gammainc "torch.special.gammainc")  and [`torch.special.gammaln()`](#torch.special.gammaln "torch.special.gammaln")  for related functions. 

Supports [broadcasting to a common shape](notes/broadcasting.html#broadcasting-semantics)  and float inputs. 

Note 

The backward pass with respect to `input`  is not yet supported.
Please open an issue on PyTorch’s Github to request it.

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the first non-negative input tensor
* **other** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the second non-negative input tensor

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a1 = torch.tensor([4.0])
>>> a2 = torch.tensor([3.0, 4.0, 5.0])
>>> a = torch.special.gammaincc(a1, a2)
tensor([0.6472, 0.4335, 0.2650])
>>> b = torch.special.gammainc(a1, a2) + torch.special.gammaincc(a1, a2)
tensor([1., 1., 1.])

```

torch.special. gammaln ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the natural logarithm of the absolute value of the gamma function on `input`  . 

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
<mi>
             ln
            </mi>
<mo>
             ⁡
            </mo>
<mi mathvariant="normal">
             Γ
            </mi>
<mo stretchy="false">
             (
            </mo>
<mi mathvariant="normal">
             ∣
            </mi>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
<mi mathvariant="normal">
             ∣
            </mi>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = ln Gamma(|text{input}_{i}|)
           </annotation>
</semantics>
</math> -->
out i = ln ⁡ Γ ( ∣ input i ∣ ) text{out}_{i} = ln Gamma(|text{input}_{i}|)

out i ​ = ln Γ ( ∣ input i ​ ∣ )

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.arange(0.5, 2, 0.5)
>>> torch.special.gammaln(a)
tensor([ 0.5724,  0.0000, -0.1208])

```

torch.special. hermite_polynomial_h ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Physicist’s Hermite polynomial <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              H
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            H_{n}(text{input})
           </annotation>
</semantics>
</math> -->H n ( input ) H_{n}(text{input})H n ​ ( input )  . 

If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 0
           </annotation>
</semantics>
</math> -->n = 0 n = 0n = 0  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  is returned. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 1
           </annotation>
</semantics>
</math> -->n = 1 n = 1n = 1  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             input
            </mtext>
</mrow>
<annotation encoding="application/x-tex">
            text{input}
           </annotation>
</semantics>
</math> -->input text{input}input  is returned. Otherwise, the recursion: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              H
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mn>
             2
            </mn>
<mo>
             ×
            </mo>
<mtext>
             input
            </mtext>
<mo>
             ×
            </mo>
<msub>
<mi>
              H
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             −
            </mo>
<msub>
<mi>
              H
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            H_{n + 1}(text{input}) = 2 times text{input} times H_{n}(text{input}) - H_{n - 1}(text{input})
           </annotation>
</semantics>
</math> -->
H n + 1 ( input ) = 2 × input × H n ( input ) − H n − 1 ( input ) H_{n + 1}(text{input}) = 2 times text{input} times H_{n}(text{input}) - H_{n - 1}(text{input})

H n + 1 ​ ( input ) = 2 × input × H n ​ ( input ) − H n − 1 ​ ( input )

is evaluated. 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. hermite_polynomial_he ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Probabilist’s Hermite polynomial <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             H
            </mi>
<msub>
<mi>
              e
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            He_{n}(text{input})
           </annotation>
</semantics>
</math> -->H e n ( input ) He_{n}(text{input})H e n ​ ( input )  . 

If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 0
           </annotation>
</semantics>
</math> -->n = 0 n = 0n = 0  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  is returned. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 1
           </annotation>
</semantics>
</math> -->n = 1 n = 1n = 1  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             input
            </mtext>
</mrow>
<annotation encoding="application/x-tex">
            text{input}
           </annotation>
</semantics>
</math> -->input text{input}input  is returned. Otherwise, the recursion: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             H
            </mi>
<msub>
<mi>
              e
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mn>
             2
            </mn>
<mo>
             ×
            </mo>
<mtext>
             input
            </mtext>
<mo>
             ×
            </mo>
<mi>
             H
            </mi>
<msub>
<mi>
              e
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             −
            </mo>
<mi>
             H
            </mi>
<msub>
<mi>
              e
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            He_{n + 1}(text{input}) = 2 times text{input} times He_{n}(text{input}) - He_{n - 1}(text{input})
           </annotation>
</semantics>
</math> -->
H e n + 1 ( input ) = 2 × input × H e n ( input ) − H e n − 1 ( input ) He_{n + 1}(text{input}) = 2 times text{input} times He_{n}(text{input}) - He_{n - 1}(text{input})

H e n + 1 ​ ( input ) = 2 × input × H e n ​ ( input ) − H e n − 1 ​ ( input )

is evaluated. 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. i0 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the zeroth order modified Bessel function of the first kind for each element of `input`  . 

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
<msub>
<mi>
              I
             </mi>
<mn>
              0
             </mn>
</msub>
<mo stretchy="false">
             (
            </mo>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<munderover>
<mo>
              ∑
             </mo>
<mrow>
<mi>
               k
              </mi>
<mo>
               =
              </mo>
<mn>
               0
              </mn>
</mrow>
<mi mathvariant="normal">
              ∞
             </mi>
</munderover>
<mfrac>
<mrow>
<mo stretchy="false">
               (
              </mo>
<msubsup>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
<mn>
                2
               </mn>
</msubsup>
<mi mathvariant="normal">
               /
              </mi>
<mn>
               4
              </mn>
<msup>
<mo stretchy="false">
                )
               </mo>
<mi>
                k
               </mi>
</msup>
</mrow>
<mrow>
<mo stretchy="false">
               (
              </mo>
<mi>
               k
              </mi>
<mo stretchy="false">
               !
              </mo>
<msup>
<mo stretchy="false">
                )
               </mo>
<mn>
                2
               </mn>
</msup>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = I_0(text{input}_{i}) = sum_{k=0}^{infty} frac{(text{input}_{i}^2/4)^k}{(k!)^2}
           </annotation>
</semantics>
</math> -->
out i = I 0 ( input i ) = ∑ k = 0 ∞ ( input i 2 / 4 ) k ( k ! ) 2 text{out}_{i} = I_0(text{input}_{i}) = sum_{k=0}^{infty} frac{(text{input}_{i}^2/4)^k}{(k!)^2}

out i ​ = I 0 ​ ( input i ​ ) = k = 0 ∑ ∞ ​ ( k ! ) 2 ( input i 2 ​ /4 ) k ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.i0(torch.arange(5, dtype=torch.float32))
tensor([ 1.0000,  1.2661,  2.2796,  4.8808, 11.3019])

```

torch.special. i0e ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below)
for each element of `input`  . 

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
<mi>
             exp
            </mi>
<mo>
             ⁡
            </mo>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<mi mathvariant="normal">
             ∣
            </mi>
<mi>
             x
            </mi>
<mi mathvariant="normal">
             ∣
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo>
             ∗
            </mo>
<mi>
             i
            </mi>
<mn>
             0
            </mn>
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
             exp
            </mi>
<mo>
             ⁡
            </mo>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<mi mathvariant="normal">
             ∣
            </mi>
<mi>
             x
            </mi>
<mi mathvariant="normal">
             ∣
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo>
             ∗
            </mo>
<munderover>
<mo>
              ∑
             </mo>
<mrow>
<mi>
               k
              </mi>
<mo>
               =
              </mo>
<mn>
               0
              </mn>
</mrow>
<mi mathvariant="normal">
              ∞
             </mi>
</munderover>
<mfrac>
<mrow>
<mo stretchy="false">
               (
              </mo>
<msubsup>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
<mn>
                2
               </mn>
</msubsup>
<mi mathvariant="normal">
               /
              </mi>
<mn>
               4
              </mn>
<msup>
<mo stretchy="false">
                )
               </mo>
<mi>
                k
               </mi>
</msup>
</mrow>
<mrow>
<mo stretchy="false">
               (
              </mo>
<mi>
               k
              </mi>
<mo stretchy="false">
               !
              </mo>
<msup>
<mo stretchy="false">
                )
               </mo>
<mn>
                2
               </mn>
</msup>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = exp(-|x|) * i0(x) = exp(-|x|) * sum_{k=0}^{infty} frac{(text{input}_{i}^2/4)^k}{(k!)^2}
           </annotation>
</semantics>
</math> -->
out i = exp ⁡ ( − ∣ x ∣ ) ∗ i 0 ( x ) = exp ⁡ ( − ∣ x ∣ ) ∗ ∑ k = 0 ∞ ( input i 2 / 4 ) k ( k ! ) 2 text{out}_{i} = exp(-|x|) * i0(x) = exp(-|x|) * sum_{k=0}^{infty} frac{(text{input}_{i}^2/4)^k}{(k!)^2}

out i ​ = exp ( − ∣ x ∣ ) ∗ i 0 ( x ) = exp ( − ∣ x ∣ ) ∗ k = 0 ∑ ∞ ​ ( k ! ) 2 ( input i 2 ​ /4 ) k ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.i0e(torch.arange(5, dtype=torch.float32))
tensor([1.0000, 0.4658, 0.3085, 0.2430, 0.2070])

```

torch.special. i1 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the first order modified Bessel function of the first kind (as defined below)
for each element of `input`  . 

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
<mfrac>
<mrow>
<mo stretchy="false">
               (
              </mo>
<msub>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
</msub>
<mo stretchy="false">
               )
              </mo>
</mrow>
<mn>
              2
             </mn>
</mfrac>
<mo>
             ∗
            </mo>
<munderover>
<mo>
              ∑
             </mo>
<mrow>
<mi>
               k
              </mi>
<mo>
               =
              </mo>
<mn>
               0
              </mn>
</mrow>
<mi mathvariant="normal">
              ∞
             </mi>
</munderover>
<mfrac>
<mrow>
<mo stretchy="false">
               (
              </mo>
<msubsup>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
<mn>
                2
               </mn>
</msubsup>
<mi mathvariant="normal">
               /
              </mi>
<mn>
               4
              </mn>
<msup>
<mo stretchy="false">
                )
               </mo>
<mi>
                k
               </mi>
</msup>
</mrow>
<mrow>
<mo stretchy="false">
               (
              </mo>
<mi>
               k
              </mi>
<mo stretchy="false">
               !
              </mo>
<mo stretchy="false">
               )
              </mo>
<mo>
               ∗
              </mo>
<mo stretchy="false">
               (
              </mo>
<mi>
               k
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
<mo stretchy="false">
               !
              </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = frac{(text{input}_{i})}{2} * sum_{k=0}^{infty} frac{(text{input}_{i}^2/4)^k}{(k!) * (k+1)!}
           </annotation>
</semantics>
</math> -->
out i = ( input i ) 2 ∗ ∑ k = 0 ∞ ( input i 2 / 4 ) k ( k ! ) ∗ ( k + 1 ) ! text{out}_{i} = frac{(text{input}_{i})}{2} * sum_{k=0}^{infty} frac{(text{input}_{i}^2/4)^k}{(k!) * (k+1)!}

out i ​ = 2 ( input i ​ ) ​ ∗ k = 0 ∑ ∞ ​ ( k !) ∗ ( k + 1 )! ( input i 2 ​ /4 ) k ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.i1(torch.arange(5, dtype=torch.float32))
tensor([0.0000, 0.5652, 1.5906, 3.9534, 9.7595])

```

torch.special. i1e ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below)
for each element of `input`  . 

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
<mi>
             exp
            </mi>
<mo>
             ⁡
            </mo>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<mi mathvariant="normal">
             ∣
            </mi>
<mi>
             x
            </mi>
<mi mathvariant="normal">
             ∣
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo>
             ∗
            </mo>
<mi>
             i
            </mi>
<mn>
             1
            </mn>
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
             exp
            </mi>
<mo>
             ⁡
            </mo>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<mi mathvariant="normal">
             ∣
            </mi>
<mi>
             x
            </mi>
<mi mathvariant="normal">
             ∣
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo>
             ∗
            </mo>
<mfrac>
<mrow>
<mo stretchy="false">
               (
              </mo>
<msub>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
</msub>
<mo stretchy="false">
               )
              </mo>
</mrow>
<mn>
              2
             </mn>
</mfrac>
<mo>
             ∗
            </mo>
<munderover>
<mo>
              ∑
             </mo>
<mrow>
<mi>
               k
              </mi>
<mo>
               =
              </mo>
<mn>
               0
              </mn>
</mrow>
<mi mathvariant="normal">
              ∞
             </mi>
</munderover>
<mfrac>
<mrow>
<mo stretchy="false">
               (
              </mo>
<msubsup>
<mtext>
                input
               </mtext>
<mi>
                i
               </mi>
<mn>
                2
               </mn>
</msubsup>
<mi mathvariant="normal">
               /
              </mi>
<mn>
               4
              </mn>
<msup>
<mo stretchy="false">
                )
               </mo>
<mi>
                k
               </mi>
</msup>
</mrow>
<mrow>
<mo stretchy="false">
               (
              </mo>
<mi>
               k
              </mi>
<mo stretchy="false">
               !
              </mo>
<mo stretchy="false">
               )
              </mo>
<mo>
               ∗
              </mo>
<mo stretchy="false">
               (
              </mo>
<mi>
               k
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
<mo stretchy="false">
               !
              </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = exp(-|x|) * i1(x) =
    exp(-|x|) * frac{(text{input}_{i})}{2} * sum_{k=0}^{infty} frac{(text{input}_{i}^2/4)^k}{(k!) * (k+1)!}
           </annotation>
</semantics>
</math> -->
out i = exp ⁡ ( − ∣ x ∣ ) ∗ i 1 ( x ) = exp ⁡ ( − ∣ x ∣ ) ∗ ( input i ) 2 ∗ ∑ k = 0 ∞ ( input i 2 / 4 ) k ( k ! ) ∗ ( k + 1 ) ! text{out}_{i} = exp(-|x|) * i1(x) =
 exp(-|x|) * frac{(text{input}_{i})}{2} * sum_{k=0}^{infty} frac{(text{input}_{i}^2/4)^k}{(k!) * (k+1)!}

out i ​ = exp ( − ∣ x ∣ ) ∗ i 1 ( x ) = exp ( − ∣ x ∣ ) ∗ 2 ( input i ​ ) ​ ∗ k = 0 ∑ ∞ ​ ( k !) ∗ ( k + 1 )! ( input i 2 ​ /4 ) k ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.i1e(torch.arange(5, dtype=torch.float32))
tensor([0.0000, 0.2079, 0.2153, 0.1968, 0.1788])

```

torch.special. laguerre_polynomial_l ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Laguerre polynomial <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              L
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            L_{n}(text{input})
           </annotation>
</semantics>
</math> -->L n ( input ) L_{n}(text{input})L n ​ ( input )  . 

If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 0
           </annotation>
</semantics>
</math> -->n = 0 n = 0n = 0  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  is returned. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 1
           </annotation>
</semantics>
</math> -->n = 1 n = 1n = 1  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             input
            </mtext>
</mrow>
<annotation encoding="application/x-tex">
            text{input}
           </annotation>
</semantics>
</math> -->input text{input}input  is returned. Otherwise, the recursion: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              L
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mn>
             2
            </mn>
<mo>
             ×
            </mo>
<mtext>
             input
            </mtext>
<mo>
             ×
            </mo>
<msub>
<mi>
              L
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             −
            </mo>
<msub>
<mi>
              L
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            L_{n + 1}(text{input}) = 2 times text{input} times L_{n}(text{input}) - L_{n - 1}(text{input})
           </annotation>
</semantics>
</math> -->
L n + 1 ( input ) = 2 × input × L n ( input ) − L n − 1 ( input ) L_{n + 1}(text{input}) = 2 times text{input} times L_{n}(text{input}) - L_{n - 1}(text{input})

L n + 1 ​ ( input ) = 2 × input × L n ​ ( input ) − L n − 1 ​ ( input )

is evaluated. 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. legendre_polynomial_p ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Legendre polynomial <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              P
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            P_{n}(text{input})
           </annotation>
</semantics>
</math> -->P n ( input ) P_{n}(text{input})P n ​ ( input )  . 

If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 0
           </annotation>
</semantics>
</math> -->n = 0 n = 0n = 0  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  is returned. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             =
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n = 1
           </annotation>
</semantics>
</math> -->n = 1 n = 1n = 1  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             input
            </mtext>
</mrow>
<annotation encoding="application/x-tex">
            text{input}
           </annotation>
</semantics>
</math> -->input text{input}input  is returned. Otherwise, the recursion: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              P
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               +
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mn>
             2
            </mn>
<mo>
             ×
            </mo>
<mtext>
             input
            </mtext>
<mo>
             ×
            </mo>
<msub>
<mi>
              P
             </mi>
<mi>
              n
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             −
            </mo>
<msub>
<mi>
              P
             </mi>
<mrow>
<mi>
               n
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msub>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            P_{n + 1}(text{input}) = 2 times text{input} times P_{n}(text{input}) - P_{n - 1}(text{input})
           </annotation>
</semantics>
</math> -->
P n + 1 ( input ) = 2 × input × P n ( input ) − P n − 1 ( input ) P_{n + 1}(text{input}) = 2 times text{input} times P_{n}(text{input}) - P_{n - 1}(text{input})

P n + 1 ​ ( input ) = 2 × input × P n ​ ( input ) − P n − 1 ​ ( input )

is evaluated. 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. log1p ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Alias for [`torch.log1p()`](generated/torch.log1p.html#torch.log1p "torch.log1p")  .

torch.special. log_ndtr ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the log of the area under the standard Gaussian probability density function,
integrated from minus infinity to `input`  , elementwise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             log_ndtr
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
             log
            </mi>
<mo>
             ⁡
            </mo>
<mrow>
<mo fence="true">
              (
             </mo>
<mfrac>
<mn>
               1
              </mn>
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
</mfrac>
<msubsup>
<mo>
               ∫
              </mo>
<mrow>
<mo>
                −
               </mo>
<mi mathvariant="normal">
                ∞
               </mi>
</mrow>
<mi>
               x
              </mi>
</msubsup>
<msup>
<mi>
               e
              </mi>
<mrow>
<mo>
                −
               </mo>
<mfrac>
<mn>
                 1
                </mn>
<mn>
                 2
                </mn>
</mfrac>
<msup>
<mi>
                 t
                </mi>
<mn>
                 2
                </mn>
</msup>
</mrow>
</msup>
<mi>
              d
             </mi>
<mi>
              t
             </mi>
<mo fence="true">
              )
             </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
            text{log_ndtr}(x) = logleft(frac{1}{sqrt{2 pi}}int_{-infty}^{x} e^{-frac{1}{2}t^2} dt right)
           </annotation>
</semantics>
</math> -->
log_ndtr ( x ) = log ⁡ ( 1 2 π ∫ − ∞ x e − 1 2 t 2 d t ) text{log_ndtr}(x) = logleft(frac{1}{sqrt{2 pi}}int_{-infty}^{x} e^{-frac{1}{2}t^2} dt right)

log_ndtr ( x ) = lo g ( 2 π ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 1 ​ ∫ − ∞ x ​ e − 2 1 ​ t 2 d t )

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.log_ndtr(torch.tensor([-3., -2, -1, 0, 1, 2, 3]))
tensor([-6.6077 -3.7832 -1.841  -0.6931 -0.1728 -0.023  -0.0014])

```

torch.special. log_softmax ( *input*  , *dim*  , *** , *dtype = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes softmax followed by a logarithm. 

While mathematically equivalent to log(softmax(x)), doing these two
operations separately is slower and numerically unstable. This function
is computed as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             log_softmax
            </mtext>
<mo stretchy="false">
             (
            </mo>
<msub>
<mi>
              x
             </mi>
<mi>
              i
             </mi>
</msub>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mi>
             log
            </mi>
<mo>
             ⁡
            </mo>
<mrow>
<mo fence="true">
              (
             </mo>
<mfrac>
<mrow>
<mi>
                exp
               </mi>
<mo>
                ⁡
               </mo>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 x
                </mi>
<mi>
                 i
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<mrow>
<munder>
<mo>
                 ∑
                </mo>
<mi>
                 j
                </mi>
</munder>
<mi>
                exp
               </mi>
<mo>
                ⁡
               </mo>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 x
                </mi>
<mi>
                 j
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mfrac>
<mo fence="true">
              )
             </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
            text{log_softmax}(x_{i}) = logleft(frac{exp(x_i) }{ sum_j exp(x_j)} right)
           </annotation>
</semantics>
</math> -->
log_softmax ( x i ) = log ⁡ ( exp ⁡ ( x i ) ∑ j exp ⁡ ( x j ) ) text{log_softmax}(x_{i}) = logleft(frac{exp(x_i) }{ sum_j exp(x_j)} right)

log_softmax ( x i ​ ) = lo g ( ∑ j ​ exp ( x j ​ ) exp ( x i ​ ) ​ )

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – input
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – A dimension along which log_softmax will be computed.
* **dtype** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
If specified, the input tensor is cast to `dtype`  before the operation
is performed. This is useful for preventing data type overflows. Default: None.

Example: 

```
>>> t = torch.ones(2, 2)
>>> torch.special.log_softmax(t, 0)
tensor([[-0.6931, -0.6931],
        [-0.6931, -0.6931]])

```

torch.special. logit ( *input*  , *eps = None*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the logit of the elements of `input`  . `input`  is clamped to [eps, 1 - eps] when eps is not None.
When eps is None and `input`  < 0 or `input`  > 1, the function will yields NaN. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd class="mtr-glue">
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<msub>
<mi>
                 y
                </mi>
<mi>
                 i
                </mi>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                 =
                </mo>
<mi>
                 ln
                </mi>
<mo>
                 ⁡
                </mo>
<mo stretchy="false">
                 (
                </mo>
<mfrac>
<msub>
<mi>
                   z
                  </mi>
<mi>
                   i
                  </mi>
</msub>
<mrow>
<mn>
                   1
                  </mn>
<mo>
                   −
                  </mo>
<msub>
<mi>
                    z
                   </mi>
<mi>
                    i
                   </mi>
</msub>
</mrow>
</mfrac>
<mo stretchy="false">
                 )
                </mo>
</mrow>
</mstyle>
</mtd>
<mtd class="mtr-glue">
</mtd>
<mtd class="mml-eqn-num">
</mtd>
</mtr>
<mtr>
<mtd class="mtr-glue">
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<msub>
<mi>
                 z
                </mi>
<mi>
                 i
                </mi>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                 =
                </mo>
<mrow>
<mo fence="true">
                  {
                 </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msub>
<mi>
                       x
                      </mi>
<mi>
                       i
                      </mi>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                      if eps is None
                     </mtext>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                      eps
                     </mtext>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                       if
                      </mtext>
<msub>
<mi>
                        x
                       </mi>
<mi>
                        i
                       </mi>
</msub>
<mo>
                       &lt;
                      </mo>
<mtext>
                       eps
                      </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msub>
<mi>
                       x
                      </mi>
<mi>
                       i
                      </mi>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                       if eps
                      </mtext>
<mo>
                       ≤
                      </mo>
<msub>
<mi>
                        x
                       </mi>
<mi>
                        i
                       </mi>
</msub>
<mo>
                       ≤
                      </mo>
<mn>
                       1
                      </mn>
<mo>
                       −
                      </mo>
<mtext>
                       eps
                      </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mn>
                       1
                      </mn>
<mo>
                       −
                      </mo>
<mtext>
                       eps
                      </mtext>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                       if
                      </mtext>
<msub>
<mi>
                        x
                       </mi>
<mi>
                        i
                       </mi>
</msub>
<mo>
                       &gt;
                      </mo>
<mn>
                       1
                      </mn>
<mo>
                       −
                      </mo>
<mtext>
                       eps
                      </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
</mstyle>
</mtd>
<mtd class="mtr-glue">
</mtd>
<mtd class="mml-eqn-num">
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{align}
y_{i} &amp;= ln(frac{z_{i}}{1 - z_{i}}) 
z_{i} &amp;= begin{cases}
    x_{i} &amp; text{if eps is None} 
    text{eps} &amp; text{if } x_{i} &lt; text{eps} 
    x_{i} &amp; text{if } text{eps} leq x_{i} leq 1 - text{eps} 
    1 - text{eps} &amp; text{if } x_{i} &gt; 1 - text{eps}
end{cases}
end{align}
           </annotation>
</semantics>
</math> -->
y i = ln ⁡ ( z i 1 − z i ) z i = { x i if eps is None eps if x i < eps x i if eps ≤ x i ≤ 1 − eps 1 − eps if x i > 1 − eps begin{align}
y_{i} &= ln(frac{z_{i}}{1 - z_{i}}) 
z_{i} &= begin{cases}
 x_{i} & text{if eps is None} 
 text{eps} & text{if } x_{i} < text{eps} 
 x_{i} & text{if } text{eps} leq x_{i} leq 1 - text{eps} 
 1 - text{eps} & text{if } x_{i} > 1 - text{eps}
end{cases}
end{align}

y i ​ z i ​ ​ = ln ( 1 − z i ​ z i ​ ​ ) = ⎩ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuOTE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgOTE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFY5MTYgSDM4NHogTTM4NCAwIEg1MDQgVjkxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎨ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuOTE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgOTE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFY5MTYgSDM4NHogTTM4NCAwIEg1MDQgVjkxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎧ ​ x i ​ eps x i ​ 1 − eps ​ if eps is None if x i ​ < eps if eps ≤ x i ​ ≤ 1 − eps if x i ​ > 1 − eps ​ ​ ​

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – the epsilon for input clamp bound. Default: `None`

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.rand(5)
>>> a
tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
>>> torch.special.logit(a, eps=1e-6)
tensor([-0.9466,  2.6352,  0.6131, -1.7169,  0.6261])

```

torch.special. logsumexp ( *input*  , *dim*  , *keepdim = False*  , *** , *out = None* ) 
:   Alias for [`torch.logsumexp()`](generated/torch.logsumexp.html#torch.logsumexp "torch.logsumexp")  .

torch.special. modified_bessel_i0 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Modified Bessel function of the first kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            0
           </annotation>
</semantics>
</math> -->0 00  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. modified_bessel_i1 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Modified Bessel function of the first kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. modified_bessel_k0 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Modified Bessel function of the second kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            0
           </annotation>
</semantics>
</math> -->0 00  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. modified_bessel_k1 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Modified Bessel function of the second kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. multigammaln ( *input*  , *p*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the [multivariate log-gamma function](https://en.wikipedia.org/wiki/Multivariate_gamma_function)  with dimension <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             p
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            p
           </annotation>
</semantics>
</math> -->p pp  element-wise, given by 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             log
            </mi>
<mo>
             ⁡
            </mo>
<mo stretchy="false">
             (
            </mo>
<msub>
<mi mathvariant="normal">
              Γ
             </mi>
<mi>
              p
             </mi>
</msub>
<mo stretchy="false">
             (
            </mo>
<mi>
             a
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mi>
             C
            </mi>
<mo>
             +
            </mo>
<mstyle displaystyle="true" scriptlevel="0">
<munderover>
<mo>
               ∑
              </mo>
<mrow>
<mi>
                i
               </mi>
<mo>
                =
               </mo>
<mn>
                1
               </mn>
</mrow>
<mi>
               p
              </mi>
</munderover>
<mi>
              log
             </mi>
<mo>
              ⁡
             </mo>
<mrow>
<mo fence="true">
               (
              </mo>
<mi mathvariant="normal">
               Γ
              </mi>
<mrow>
<mo fence="true">
                (
               </mo>
<mi>
                a
               </mi>
<mo>
                −
               </mo>
<mfrac>
<mrow>
<mi>
                  i
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
<mn>
                 2
                </mn>
</mfrac>
<mo fence="true">
                )
               </mo>
</mrow>
<mo fence="true">
               )
              </mo>
</mrow>
</mstyle>
</mrow>
<annotation encoding="application/x-tex">
            log(Gamma_{p}(a)) = C + displaystyle sum_{i=1}^{p} logleft(Gammaleft(a - frac{i - 1}{2}right)right)
           </annotation>
</semantics>
</math> -->
log ⁡ ( Γ p ( a ) ) = C + ∑ i = 1 p log ⁡ ( Γ ( a − i − 1 2 ) ) log(Gamma_{p}(a)) = C + displaystyle sum_{i=1}^{p} logleft(Gammaleft(a - frac{i - 1}{2}right)right)

lo g ( Γ p ​ ( a )) = C + i = 1 ∑ p ​ lo g ( Γ ( a − 2 i − 1 ​ ) )

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             C
            </mi>
<mo>
             =
            </mo>
<mi>
             log
            </mi>
<mo>
             ⁡
            </mo>
<mo stretchy="false">
             (
            </mo>
<mi>
             π
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo>
             ⋅
            </mo>
<mfrac>
<mrow>
<mi>
               p
              </mi>
<mo stretchy="false">
               (
              </mo>
<mi>
               p
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
</mrow>
<mn>
              4
             </mn>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            C = log(pi) cdot frac{p (p - 1)}{4}
           </annotation>
</semantics>
</math> -->C = log ⁡ ( π ) ⋅ p ( p − 1 ) 4 C = log(pi) cdot frac{p (p - 1)}{4}C = lo g ( π ) ⋅ 4 p ( p − 1 ) ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             Γ
            </mi>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            Gamma(-)
           </annotation>
</semantics>
</math> -->Γ ( − ) Gamma(-)Γ ( − )  is the Gamma function. 

All elements must be greater than <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mrow>
<mi>
               p
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
<mn>
              2
             </mn>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            frac{p - 1}{2}
           </annotation>
</semantics>
</math> -->p − 1 2 frac{p - 1}{2}2 p − 1 ​  , otherwise the behavior is undefined. 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to compute the multivariate log-gamma function
* **p** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the number of dimensions

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.empty(2, 3).uniform_(1, 2)
>>> a
tensor([[1.6835, 1.8474, 1.1929],
        [1.0475, 1.7162, 1.4180]])
>>> torch.special.multigammaln(a, 2)
tensor([[0.3928, 0.4007, 0.7586],
        [1.0311, 0.3901, 0.5049]])

```

torch.special. ndtr ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the area under the standard Gaussian probability density function,
integrated from minus infinity to `input`  , elementwise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             ndtr
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
<mfrac>
<mn>
              1
             </mn>
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
</mfrac>
<msubsup>
<mo>
              ∫
             </mo>
<mrow>
<mo>
               −
              </mo>
<mi mathvariant="normal">
               ∞
              </mi>
</mrow>
<mi>
              x
             </mi>
</msubsup>
<msup>
<mi>
              e
             </mi>
<mrow>
<mo>
               −
              </mo>
<mfrac>
<mn>
                1
               </mn>
<mn>
                2
               </mn>
</mfrac>
<msup>
<mi>
                t
               </mi>
<mn>
                2
               </mn>
</msup>
</mrow>
</msup>
<mi>
             d
            </mi>
<mi>
             t
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            text{ndtr}(x) = frac{1}{sqrt{2 pi}}int_{-infty}^{x} e^{-frac{1}{2}t^2} dt
           </annotation>
</semantics>
</math> -->
ndtr ( x ) = 1 2 π ∫ − ∞ x e − 1 2 t 2 d t text{ndtr}(x) = frac{1}{sqrt{2 pi}}int_{-infty}^{x} e^{-frac{1}{2}t^2} dt

ndtr ( x ) = 2 π ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 1 ​ ∫ − ∞ x ​ e − 2 1 ​ t 2 d t

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.ndtr(torch.tensor([-3., -2, -1, 0, 1, 2, 3]))
tensor([0.0013, 0.0228, 0.1587, 0.5000, 0.8413, 0.9772, 0.9987])

```

torch.special. ndtri ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the argument, x, for which the area under the Gaussian probability density function
(integrated from minus infinity to x) is equal to `input`  , elementwise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             ndtri
            </mtext>
<mo stretchy="false">
             (
            </mo>
<mi>
             p
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<msqrt>
<mn>
              2
             </mn>
</msqrt>
<msup>
<mtext>
              erf
             </mtext>
<mrow>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</msup>
<mo stretchy="false">
             (
            </mo>
<mn>
             2
            </mn>
<mi>
             p
            </mi>
<mo>
             −
            </mo>
<mn>
             1
            </mn>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            text{ndtri}(p) = sqrt{2}text{erf}^{-1}(2p - 1)
           </annotation>
</semantics>
</math> -->
ndtri ( p ) = 2 erf − 1 ( 2 p − 1 ) text{ndtri}(p) = sqrt{2}text{erf}^{-1}(2p - 1)

ndtri ( p ) = 2 ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ erf − 1 ( 2 p − 1 )

Note 

Also known as quantile function for Normal Distribution.

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.special.ndtri(torch.tensor([0, 0.25, 0.5, 0.75, 1]))
tensor([   -inf, -0.6745,  0.0000,  0.6745,     inf])

```

torch.special. polygamma ( *n*  , *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
              n
             </mi>
<mrow>
<mi>
               t
              </mi>
<mi>
               h
              </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
            n^{th}
           </annotation>
</semantics>
</math> -->n t h n^{th}n t h  derivative of the digamma function on `input`  . <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             n
            </mi>
<mo>
             ≥
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            n geq 0
           </annotation>
</semantics>
</math> -->n ≥ 0 n geq 0n ≥ 0  is called the order of the polygamma function. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
              ψ
             </mi>
<mrow>
<mo stretchy="false">
               (
              </mo>
<mi>
               n
              </mi>
<mo stretchy="false">
               )
              </mo>
</mrow>
</msup>
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
<msup>
<mi>
               d
              </mi>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                n
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
</msup>
<mrow>
<mi>
               d
              </mi>
<msup>
<mi>
                x
               </mi>
<mrow>
<mo stretchy="false">
                 (
                </mo>
<mi>
                 n
                </mi>
<mo stretchy="false">
                 )
                </mo>
</mrow>
</msup>
</mrow>
</mfrac>
<mi>
             ψ
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
            psi^{(n)}(x) = frac{d^{(n)}}{dx^{(n)}} psi(x)
           </annotation>
</semantics>
</math> -->
ψ ( n ) ( x ) = d ( n ) d x ( n ) ψ ( x ) psi^{(n)}(x) = frac{d^{(n)}}{dx^{(n)}} psi(x)

ψ ( n ) ( x ) = d x ( n ) d ( n ) ​ ψ ( x )

Note 

This function is implemented only for nonnegative integers <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              n
             </mi>
<mo>
              ≥
             </mo>
<mn>
              0
             </mn>
</mrow>
<annotation encoding="application/x-tex">
             n geq 0
            </annotation>
</semantics>
</math> -->n ≥ 0 n geq 0n ≥ 0  .

Parameters
:   * **n** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the order of the polygamma function
* **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.tensor([1, 0.5])
>>> torch.special.polygamma(1, a)
tensor([1.64493, 4.9348])
>>> torch.special.polygamma(2, a)
tensor([ -2.4041, -16.8288])
>>> torch.special.polygamma(3, a)
tensor([ 6.4939, 97.4091])
>>> torch.special.polygamma(4, a)
tensor([ -24.8863, -771.4742])

```

torch.special. psi ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Alias for [`torch.special.digamma()`](#torch.special.digamma "torch.special.digamma")  .

torch.special. round ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Alias for [`torch.round()`](generated/torch.round.html#torch.round "torch.round")  .

torch.special. scaled_modified_bessel_k0 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Scaled modified Bessel function of the second kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            0
           </annotation>
</semantics>
</math> -->0 00  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. scaled_modified_bessel_k1 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Scaled modified Bessel function of the second kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            1
           </annotation>
</semantics>
</math> -->1 11  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. shifted_chebyshev_polynomial_t ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Chebyshev polynomial of the first kind <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
              T
             </mi>
<mi>
              n
             </mi>
<mo lspace="0em" rspace="0em">
              ∗
             </mo>
</msubsup>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            T_{n}^{ast}(text{input})
           </annotation>
</semantics>
</math> -->T n ∗ ( input ) T_{n}^{ast}(text{input})T n ∗ ​ ( input )  . 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. shifted_chebyshev_polynomial_u ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Chebyshev polynomial of the second kind <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
              U
             </mi>
<mi>
              n
             </mi>
<mo lspace="0em" rspace="0em">
              ∗
             </mo>
</msubsup>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            U_{n}^{ast}(text{input})
           </annotation>
</semantics>
</math> -->U n ∗ ( input ) U_{n}^{ast}(text{input})U n ∗ ​ ( input )  . 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. shifted_chebyshev_polynomial_v ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Chebyshev polynomial of the third kind <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
              V
             </mi>
<mi>
              n
             </mi>
<mo lspace="0em" rspace="0em">
              ∗
             </mo>
</msubsup>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            V_{n}^{ast}(text{input})
           </annotation>
</semantics>
</math> -->V n ∗ ( input ) V_{n}^{ast}(text{input})V n ∗ ​ ( input )  . 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. shifted_chebyshev_polynomial_w ( *input*  , *n*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Chebyshev polynomial of the fourth kind <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
              W
             </mi>
<mi>
              n
             </mi>
<mo lspace="0em" rspace="0em">
              ∗
             </mo>
</msubsup>
<mo stretchy="false">
             (
            </mo>
<mtext>
             input
            </mtext>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            W_{n}^{ast}(text{input})
           </annotation>
</semantics>
</math> -->W n ∗ ( input ) W_{n}^{ast}(text{input})W n ∗ ​ ( input )  . 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **n** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Degree of the polynomial.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. sinc ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the normalized sinc of `input.` 

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
<mrow>
<mo fence="true">
              {
             </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mn>
                   1
                  </mn>
<mo separator="true">
                   ,
                  </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                   if
                  </mtext>
<msub>
<mtext>
                    input
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo>
                   =
                  </mo>
<mn>
                   0
                  </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                   sin
                  </mi>
<mo>
                   ⁡
                  </mo>
<mo stretchy="false">
                   (
                  </mo>
<mi>
                   π
                  </mi>
<msub>
<mtext>
                    input
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo stretchy="false">
                   )
                  </mo>
<mi mathvariant="normal">
                   /
                  </mi>
<mo stretchy="false">
                   (
                  </mo>
<mi>
                   π
                  </mi>
<msub>
<mtext>
                    input
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo stretchy="false">
                   )
                  </mo>
<mo separator="true">
                   ,
                  </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                  otherwise
                 </mtext>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} =
begin{cases}
  1, &amp; text{if} text{input}_{i}=0 
  sin(pi text{input}_{i}) / (pi text{input}_{i}), &amp; text{otherwise}
end{cases}
           </annotation>
</semantics>
</math> -->
out i = { 1 , if input i = 0 sin ⁡ ( π input i ) / ( π input i ) , otherwise text{out}_{i} =
begin{cases}
 1, & text{if} text{input}_{i}=0 
 sin(pi text{input}_{i}) / (pi text{input}_{i}), & text{otherwise}
end{cases}

out i ​ = { 1 , sin ( π input i ​ ) / ( π input i ​ ) , ​ if input i ​ = 0 otherwise ​

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> t = torch.randn(4)
>>> t
tensor([ 0.2252, -0.2948,  1.0267, -1.1566])
>>> torch.special.sinc(t)
tensor([ 0.9186,  0.8631, -0.0259, -0.1300])

```

torch.special. softmax ( *input*  , *dim*  , *** , *dtype = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the softmax function. 

Softmax is defined as: 

<!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
             Softmax
            </mtext>
<mo stretchy="false">
             (
            </mo>
<msub>
<mi>
              x
             </mi>
<mi>
              i
             </mi>
</msub>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mfrac>
<mrow>
<mi>
               exp
              </mi>
<mo>
               ⁡
              </mo>
<mo stretchy="false">
               (
              </mo>
<msub>
<mi>
                x
               </mi>
<mi>
                i
               </mi>
</msub>
<mo stretchy="false">
               )
              </mo>
</mrow>
<mrow>
<msub>
<mo>
                ∑
               </mo>
<mi>
                j
               </mi>
</msub>
<mi>
               exp
              </mi>
<mo>
               ⁡
              </mo>
<mo stretchy="false">
               (
              </mo>
<msub>
<mi>
                x
               </mi>
<mi>
                j
               </mi>
</msub>
<mo stretchy="false">
               )
              </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            text{Softmax}(x_{i}) = frac{exp(x_i)}{sum_j exp(x_j)}
           </annotation>
</semantics>
</math> -->Softmax ( x i ) = exp ⁡ ( x i ) ∑ j exp ⁡ ( x j ) text{Softmax}(x_{i}) = frac{exp(x_i)}{sum_j exp(x_j)}Softmax ( x i ​ ) = ∑ j ​ e x p ( x j ​ ) e x p ( x i ​ ) ​ 

It is applied to all slices along dim, and will re-scale them so that the elements
lie in the range *[0, 1]* and sum to 1. 

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – input
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – A dimension along which softmax will be computed.
* **dtype** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
If specified, the input tensor is cast to `dtype`  before the operation
is performed. This is useful for preventing data type overflows. Default: None.

Examples::
:   ```
>>> t = torch.ones(2, 2)
>>> torch.special.softmax(t, 0)
tensor([[0.5000, 0.5000],
        [0.5000, 0.5000]])

```

torch.special. spherical_bessel_j0 ( *input*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Spherical Bessel function of the first kind of order <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            0
           </annotation>
</semantics>
</math> -->0 00  . 

Parameters
: **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

torch.special. xlog1py ( *input*  , *other*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes `input * log1p(other)`  with the following cases. 

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
<mrow>
<mo fence="true">
              {
             </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                  NaN
                 </mtext>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                   if
                  </mtext>
<msub>
<mtext>
                    other
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo>
                   =
                  </mo>
<mtext>
                   NaN
                  </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mn>
                  0
                 </mn>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                   if
                  </mtext>
<msub>
<mtext>
                    input
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo>
                   =
                  </mo>
<mn>
                   0.0
                  </mn>
<mtext>
                   and
                  </mtext>
<msub>
<mtext>
                    other
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo stretchy="false">
                   !
                  </mo>
<mo>
                   =
                  </mo>
<mtext>
                   NaN
                  </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<msub>
<mtext>
                    input
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo>
                   ∗
                  </mo>
<mtext>
                   log1p
                  </mtext>
<mo stretchy="false">
                   (
                  </mo>
<msub>
<mtext>
                    other
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo stretchy="false">
                   )
                  </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                  otherwise
                 </mtext>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = begin{cases}
    text{NaN} &amp; text{if } text{other}_{i} = text{NaN} 
    0 &amp; text{if } text{input}_{i} = 0.0 text{ and } text{other}_{i} != text{NaN} 
    text{input}_{i} * text{log1p}(text{other}_{i})&amp; text{otherwise}
end{cases}
           </annotation>
</semantics>
</math> -->
out i = { NaN if other i = NaN 0 if input i = 0.0 and other i ! = NaN input i ∗ log1p ( other i ) otherwise text{out}_{i} = begin{cases}
 text{NaN} & text{if } text{other}_{i} = text{NaN} 
 0 & text{if } text{input}_{i} = 0.0 text{ and } text{other}_{i} != text{NaN} 
 text{input}_{i} * text{log1p}(text{other}_{i})& text{otherwise}
end{cases}

out i ​ = ⎩ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎨ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎧ ​ NaN 0 input i ​ ∗ log1p ( other i ​ ) ​ if other i ​ = NaN if input i ​ = 0.0 and other i ​ ! = NaN otherwise ​

Similar to SciPy’s *scipy.special.xlog1py* . 

Parameters
:   * **input** ( *Number* *or* [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Multiplier
* **other** ( *Number* *or* [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Argument

Note 

At least one of `input`  or `other`  must be a tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> x = torch.zeros(5,)
>>> y = torch.tensor([-1, 0, 1, float('inf'), float('nan')])
>>> torch.special.xlog1py(x, y)
tensor([0., 0., 0., 0., nan])
>>> x = torch.tensor([1, 2, 3])
>>> y = torch.tensor([3, 2, 1])
>>> torch.special.xlog1py(x, y)
tensor([1.3863, 2.1972, 2.0794])
>>> torch.special.xlog1py(x, 4)
tensor([1.6094, 3.2189, 4.8283])
>>> torch.special.xlog1py(2, y)
tensor([2.7726, 2.1972, 1.3863])

```

torch.special. xlogy ( *input*  , *other*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes `input * log(other)`  with the following cases. 

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
<mrow>
<mo fence="true">
              {
             </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                  NaN
                 </mtext>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                   if
                  </mtext>
<msub>
<mtext>
                    other
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo>
                   =
                  </mo>
<mtext>
                   NaN
                  </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mn>
                  0
                 </mn>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                   if
                  </mtext>
<msub>
<mtext>
                    input
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo>
                   =
                  </mo>
<mn>
                   0.0
                  </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<msub>
<mtext>
                    input
                   </mtext>
<mi>
                    i
                   </mi>
</msub>
<mo>
                   ∗
                  </mo>
<mi>
                   log
                  </mi>
<mo>
                   ⁡
                  </mo>
<mrow>
<mo stretchy="false">
                    (
                   </mo>
<msub>
<mtext>
                     other
                    </mtext>
<mi>
                     i
                    </mi>
</msub>
<mo stretchy="false">
                    )
                   </mo>
</mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                  otherwise
                 </mtext>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
            text{out}_{i} = begin{cases}
    text{NaN} &amp; text{if } text{other}_{i} = text{NaN} 
    0 &amp; text{if } text{input}_{i} = 0.0 
    text{input}_{i} * log{(text{other}_{i})} &amp; text{otherwise}
end{cases}
           </annotation>
</semantics>
</math> -->
out i = { NaN if other i = NaN 0 if input i = 0.0 input i ∗ log ⁡ ( other i ) otherwise text{out}_{i} = begin{cases}
 text{NaN} & text{if } text{other}_{i} = text{NaN} 
 0 & text{if } text{input}_{i} = 0.0 
 text{input}_{i} * log{(text{other}_{i})} & text{otherwise}
end{cases}

out i ​ = ⎩ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎨ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎧ ​ NaN 0 input i ​ ∗ lo g ( other i ​ ) ​ if other i ​ = NaN if input i ​ = 0.0 otherwise ​

Similar to SciPy’s *scipy.special.xlogy* . 

Parameters
:   * **input** ( *Number* *or* [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Multiplier
* **other** ( *Number* *or* [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Argument

Note 

At least one of `input`  or `other`  must be a tensor.

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> x = torch.zeros(5,)
>>> y = torch.tensor([-1, 0, 1, float('inf'), float('nan')])
>>> torch.special.xlogy(x, y)
tensor([0., 0., 0., 0., nan])
>>> x = torch.tensor([1, 2, 3])
>>> y = torch.tensor([3, 2, 1])
>>> torch.special.xlogy(x, y)
tensor([1.0986, 1.3863, 0.0000])
>>> torch.special.xlogy(x, 4)
tensor([1.3863, 2.7726, 4.1589])
>>> torch.special.xlogy(2, y)
tensor([2.1972, 1.3863, 0.0000])

```

torch.special. zeta ( *input*  , *other*  , *** , *out = None* ) → [Tensor](tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the Hurwitz zeta function, elementwise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             ζ
            </mi>
<mo stretchy="false">
             (
            </mo>
<mi>
             x
            </mi>
<mo separator="true">
             ,
            </mo>
<mi>
             q
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<munderover>
<mo>
              ∑
             </mo>
<mrow>
<mi>
               k
              </mi>
<mo>
               =
              </mo>
<mn>
               0
              </mn>
</mrow>
<mi mathvariant="normal">
              ∞
             </mi>
</munderover>
<mfrac>
<mn>
              1
             </mn>
<mrow>
<mo stretchy="false">
               (
              </mo>
<mi>
               k
              </mi>
<mo>
               +
              </mo>
<mi>
               q
              </mi>
<msup>
<mo stretchy="false">
                )
               </mo>
<mi>
                x
               </mi>
</msup>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            zeta(x, q) = sum_{k=0}^{infty} frac{1}{(k + q)^x}
           </annotation>
</semantics>
</math> -->
ζ ( x , q ) = ∑ k = 0 ∞ 1 ( k + q ) x zeta(x, q) = sum_{k=0}^{infty} frac{1}{(k + q)^x}

ζ ( x , q ) = k = 0 ∑ ∞ ​ ( k + q ) x 1 ​

Parameters
:   * **input** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor corresponding to *x* .
* **other** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor corresponding to *q* .

Note 

The Riemann zeta function corresponds to the case when *q = 1*

Keyword Arguments
: **out** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> x = torch.tensor([2., 4.])
>>> torch.special.zeta(x, 1)
tensor([1.6449, 1.0823])
>>> torch.special.zeta(x, torch.tensor([1., 2.]))
tensor([1.6449, 0.0823])
>>> torch.special.zeta(2, torch.tensor([1., 2.]))
tensor([1.6449, 0.6449])

```

