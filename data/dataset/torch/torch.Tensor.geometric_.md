torch.Tensor.geometric_ 
==================================================================================

Tensor. geometric_ ( *p*  , *** , *generator = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Fills `self`  tensor with elements drawn from the geometric distribution: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            P
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            X
           </mi>
<mo>
            =
           </mo>
<mi>
            k
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            1
           </mn>
<mo>
            −
           </mo>
<mi>
            p
           </mi>
<msup>
<mo stretchy="false">
             )
            </mo>
<mrow>
<mi>
              k
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msup>
<mi>
            p
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            k
           </mi>
<mo>
            =
           </mo>
<mn>
            1
           </mn>
<mo separator="true">
            ,
           </mo>
<mn>
            2
           </mn>
<mo separator="true">
            ,
           </mo>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           P(X=k) = (1 - p)^{k - 1} p, k = 1, 2, ...
          </annotation>
</semantics>
</math> -->
P ( X = k ) = ( 1 − p ) k − 1 p , k = 1 , 2 , . . . P(X=k) = (1 - p)^{k - 1} p, k = 1, 2, ...

P ( X = k ) = ( 1 − p ) k − 1 p , k = 1 , 2 , ...

Note 

[`torch.Tensor.geometric_()`](#torch.Tensor.geometric_ "torch.Tensor.geometric_") *k* -th trial is the first success hence draws samples in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             {
            </mo>
<mn>
             1
            </mn>
<mo separator="true">
             ,
            </mo>
<mn>
             2
            </mn>
<mo separator="true">
             ,
            </mo>
<mo>
             …
            </mo>
<mo stretchy="false">
             }
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            {1, 2, ldots}
           </annotation>
</semantics>
</math> -->{ 1 , 2 , … } {1, 2, ldots}{ 1 , 2 , … }  , whereas [`torch.distributions.geometric.Geometric()`](../distributions.html#torch.distributions.geometric.Geometric "torch.distributions.geometric.Geometric") <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<mn>
             1
            </mn>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            (k+1)
           </annotation>
</semantics>
</math> -->( k + 1 ) (k+1)( k + 1 )  -th trial is the first success
hence draws samples in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             {
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
<mo separator="true">
             ,
            </mo>
<mo>
             …
            </mo>
<mo stretchy="false">
             }
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            {0, 1, ldots}
           </annotation>
</semantics>
</math> -->{ 0 , 1 , … } {0, 1, ldots}{ 0 , 1 , … }  .

