torch.addmv 
==========================================================

torch. addmv ( *input*  , *mat*  , *vec*  , *** , *beta = 1*  , *alpha = 1*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Performs a matrix-vector product of the matrix `mat`  and
the vector `vec`  .
The vector `input`  is added to the final result. 

If `mat`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            n
           </mi>
<mo>
            ×
           </mo>
<mi>
            m
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (n times m)
          </annotation>
</semantics>
</math> -->( n × m ) (n times m)( n × m )  tensor, `vec`  is a 1-D tensor of
size *m* , then `input`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  with a 1-D tensor of size *n* and `out`  will be 1-D tensor of size *n* . 

`alpha`  and `beta`  are scaling factors on matrix-vector product between `mat`  and `vec`  and the added tensor `input`  respectively. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            out
           </mtext>
<mo>
            =
           </mo>
<mi>
            β
           </mi>
<mtext>
            input
           </mtext>
<mo>
            +
           </mo>
<mi>
            α
           </mi>
<mtext>
</mtext>
<mo stretchy="false">
            (
           </mo>
<mtext>
            mat
           </mtext>
<mo lspace="0.22em" mathvariant="normal" rspace="0.22em">
            @
           </mo>
<mtext>
            vec
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out} = beta text{input} + alpha (text{mat} mathbin{@} text{vec})
          </annotation>
</semantics>
</math> -->
out = β input + α ( mat @ vec ) text{out} = beta text{input} + alpha (text{mat} mathbin{@} text{vec})

out = β input + α ( mat @ vec )

If `beta`  is 0, then the content of `input`  will be ignored, and *nan* and *inf* in
it will not be propagated. 

For inputs of type *FloatTensor* or *DoubleTensor* , arguments `beta`  and `alpha`  must be real numbers, otherwise they should be integers. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – vector to be added
* **mat** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – matrix to be matrix multiplied
* **vec** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – vector to be matrix multiplied

Keyword Arguments
:   * **beta** ( *Number* *,* *optional*  ) – multiplier for `input`  ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                β
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               beta
              </annotation>
</semantics>
</math> -->β betaβ  )

* **alpha** ( *Number* *,* *optional*  ) – multiplier for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                m
               </mi>
<mi>
                a
               </mi>
<mi>
                t
               </mi>
<mi mathvariant="normal">
                @
               </mi>
<mi>
                v
               </mi>
<mi>
                e
               </mi>
<mi>
                c
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               mat @ vec
              </annotation>
</semantics>
</math> -->m a t @ v e c mat @ vecma t @ v ec  ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                α
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               alpha
              </annotation>
</semantics>
</math> -->α alphaα  )

* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> M = torch.randn(2)
>>> mat = torch.randn(2, 3)
>>> vec = torch.randn(3)
>>> torch.addmv(M, mat, vec)
tensor([-0.3768, -5.5565])

```

