torch.addr 
========================================================

torch. addr ( *input*  , *vec1*  , *vec2*  , *** , *beta = 1*  , *alpha = 1*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Performs the outer-product of vectors `vec1`  and `vec2`  and adds it to the matrix `input`  . 

Optional values `beta`  and `alpha`  are scaling factors on the
outer product between `vec1`  and `vec2`  and the added matrix `input`  respectively. 

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
            vec1
           </mtext>
<mo>
            ⊗
           </mo>
<mtext>
            vec2
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out} = beta text{input} + alpha (text{vec1} otimes text{vec2})
          </annotation>
</semantics>
</math> -->
out = β input + α ( vec1 ⊗ vec2 ) text{out} = beta text{input} + alpha (text{vec1} otimes text{vec2})

out = β input + α ( vec1 ⊗ vec2 )

If `beta`  is 0, then the content of `input`  will be ignored, and *nan* and *inf* in
it will not be propagated. 

If `vec1`  is a vector of size *n* and `vec2`  is a vector
of size *m* , then `input`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  with a matrix of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( n × m ) (n times m)( n × m )  and `out`  will be a matrix of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( n × m ) (n times m)( n × m )  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – matrix to be added
* **vec1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the first vector of the outer product
* **vec2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second vector of the outer product

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
<mtext>
                vec1
               </mtext>
<mo>
                ⊗
               </mo>
<mtext>
                vec2
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               text{vec1} otimes text{vec2}
              </annotation>
</semantics>
</math> -->vec1 ⊗ vec2 text{vec1} otimes text{vec2}vec1 ⊗ vec2  ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
>>> vec1 = torch.arange(1., 4.)
>>> vec2 = torch.arange(1., 3.)
>>> M = torch.zeros(3, 2)
>>> torch.addr(M, vec1, vec2)
tensor([[ 1.,  2.],
        [ 2.,  4.],
        [ 3.,  6.]])

```

