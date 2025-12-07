torch.sparse.softmax 
============================================================================

torch.sparse. softmax ( *input*  , *dim*  , *** , *dtype = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies a softmax function. 

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
              e
             </mi>
<mi>
              x
             </mi>
<mi>
              p
             </mi>
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
              e
             </mi>
<mi>
              x
             </mi>
<mi>
              p
             </mi>
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
</math> -->Softmax ( x i ) = e x p ( x i ) ∑ j e x p ( x j ) text{Softmax}(x_{i}) = frac{exp(x_i)}{sum_j exp(x_j)}Softmax ( x i ​ ) = ∑ j ​ e x p ( x j ​ ) e x p ( x i ​ ) ​ 

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            j
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           i, j
          </annotation>
</semantics>
</math> -->i , j i, ji , j  run over sparse tensor indices and unspecified
entries are ignores. This is equivalent to defining unspecified
entries as negative infinity so that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            e
           </mi>
<mi>
            x
           </mi>
<mi>
            p
           </mi>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             x
            </mi>
<mi>
             k
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mn>
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           exp(x_k) = 0
          </annotation>
</semantics>
</math> -->e x p ( x k ) = 0 exp(x_k) = 0e x p ( x k ​ ) = 0  when the
entry with index <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            k
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           k
          </annotation>
</semantics>
</math> -->k kk  has not specified. 

It is applied to all slices along *dim* , and will re-scale them so
that the elements lie in the range *[0, 1]* and sum to 1. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – input
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – A dimension along which softmax will be computed.
* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type
of returned tensor. If specified, the input tensor is
casted to `dtype`  before the operation is
performed. This is useful for preventing data type
overflows. Default: None

