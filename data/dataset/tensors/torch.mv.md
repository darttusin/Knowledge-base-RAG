torch.mv 
====================================================

torch. mv ( *input*  , *vec*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Performs a matrix-vector product of the matrix `input`  and the vector `vec`  . 

If `input`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            m
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           m
          </annotation>
</semantics>
</math> -->m mm  , `out`  will be 1-D of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           n
          </annotation>
</semantics>
</math> -->n nn  . 

Note 

This function does not [broadcast](../notes/broadcasting.html#broadcasting-semantics)  .

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – matrix to be multiplied
* **vec** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – vector to be multiplied

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> mat = torch.randn(2, 3)
>>> vec = torch.randn(3)
>>> torch.mv(mat, vec)
tensor([ 1.0404, -0.6361])

```

