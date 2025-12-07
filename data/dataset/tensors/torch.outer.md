torch.outer 
==========================================================

torch. outer ( *input*  , *vec2*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Outer product of `input`  and `vec2`  .
If `input`  is a vector of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->n nn  and `vec2`  is a vector of
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
</math> -->m mm  , then `out`  must be a matrix of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

Note 

This function does not [broadcast](../notes/broadcasting.html#broadcasting-semantics)  .

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – 1-D input vector
* **vec2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – 1-D input vector

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – optional output matrix

Example: 

```
>>> v1 = torch.arange(1., 5.)
>>> v2 = torch.arange(1., 4.)
>>> torch.outer(v1, v2)
tensor([[  1.,   2.,   3.],
        [  2.,   4.,   6.],
        [  3.,   6.,   9.],
        [  4.,   8.,  12.]])

```

