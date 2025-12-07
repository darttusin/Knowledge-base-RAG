torch.lerp 
========================================================

torch. lerp ( *input*  , *end*  , *weight*  , *** , *out = None* ) 
:   Does a linear interpolation of two tensors `start`  (given by `input`  ) and `end`  based
on a scalar or tensor `weight`  and returns the resulting `out`  tensor. 

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
<mtext>
             start
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            +
           </mo>
<msub>
<mtext>
             weight
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            ×
           </mo>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             end
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            −
           </mo>
<msub>
<mtext>
             start
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i = text{start}_i + text{weight}_i times (text{end}_i - text{start}_i)
          </annotation>
</semantics>
</math> -->
out i = start i + weight i × ( end i − start i ) text{out}_i = text{start}_i + text{weight}_i times (text{end}_i - text{start}_i)

out i ​ = start i ​ + weight i ​ × ( end i ​ − start i ​ )

The shapes of `start`  and `end`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  . If `weight`  is a tensor, then
the shapes of `weight`  , `start`  , and `end`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor with the starting points
* **end** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor with the ending points
* **weight** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* *tensor*  ) – the weight for the interpolation formula

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> start = torch.arange(1., 5.)
>>> end = torch.empty(4).fill_(10)
>>> start
tensor([ 1.,  2.,  3.,  4.])
>>> end
tensor([ 10.,  10.,  10.,  10.])
>>> torch.lerp(start, end, 0.5)
tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
>>> torch.lerp(start, end, torch.full_like(start, 0.5))
tensor([ 5.5000,  6.0000,  6.5000,  7.0000])

```

