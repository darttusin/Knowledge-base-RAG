torch.atan2 
==========================================================

torch. atan2 ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *other : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Element-wise arctangent of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi mathvariant="normal">
            /
           </mi>
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
           text{input}_{i} / text{other}_{i}
          </annotation>
</semantics>
</math> -->input i / other i text{input}_{i} / text{other}_{i}input i ​ / other i ​  with consideration of the quadrant. Returns a new tensor with the signed angles
in radians between vector <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<mo separator="true">
            ,
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
<annotation encoding="application/x-tex">
           (text{other}_{i}, text{input}_{i})
          </annotation>
</semantics>
</math> -->( other i , input i ) (text{other}_{i}, text{input}_{i})( other i ​ , input i ​ )  and vector <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mn>
            1
           </mn>
<mo separator="true">
            ,
           </mo>
<mn>
            0
           </mn>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (1, 0)
          </annotation>
</semantics>
</math> -->( 1 , 0 ) (1, 0)( 1 , 0 )  . (Note that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
           text{other}_{i}
          </annotation>
</semantics>
</math> -->other i text{other}_{i}other i ​  , the second
parameter, is the x-coordinate, while <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
           text{input}_{i}
          </annotation>
</semantics>
</math> -->input i text{input}_{i}input i ​  , the first
parameter, is the y-coordinate.) 

The shapes of `input`  and `other`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the first input tensor
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second input tensor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4)
>>> a
tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
>>> torch.atan2(a, torch.randn(4))
tensor([ 0.9833,  0.0811, -1.9743, -1.4151])

```

