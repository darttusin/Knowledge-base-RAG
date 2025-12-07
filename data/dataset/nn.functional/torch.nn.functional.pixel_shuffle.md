torch.nn.functional.pixel_shuffle 
=======================================================================================================

torch.nn.functional. pixel_shuffle ( *input*  , *upscale_factor* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Rearranges elements in a tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mo>
            ∗
           </mo>
<mo separator="true">
            ,
           </mo>
<mi>
            C
           </mi>
<mo>
            ×
           </mo>
<msup>
<mi>
             r
            </mi>
<mn>
             2
            </mn>
</msup>
<mo separator="true">
            ,
           </mo>
<mi>
            H
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            W
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (*, C times r^2, H, W)
          </annotation>
</semantics>
</math> -->( ∗ , C × r 2 , H , W ) (*, C times r^2, H, W)( ∗ , C × r 2 , H , W )  to a
tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mo>
            ∗
           </mo>
<mo separator="true">
            ,
           </mo>
<mi>
            C
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            H
           </mi>
<mo>
            ×
           </mo>
<mi>
            r
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            W
           </mi>
<mo>
            ×
           </mo>
<mi>
            r
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (*, C, H times r, W times r)
          </annotation>
</semantics>
</math> -->( ∗ , C , H × r , W × r ) (*, C, H times r, W times r)( ∗ , C , H × r , W × r )  , where r is the `upscale_factor`  . 

See [`PixelShuffle`](torch.nn.PixelShuffle.html#torch.nn.PixelShuffle "torch.nn.PixelShuffle")  for details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor
* **upscale_factor** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – factor to increase spatial resolution by

Examples: 

```
>>> input = torch.randn(1, 9, 4, 4)
>>> output = torch.nn.functional.pixel_shuffle(input, 3)
>>> print(output.size())
torch.Size([1, 1, 12, 12])

```

