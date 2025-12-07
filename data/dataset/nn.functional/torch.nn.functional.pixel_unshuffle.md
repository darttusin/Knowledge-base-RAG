torch.nn.functional.pixel_unshuffle 
===========================================================================================================

torch.nn.functional. pixel_unshuffle ( *input*  , *downscale_factor* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Reverses the [`PixelShuffle`](torch.nn.PixelShuffle.html#torch.nn.PixelShuffle "torch.nn.PixelShuffle")  operation by rearranging elements in a
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
</math> -->( ∗ , C , H × r , W × r ) (*, C, H times r, W times r)( ∗ , C , H × r , W × r )  to a tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ , C × r 2 , H , W ) (*, C times r^2, H, W)( ∗ , C × r 2 , H , W )  , where r is the `downscale_factor`  . 

See [`PixelUnshuffle`](torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle "torch.nn.PixelUnshuffle")  for details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor
* **downscale_factor** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – factor to increase spatial resolution by

Examples: 

```
>>> input = torch.randn(1, 1, 12, 12)
>>> output = torch.nn.functional.pixel_unshuffle(input, 3)
>>> print(output.size())
torch.Size([1, 9, 4, 4])

```

