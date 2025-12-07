torch.fake_quantize_per_channel_affine 
====================================================================================================================

torch. fake_quantize_per_channel_affine ( *input*  , *scale*  , *zero_point*  , *axis*  , *quant_min*  , *quant_max* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the data in `input`  fake quantized per channel using `scale`  , `zero_point`  , `quant_min`  and `quant_max`  , across the channel specified by `axis`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            output
           </mtext>
<mo>
            =
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            m
           </mi>
<mi>
            i
           </mi>
<mi>
            n
           </mi>
<mo stretchy="false">
            (
           </mo>
<mtext>
            quant_max
           </mtext>
<mo separator="true">
            ,
           </mo>
<mi>
            m
           </mi>
<mi>
            a
           </mi>
<mi>
            x
           </mi>
<mo stretchy="false">
            (
           </mo>
<mtext>
            quant_min
           </mtext>
<mo separator="true">
            ,
           </mo>
<mtext>
            std::nearby_int
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext>
            input
           </mtext>
<mi mathvariant="normal">
            /
           </mi>
<mtext>
            scale
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
           </mo>
<mtext>
            zero_point
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
<mo>
            −
           </mo>
<mtext>
            zero_point
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mo>
            ×
           </mo>
<mtext>
            scale
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{output} = (
    min(
        text{quant_max},
        max(
            text{quant_min},
            text{std::nearby_int}(text{input} / text{scale}) + text{zero_point}
        )
    ) - text{zero_point}
) times text{scale}
          </annotation>
</semantics>
</math> -->
output = ( m i n ( quant_max , m a x ( quant_min , std::nearby_int ( input / scale ) + zero_point ) ) − zero_point ) × scale text{output} = (
 min(
 text{quant_max},
 max(
 text{quant_min},
 text{std::nearby_int}(text{input} / text{scale}) + text{zero_point}
 )
 ) - text{zero_point}
) times text{scale}

output = ( min ( quant_max , ma x ( quant_min , std::nearby_int ( input / scale ) + zero_point )) − zero_point ) × scale

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input value(s), in `torch.float32`
* **scale** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – quantization scale, per channel in `torch.float32`
* **zero_point** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – quantization zero_point, per channel in `torch.int32`  or `torch.half`  or `torch.float32`
* **axis** ( *int32*  ) – channel axis
* **quant_min** ( *int64*  ) – lower bound of the quantized domain
* **quant_max** ( *int64*  ) – upper bound of the quantized domain

Returns
:   A newly fake_quantized per channel `torch.float32`  tensor

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

Example: 

```
>>> x = torch.randn(2, 2, 2)
>>> x
tensor([[[-0.2525, -0.0466],
         [ 0.3491, -0.2168]],

        [[-0.5906,  1.6258],
         [ 0.6444, -0.0542]]])
>>> scales = (torch.randn(2) + 1) * 0.05
>>> scales
tensor([0.0475, 0.0486])
>>> zero_points = torch.zeros(2).to(torch.int32)
>>> zero_points
tensor([0, 0])
>>> torch.fake_quantize_per_channel_affine(x, scales, zero_points, 1, 0, 255)
tensor([[[0.0000, 0.0000],
         [0.3405, 0.0000]],

        [[0.0000, 1.6134],
        [0.6323, 0.0000]]])

```

