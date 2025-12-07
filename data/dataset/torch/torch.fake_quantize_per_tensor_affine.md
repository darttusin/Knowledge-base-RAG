torch.fake_quantize_per_tensor_affine 
==================================================================================================================

torch. fake_quantize_per_tensor_affine ( *input*  , *scale*  , *zero_point*  , *quant_min*  , *quant_max* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the data in `input`  fake quantized using `scale`  , `zero_point`  , `quant_min`  and `quant_max`  . 

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
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input value(s), `torch.float32`  tensor
* **scale** (double scalar or `float32`  Tensor) – quantization scale
* **zero_point** (int64 scalar or `int32`  Tensor) – quantization zero_point
* **quant_min** ( *int64*  ) – lower bound of the quantized domain
* **quant_max** ( *int64*  ) – upper bound of the quantized domain

Returns
:   A newly fake_quantized `torch.float32`  tensor

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

Example: 

```
>>> x = torch.randn(4)
>>> x
tensor([ 0.0552,  0.9730,  0.3973, -1.0780])
>>> torch.fake_quantize_per_tensor_affine(x, 0.1, 0, 0, 255)
tensor([0.1000, 1.0000, 0.4000, 0.0000])
>>> torch.fake_quantize_per_tensor_affine(x, torch.tensor(0.1), torch.tensor(0), 0, 255)
tensor([0.1000, 1.0000, 0.4000, 0.0000])

```

