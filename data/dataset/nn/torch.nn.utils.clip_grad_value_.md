torch.nn.utils.clip_grad_value_ 
====================================================================================================

torch.nn.utils. clip_grad_value_ ( *parameters*  , *clip_value*  , *foreach = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/clip_grad.py#L247) 
:   Clip the gradients of an iterable of parameters at specified value. 

Gradients are modified in-place. 

Parameters
:   * **parameters** ( *Iterable* *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *] or* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – an iterable of Tensors or a
single Tensor that will have gradients normalized
* **clip_value** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – maximum allowed value of the gradients.
The gradients are clipped in the range <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo fence="true">
                [
               </mo>
<mtext>
                -clip_value
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
                clip_value
               </mtext>
<mo fence="true">
                ]
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               left[text{-clip_value}, text{clip_value}right]
              </annotation>
</semantics>
</math> -->[ -clip_value , clip_value ] left[text{-clip_value}, text{clip_value}right][ -clip_value , clip_value ]

* **foreach** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – use the faster foreach-based implementation
If `None`  , use the foreach implementation for CUDA and CPU native tensors and
silently fall back to the slow implementation for other device types.
Default: `None`

