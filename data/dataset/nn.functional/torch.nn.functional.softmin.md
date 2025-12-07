torch.nn.functional.softmin 
==========================================================================================

torch.nn.functional. softmin ( *input*  , *dim = None*  , *_stacklevel = 3*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L2067) 
:   Apply a softmin function. 

Note that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Softmin
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mtext>
            Softmax
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mo>
            −
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{Softmin}(x) = text{Softmax}(-x)
          </annotation>
</semantics>
</math> -->Softmin ( x ) = Softmax ( − x ) text{Softmin}(x) = text{Softmax}(-x)Softmin ( x ) = Softmax ( − x )  . See softmax definition for mathematical formula. 

See [`Softmin`](torch.nn.Softmin.html#torch.nn.Softmin "torch.nn.Softmin")  for more details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – input
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – A dimension along which softmin will be computed (so every slice
along dim will sum to 1).
* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
If specified, the input tensor is casted to `dtype`  before the operation
is performed. This is useful for preventing data type overflows. Default: None.

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

