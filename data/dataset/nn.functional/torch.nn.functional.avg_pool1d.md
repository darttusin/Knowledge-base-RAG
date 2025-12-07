torch.nn.functional.avg_pool1d 
=================================================================================================

torch.nn.functional. avg_pool1d ( *input*  , *kernel_size*  , *stride = None*  , *padding = 0*  , *ceil_mode = False*  , *count_include_pad = True* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies a 1D average pooling over an input signal composed of several
input planes. 

Note 

pad should be at most half of effective kernel size.

See [`AvgPool1d`](torch.nn.AvgPool1d.html#torch.nn.AvgPool1d "torch.nn.AvgPool1d")  for details and output shape. 

Parameters
:   * **input** – input tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mtext>
                minibatch
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
                in_channels
               </mtext>
<mo separator="true">
                ,
               </mo>
<mi>
                i
               </mi>
<mi>
                W
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (text{minibatch} , text{in_channels} , iW)
              </annotation>
</semantics>
</math> -->( minibatch , in_channels , i W ) (text{minibatch} , text{in_channels} , iW)( minibatch , in_channels , iW )

* **kernel_size** – the size of the window. Can be a single number or a
tuple *(kW,)*
* **stride** – the stride of the window. Can be a single number or a tuple *(sW,)* . Default: `kernel_size`
* **padding** – implicit zero paddings on both sides of the input. Can be a
single number or a tuple *(padW,)* . Default: 0
* **ceil_mode** – when True, will use *ceil* instead of *floor* to compute the
output shape. Default: `False`
* **count_include_pad** – when True, will include the zero-padding in the
averaging calculation. Default: `True`

Examples: 

```
>>> # pool of square window of size=3, stride=2
>>> input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
>>> F.avg_pool1d(input, kernel_size=3, stride=2)
tensor([[[ 2.,  4.,  6.]]])

```

