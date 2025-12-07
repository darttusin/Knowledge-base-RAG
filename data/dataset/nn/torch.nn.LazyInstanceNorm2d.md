LazyInstanceNorm2d 
========================================================================

*class* torch.nn. LazyInstanceNorm2d ( *eps = 1e-05*  , *momentum = 0.1*  , *affine = True*  , *track_running_stats = True*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/instancenorm.py#L320) 
:   A [`torch.nn.InstanceNorm2d`](torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d "torch.nn.InstanceNorm2d")  module with lazy initialization of the `num_features`  argument. 

The `num_features`  argument of the [`InstanceNorm2d`](torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d "torch.nn.InstanceNorm2d")  is inferred from the `input.size(1)`  .
The attributes that will be lazily initialized are *weight* , *bias* , *running_mean* and *running_var* . 

Check the [`torch.nn.modules.lazy.LazyModuleMixin`](torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin "torch.nn.modules.lazy.LazyModuleMixin")  for further documentation
on lazy modules and their limitations. 

Parameters
:   * **num_features** – <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                C
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               C
              </annotation>
</semantics>
</math> -->C CC  from an expected input of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
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
               (N, C, H, W)
              </annotation>
</semantics>
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
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
               (C, H, W)
              </annotation>
</semantics>
</math> -->( C , H , W ) (C, H, W)( C , H , W )

* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – a value added to the denominator for numerical stability. Default: 1e-5
* **momentum** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]*  ) – the value used for the running_mean and running_var computation. Default: 0.1
* **affine** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this module has
learnable affine parameters, initialized the same way as done for batch normalization.
Default: `False`  .
* **track_running_stats** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this
module tracks the running mean and variance, and when set to `False`  ,
this module does not track such statistics and always uses batch
statistics in both training and eval modes. Default: `False`

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
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
               (N, C, H, W)
              </annotation>
</semantics>
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
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
               (C, H, W)
              </annotation>
</semantics>
</math> -->( C , H , W ) (C, H, W)( C , H , W )

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
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
               (N, C, H, W)
              </annotation>
</semantics>
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
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
               (C, H, W)
              </annotation>
</semantics>
</math> -->( C , H , W ) (C, H, W)( C , H , W )  (same shape as input)

cls_to_become [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/instancenorm.py#L241) 
:   alias of [`InstanceNorm2d`](torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d "torch.nn.modules.instancenorm.InstanceNorm2d")

