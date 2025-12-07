torch.nn.utils.clip_grads_with_norm_ 
===============================================================================================================

torch.nn.utils. clip_grads_with_norm_ ( *parameters*  , *max_norm*  , *total_norm*  , *foreach = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/clip_grad.py#L113) 
:   Scale the gradients of an iterable of parameters given a pre-calculated total norm and desired max norm. 

The gradients will be scaled by the following calculation 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            g
           </mi>
<mi>
            r
           </mi>
<mi>
            a
           </mi>
<mi>
            d
           </mi>
<mo>
            =
           </mo>
<mi>
            g
           </mi>
<mi>
            r
           </mi>
<mi>
            a
           </mi>
<mi>
            d
           </mi>
<mo>
            ∗
           </mo>
<mfrac>
<mrow>
<mi>
              m
             </mi>
<mi>
              a
             </mi>
<mi>
              x
             </mi>
<mi mathvariant="normal">
              _
             </mi>
<mi>
              n
             </mi>
<mi>
              o
             </mi>
<mi>
              r
             </mi>
<mi>
              m
             </mi>
</mrow>
<mrow>
<mi>
              t
             </mi>
<mi>
              o
             </mi>
<mi>
              t
             </mi>
<mi>
              a
             </mi>
<mi>
              l
             </mi>
<mi mathvariant="normal">
              _
             </mi>
<mi>
              n
             </mi>
<mi>
              o
             </mi>
<mi>
              r
             </mi>
<mi>
              m
             </mi>
<mo>
              +
             </mo>
<mn>
              1
             </mn>
<mi>
              e
             </mi>
<mo>
              −
             </mo>
<mn>
              6
             </mn>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           grad = grad * frac{max_norm}{total_norm + 1e-6}
          </annotation>
</semantics>
</math> -->
g r a d = g r a d ∗ m a x _ n o r m t o t a l _ n o r m + 1 e − 6 grad = grad * frac{max_norm}{total_norm + 1e-6}

g r a d = g r a d ∗ t o t a l _ n or m + 1 e − 6 ma x _ n or m ​

Gradients are modified in-place. 

This function is equivalent to [`torch.nn.utils.clip_grad_norm_()`](torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_ "torch.nn.utils.clip_grad_norm_")  with a pre-calculated
total norm. 

Parameters
:   * **parameters** ( *Iterable* *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *] or* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – an iterable of Tensors or a
single Tensor that will have gradients normalized
* **max_norm** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – max norm of the gradients
* **total_norm** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – total norm of the gradients to use for clipping
* **foreach** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – use the faster foreach-based implementation.
If `None`  , use the foreach implementation for CUDA and CPU native tensors and silently
fall back to the slow implementation for other device types.
Default: `None`

Returns
:   None

Return type
:   None

