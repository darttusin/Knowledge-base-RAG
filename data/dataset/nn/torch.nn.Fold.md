Fold 
============================================

*class* torch.nn. Fold ( *output_size*  , *kernel_size*  , *dilation = 1*  , *padding = 0*  , *stride = 1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/fold.py#L11) 
:   Combines an array of sliding local blocks into a large containing tensor. 

Consider a batched `input`  tensor containing sliding local blocks,
e.g., patches of images, of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo>
            ×
           </mo>
<mo>
            ∏
           </mo>
<mo stretchy="false">
            (
           </mo>
<mtext>
            kernel_size
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mo separator="true">
            ,
           </mo>
<mi>
            L
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (N, C times  prod(text{kernel_size}), L)
          </annotation>
</semantics>
</math> -->( N , C × ∏ ( kernel_size ) , L ) (N, C times prod(text{kernel_size}), L)( N , C × ∏ ( kernel_size ) , L )  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            N
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           N
          </annotation>
</semantics>
</math> -->N NN  is batch dimension, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            C
           </mi>
<mo>
            ×
           </mo>
<mo>
            ∏
           </mo>
<mo stretchy="false">
            (
           </mo>
<mtext>
            kernel_size
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           C times prod(text{kernel_size})
          </annotation>
</semantics>
</math> -->C × ∏ ( kernel_size ) C times prod(text{kernel_size})C × ∏ ( kernel_size )  is the number of values within a block (a block has <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ∏
           </mo>
<mo stretchy="false">
            (
           </mo>
<mtext>
            kernel_size
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           prod(text{kernel_size})
          </annotation>
</semantics>
</math> -->∏ ( kernel_size ) prod(text{kernel_size})∏ ( kernel_size )  spatial locations each containing a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->C CC  -channeled vector), and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            L
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           L
          </annotation>
</semantics>
</math> -->L LL  is the total number of blocks. (This is exactly the
same specification as the output shape of [`Unfold`](torch.nn.Unfold.html#torch.nn.Unfold "torch.nn.Unfold")  .) This
operation combines these local blocks into the large `output`  tensor
of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mtext>
            output_size
           </mtext>
<mo stretchy="false">
            [
           </mo>
<mn>
            0
           </mn>
<mo stretchy="false">
            ]
           </mo>
<mo separator="true">
            ,
           </mo>
<mtext>
            output_size
           </mtext>
<mo stretchy="false">
            [
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            ]
           </mo>
<mo separator="true">
            ,
           </mo>
<mo>
            …
           </mo>
<mtext>
</mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (N, C, text{output_size}[0], text{output_size}[1], dots)
          </annotation>
</semantics>
</math> -->( N , C , output_size [ 0 ] , output_size [ 1 ] , … ) (N, C, text{output_size}[0], text{output_size}[1], dots)( N , C , output_size [ 0 ] , output_size [ 1 ] , … )  by summing the overlapping values. Similar to [`Unfold`](torch.nn.Unfold.html#torch.nn.Unfold "torch.nn.Unfold")  , the
arguments must satisfy 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            L
           </mi>
<mo>
            =
           </mo>
<munder>
<mo>
             ∏
            </mo>
<mi>
             d
            </mi>
</munder>
<mrow>
<mo fence="true">
             ⌊
            </mo>
<mfrac>
<mrow>
<mtext>
               output_size
              </mtext>
<mo stretchy="false">
               [
              </mo>
<mi>
               d
              </mi>
<mo stretchy="false">
               ]
              </mo>
<mo>
               +
              </mo>
<mn>
               2
              </mn>
<mo>
               ×
              </mo>
<mtext>
               padding
              </mtext>
<mo stretchy="false">
               [
              </mo>
<mi>
               d
              </mi>
<mo stretchy="false">
               ]
              </mo>
<mo>
               −
              </mo>
<mtext>
               dilation
              </mtext>
<mo stretchy="false">
               [
              </mo>
<mi>
               d
              </mi>
<mo stretchy="false">
               ]
              </mo>
<mo>
               ×
              </mo>
<mo stretchy="false">
               (
              </mo>
<mtext>
               kernel_size
              </mtext>
<mo stretchy="false">
               [
              </mo>
<mi>
               d
              </mi>
<mo stretchy="false">
               ]
              </mo>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
<mrow>
<mtext>
               stride
              </mtext>
<mo stretchy="false">
               [
              </mo>
<mi>
               d
              </mi>
<mo stretchy="false">
               ]
              </mo>
</mrow>
</mfrac>
<mo>
             +
            </mo>
<mn>
             1
            </mn>
<mo fence="true">
             ⌋
            </mo>
</mrow>
<mo separator="true">
            ,
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           L = prod_d leftlfloorfrac{text{output_size}[d] + 2 times text{padding}[d] %
    - text{dilation}[d] times (text{kernel_size}[d] - 1) - 1}{text{stride}[d]} + 1rightrfloor,
          </annotation>
</semantics>
</math> -->
L = ∏ d ⌊ output_size [ d ] + 2 × padding [ d ] − dilation [ d ] × ( kernel_size [ d ] − 1 ) − 1 stride [ d ] + 1 ⌋ , L = prod_d leftlfloorfrac{text{output_size}[d] + 2 times text{padding}[d] %
 - text{dilation}[d] times (text{kernel_size}[d] - 1) - 1}{text{stride}[d]} + 1rightrfloor,

L = d ∏ ​ ⌊ stride [ d ] output_size [ d ] + 2 × padding [ d ] − dilation [ d ] × ( kernel_size [ d ] − 1 ) − 1 ​ + 1 ⌋ ,

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            d
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           d
          </annotation>
</semantics>
</math> -->d dd  is over all spatial dimensions. 

* `output_size`  describes the spatial shape of the large containing
tensor of the sliding local blocks. It is useful to resolve the ambiguity
when multiple input shapes map to same number of sliding blocks, e.g.,
with `stride > 0`  .

The `padding`  , `stride`  and `dilation`  arguments specify
how the sliding blocks are retrieved. 

* `stride`  controls the stride for the sliding blocks.
* `padding`  controls the amount of implicit zero-paddings on both
sides for `padding`  number of points for each dimension before
reshaping.
* `dilation`  controls the spacing between the kernel points; also known as the à trous algorithm.
It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  has a nice visualization of what `dilation`  does.

Parameters
:   * **output_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – the shape of the spatial dimensions of the
output (i.e., `output.sizes()[2:]`  )
* **kernel_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – the size of the sliding blocks
* **dilation** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – a parameter that controls the
stride of elements within the
neighborhood. Default: 1
* **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – implicit zero padding to be added on
both sides of input. Default: 0
* **stride** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – the stride of the sliding blocks in the input
spatial dimensions. Default: 1

* If `output_size`  , `kernel_size`  , `dilation`  , `padding`  or `stride`  is an int or a tuple of length 1 then
their values will be replicated across all spatial dimensions.
* For the case of two output spatial dimensions this operation is sometimes
called `col2im`  .

Note 

[`Fold`](#torch.nn.Fold "torch.nn.Fold")  calculates each combined value in the resulting
large tensor by summing all values from all containing blocks. [`Unfold`](torch.nn.Unfold.html#torch.nn.Unfold "torch.nn.Unfold")  extracts the values in the local blocks by
copying from the large tensor. So, if the blocks overlap, they are not
inverses of each other. 

In general, folding and unfolding operations are related as
follows. Consider [`Fold`](#torch.nn.Fold "torch.nn.Fold")  and [`Unfold`](torch.nn.Unfold.html#torch.nn.Unfold "torch.nn.Unfold")  instances created with the same
parameters: 

```
>>> fold_params = dict(kernel_size=..., dilation=..., padding=..., stride=...)
>>> fold = nn.Fold(output_size=..., **fold_params)
>>> unfold = nn.Unfold(**fold_params)

```

Then for any (supported) `input`  tensor the following
equality holds: 

```
fold(unfold(input)) == divisor * input

```

where `divisor`  is a tensor that depends only on the shape
and dtype of the `input`  : 

```
>>> input_ones = torch.ones(input.shape, dtype=input.dtype)
>>> divisor = fold(unfold(input_ones))

```

When the `divisor`  tensor contains no zero elements, then `fold`  and `unfold`  operations are inverses of each
other (up to constant divisor).

Warning 

Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.

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
<mo>
                ×
               </mo>
<mo>
                ∏
               </mo>
<mo stretchy="false">
                (
               </mo>
<mtext>
                kernel_size
               </mtext>
<mo stretchy="false">
                )
               </mo>
<mo separator="true">
                ,
               </mo>
<mi>
                L
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C times prod(text{kernel_size}), L)
              </annotation>
</semantics>
</math> -->( N , C × ∏ ( kernel_size ) , L ) (N, C times prod(text{kernel_size}), L)( N , C × ∏ ( kernel_size ) , L )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                C
               </mi>
<mo>
                ×
               </mo>
<mo>
                ∏
               </mo>
<mo stretchy="false">
                (
               </mo>
<mtext>
                kernel_size
               </mtext>
<mo stretchy="false">
                )
               </mo>
<mo separator="true">
                ,
               </mo>
<mi>
                L
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C times prod(text{kernel_size}), L)
              </annotation>
</semantics>
</math> -->( C × ∏ ( kernel_size ) , L ) (C times prod(text{kernel_size}), L)( C × ∏ ( kernel_size ) , L )

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
<mtext>
                output_size
               </mtext>
<mo stretchy="false">
                [
               </mo>
<mn>
                0
               </mn>
<mo stretchy="false">
                ]
               </mo>
<mo separator="true">
                ,
               </mo>
<mtext>
                output_size
               </mtext>
<mo stretchy="false">
                [
               </mo>
<mn>
                1
               </mn>
<mo stretchy="false">
                ]
               </mo>
<mo separator="true">
                ,
               </mo>
<mo>
                …
               </mo>
<mtext>
</mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, text{output_size}[0], text{output_size}[1], dots)
              </annotation>
</semantics>
</math> -->( N , C , output_size [ 0 ] , output_size [ 1 ] , … ) (N, C, text{output_size}[0], text{output_size}[1], dots)( N , C , output_size [ 0 ] , output_size [ 1 ] , … )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mtext>
                output_size
               </mtext>
<mo stretchy="false">
                [
               </mo>
<mn>
                0
               </mn>
<mo stretchy="false">
                ]
               </mo>
<mo separator="true">
                ,
               </mo>
<mtext>
                output_size
               </mtext>
<mo stretchy="false">
                [
               </mo>
<mn>
                1
               </mn>
<mo stretchy="false">
                ]
               </mo>
<mo separator="true">
                ,
               </mo>
<mo>
                …
               </mo>
<mtext>
</mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C, text{output_size}[0], text{output_size}[1], dots)
              </annotation>
</semantics>
</math> -->( C , output_size [ 0 ] , output_size [ 1 ] , … ) (C, text{output_size}[0], text{output_size}[1], dots)( C , output_size [ 0 ] , output_size [ 1 ] , … )  as described above

Examples: 

```
>>> fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
>>> input = torch.randn(1, 3 * 2 * 2, 12)
>>> output = fold(input)
>>> output.size()
torch.Size([1, 3, 4, 5])

```

