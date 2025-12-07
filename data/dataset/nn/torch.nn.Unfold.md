Unfold 
================================================

*class* torch.nn. Unfold ( *kernel_size*  , *dilation = 1*  , *padding = 0*  , *stride = 1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/fold.py#L168) 
:   Extracts sliding local blocks from a batched input tensor. 

Consider a batched `input`  tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo>
            ∗
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (N, C, *)
          </annotation>
</semantics>
</math> -->( N , C , ∗ ) (N, C, *)( N , C , ∗ )  ,
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
</math> -->N NN  is the batch dimension, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->C CC  is the channel dimension,
and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ∗
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           *
          </annotation>
</semantics>
</math> -->∗ *∗  represent arbitrary spatial dimensions. This operation flattens
each sliding `kernel_size`  -sized block within the spatial dimensions
of `input`  into a column (i.e., last dimension) of a 3-D `output`  tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C × ∏ ( kernel_size ) , L ) (N, C times prod(text{kernel_size}), L)( N , C × ∏ ( kernel_size ) , L )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->C × ∏ ( kernel_size ) C times prod(text{kernel_size})C × ∏ ( kernel_size )  is the total number of values
within each block (a block has <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∏ ( kernel_size ) prod(text{kernel_size})∏ ( kernel_size )  spatial
locations each containing a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->L LL  is
the total number of such blocks: 

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
               spatial_size
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
           L = prod_d leftlfloorfrac{text{spatial_size}[d] + 2 times text{padding}[d] %
    - text{dilation}[d] times (text{kernel_size}[d] - 1) - 1}{text{stride}[d]} + 1rightrfloor,
          </annotation>
</semantics>
</math> -->
L = ∏ d ⌊ spatial_size [ d ] + 2 × padding [ d ] − dilation [ d ] × ( kernel_size [ d ] − 1 ) − 1 stride [ d ] + 1 ⌋ , L = prod_d leftlfloorfrac{text{spatial_size}[d] + 2 times text{padding}[d] %
 - text{dilation}[d] times (text{kernel_size}[d] - 1) - 1}{text{stride}[d]} + 1rightrfloor,

L = d ∏ ​ ⌊ stride [ d ] spatial_size [ d ] + 2 × padding [ d ] − dilation [ d ] × ( kernel_size [ d ] − 1 ) − 1 ​ + 1 ⌋ ,

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            spatial_size
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{spatial_size}
          </annotation>
</semantics>
</math> -->spatial_size text{spatial_size}spatial_size  is formed by the spatial dimensions
of `input`  ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ∗
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           *
          </annotation>
</semantics>
</math> -->∗ *∗  above), and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->d dd  is over all spatial
dimensions. 

Therefore, indexing `output`  at the last dimension (column dimension)
gives all values within a certain block. 

The `padding`  , `stride`  and `dilation`  arguments specify
how the sliding blocks are retrieved. 

* `stride`  controls the stride for the sliding blocks.
* `padding`  controls the amount of implicit zero-paddings on both
sides for `padding`  number of points for each dimension before
reshaping.
* `dilation`  controls the spacing between the kernel points; also known as the à trous algorithm.
It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  has a nice visualization of what `dilation`  does.

Parameters
:   * **kernel_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – the size of the sliding blocks
* **dilation** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – a parameter that controls the
stride of elements within the
neighborhood. Default: 1
* **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – implicit zero padding to be added on
both sides of input. Default: 0
* **stride** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – the stride of the sliding blocks in the input
spatial dimensions. Default: 1

* If `kernel_size`  , `dilation`  , `padding`  or `stride`  is an int or a tuple of length 1, their values will be
replicated across all spatial dimensions.
* For the case of two input spatial dimensions this operation is sometimes
called `im2col`  .

Note 

[`Fold`](torch.nn.Fold.html#torch.nn.Fold "torch.nn.Fold")  calculates each combined value in the resulting
large tensor by summing all values from all containing blocks. [`Unfold`](#torch.nn.Unfold "torch.nn.Unfold")  extracts the values in the local blocks by
copying from the large tensor. So, if the blocks overlap, they are not
inverses of each other. 

In general, folding and unfolding operations are related as
follows. Consider [`Fold`](torch.nn.Fold.html#torch.nn.Fold "torch.nn.Fold")  and [`Unfold`](#torch.nn.Unfold "torch.nn.Unfold")  instances created with the same
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

Currently, only 4-D input tensors (batched image-like tensors) are
supported.

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
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, *)
              </annotation>
</semantics>
</math> -->( N , C , ∗ ) (N, C, *)( N , C , ∗ )

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
</math> -->( N , C × ∏ ( kernel_size ) , L ) (N, C times prod(text{kernel_size}), L)( N , C × ∏ ( kernel_size ) , L )  as described above

Examples: 

```
>>> unfold = nn.Unfold(kernel_size=(2, 3))
>>> input = torch.randn(2, 5, 3, 4)
>>> output = unfold(input)
>>> # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
>>> # 4 blocks (2x3 kernels) in total in the 3x4 input
>>> output.size()
torch.Size([2, 30, 4])

>>> # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
>>> inp = torch.randn(1, 3, 10, 12)
>>> w = torch.randn(2, 3, 4, 5)
>>> inp_unf = torch.nn.functional.unfold(inp, (4, 5))
>>> out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
>>> out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
>>> # or equivalently (and avoiding a copy),
>>> # out = out_unf.view(1, 2, 7, 8)
>>> (torch.nn.functional.conv2d(inp, w) - out).abs().max()
tensor(1.9073e-06)

```

