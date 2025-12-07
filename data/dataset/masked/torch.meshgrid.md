torch.meshgrid 
================================================================

torch. meshgrid ( ** tensors*  , *indexing = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/functional.py#L445) 
:   Creates grids of coordinates specified by the 1D inputs in *attr* :tensors. 

This is helpful when you want to visualize data over some
range of inputs. See below for a plotting example. 

Given <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  1D tensors <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             T
            </mi>
<mn>
             0
            </mn>
</msub>
<mo>
            …
           </mo>
<msub>
<mi>
             T
            </mi>
<mrow>
<mi>
              N
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           T_0 ldots T_{N-1}
          </annotation>
</semantics>
</math> -->T 0 … T N − 1 T_0 ldots T_{N-1}T 0 ​ … T N − 1 ​  as
inputs with corresponding sizes <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             S
            </mi>
<mn>
             0
            </mn>
</msub>
<mo>
            …
           </mo>
<msub>
<mi>
             S
            </mi>
<mrow>
<mi>
              N
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           S_0 ldots S_{N-1}
          </annotation>
</semantics>
</math> -->S 0 … S N − 1 S_0 ldots S_{N-1}S 0 ​ … S N − 1 ​  ,
this creates <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  N-dimensional tensors <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             G
            </mi>
<mn>
             0
            </mn>
</msub>
<mo>
            …
           </mo>
<msub>
<mi>
             G
            </mi>
<mrow>
<mi>
              N
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           G_0 ldots
G_{N-1}
          </annotation>
</semantics>
</math> -->G 0 … G N − 1 G_0 ldots
G_{N-1}G 0 ​ … G N − 1 ​  , each with shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             S
            </mi>
<mn>
             0
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             S
            </mi>
<mrow>
<mi>
              N
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (S_0, ..., S_{N-1})
          </annotation>
</semantics>
</math> -->( S 0 , . . . , S N − 1 ) (S_0, ..., S_{N-1})( S 0 ​ , ... , S N − 1 ​ )  where
the output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             G
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           G_i
          </annotation>
</semantics>
</math> -->G i G_iG i ​  is constructed by expanding <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             T
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           T_i
          </annotation>
</semantics>
</math> -->T i T_iT i ​  to the result shape. 

Note 

0D inputs are treated equivalently to 1D inputs of a
single element.

Warning 

*torch.meshgrid(*tensors)* currently has the same behavior
as calling *numpy.meshgrid(*arrays, indexing=’ij’)* . 

In the future *torch.meshgrid* will transition to *indexing=’xy’* as the default. 

[pytorch/pytorch#50276](https://github.com/pytorch/pytorch/issues/50276)  tracks
this issue with the goal of migrating to NumPy’s behavior.

See also 

[`torch.cartesian_prod()`](torch.cartesian_prod.html#torch.cartesian_prod "torch.cartesian_prod")  has the same effect but it
collects the data in a tensor of vectors.

Parameters
:   * **tensors** ( [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)") *of* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – list of scalars or 1 dimensional tensors. Scalars will be
treated as tensors of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mn>
                1
               </mn>
<mo separator="true">
                ,
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (1,)
              </annotation>
</semantics>
</math> -->( 1 , ) (1,)( 1 , )  automatically

* **indexing** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]*  ) –

    (str, optional): the indexing mode, either “xy”
        or “ij”, defaults to “ij”. See warning for future changes.

    If “xy” is selected, the first dimension corresponds
        to the cardinality of the second input and the second
        dimension corresponds to the cardinality of the first
        input.

    If “ij” is selected, the dimensions are in the same
        order as the cardinality of the inputs.

Returns
:   If the input has <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  tensors of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
               S
              </mi>
<mn>
               0
              </mn>
</msub>
<mo>
              …
             </mo>
<msub>
<mi>
               S
              </mi>
<mrow>
<mi>
                N
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
             S_0 ldots S_{N-1}
            </annotation>
</semantics>
</math> -->S 0 … S N − 1 S_0 ldots S_{N-1}S 0 ​ … S N − 1 ​  , then the
output will also have <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  tensors, where each tensor
is of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               S
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               S
              </mi>
<mrow>
<mi>
                N
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
</msub>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (S_0, ..., S_{N-1})
            </annotation>
</semantics>
</math> -->( S 0 , . . . , S N − 1 ) (S_0, ..., S_{N-1})( S 0 ​ , ... , S N − 1 ​ )  .

Return type
:   seq (sequence of Tensors)

Example: 

```
>>> x = torch.tensor([1, 2, 3])
>>> y = torch.tensor([4, 5, 6])

Observe the element-wise pairings across the grid, (1, 4),
(1, 5), ..., (3, 6). This is the same thing as the
cartesian product.
>>> grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
>>> grid_x
tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])
>>> grid_y
tensor([[4, 5, 6],
        [4, 5, 6],
        [4, 5, 6]])

This correspondence can be seen when these grids are
stacked properly.
>>> torch.equal(torch.cat(tuple(torch.dstack([grid_x, grid_y]))),
...             torch.cartesian_prod(x, y))
True

`torch.meshgrid` is commonly used to produce a grid for
plotting.
>>> import matplotlib.pyplot as plt
>>> xs = torch.linspace(-5, 5, steps=100)
>>> ys = torch.linspace(-5, 5, steps=100)
>>> x, y = torch.meshgrid(xs, ys, indexing='xy')
>>> z = torch.sin(torch.sqrt(x * x + y * y))
>>> ax = plt.axes(projection='3d')
>>> ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
>>> plt.show()

```

[![../_images/meshgrid.png](../_images/meshgrid.png)](../_images/meshgrid.png)

