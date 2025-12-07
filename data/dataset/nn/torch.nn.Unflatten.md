Unflatten 
======================================================

*class* torch.nn. Unflatten ( *dim*  , *unflattened_size* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/flatten.py#L59) 
:   Unflattens a tensor dim expanding it to a desired shape. For use with `Sequential`  . 

* `dim`  specifies the dimension of the input tensor to be unflattened, and it can
be either *int* or *str* when *Tensor* or *NamedTensor* is used, respectively.
* `unflattened_size`  is the new shape of the unflattened dimension of the tensor and it can be
a *tuple* of ints or a *list* of ints or *torch.Size* for *Tensor* input; a *NamedShape* (tuple of *(name, size)* tuples) for *NamedTensor* input.

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 S
                </mi>
<mtext>
                 dim
                </mtext>
</msub>
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
               (*, S_{text{dim}}, *)
              </annotation>
</semantics>
</math> -->( ∗ , S dim , ∗ ) (*, S_{text{dim}}, *)( ∗ , S dim ​ , ∗ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 S
                </mi>
<mtext>
                 dim
                </mtext>
</msub>
</mrow>
<annotation encoding="application/x-tex">
               S_{text{dim}}
              </annotation>
</semantics>
</math> -->S dim S_{text{dim}}S dim ​  is the size at
dimension `dim`  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  means any number of dimensions including none.

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 U
                </mi>
<mn>
                 1
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
                 U
                </mi>
<mi>
                 n
                </mi>
</msub>
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
               (*, U_1, ..., U_n, *)
              </annotation>
</semantics>
</math> -->( ∗ , U 1 , . . . , U n , ∗ ) (*, U_1, ..., U_n, *)( ∗ , U 1 ​ , ... , U n ​ , ∗ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                U
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               U
              </annotation>
</semantics>
</math> -->U UU  = `unflattened_size`  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mo>
                 ∏
                </mo>
<mrow>
<mi>
                  i
                 </mi>
<mo>
                  =
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
<mi>
                 n
                </mi>
</msubsup>
<msub>
<mi>
                 U
                </mi>
<mi>
                 i
                </mi>
</msub>
<mo>
                =
               </mo>
<msub>
<mi>
                 S
                </mi>
<mtext>
                 dim
                </mtext>
</msub>
</mrow>
<annotation encoding="application/x-tex">
               prod_{i=1}^n U_i = S_{text{dim}}
              </annotation>
</semantics>
</math> -->∏ i = 1 n U i = S dim prod_{i=1}^n U_i = S_{text{dim}}∏ i = 1 n ​ U i ​ = S dim ​  .

Parameters
:   * **dim** ( *Union* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]*  ) – Dimension to be unflattened
* **unflattened_size** ( *Union* *[* [*torch.Size*](../size.html#torch.Size "torch.Size") *,* *Tuple* *,* *List* *,* *NamedShape* *]*  ) – New shape of the unflattened dimension

Examples 

```
>>> input = torch.randn(2, 50)
>>> # With tuple of ints
>>> m = nn.Sequential(
>>>     nn.Linear(50, 50),
>>>     nn.Unflatten(1, (2, 5, 5))
>>> )
>>> output = m(input)
>>> output.size()
torch.Size([2, 2, 5, 5])
>>> # With torch.Size
>>> m = nn.Sequential(
>>>     nn.Linear(50, 50),
>>>     nn.Unflatten(1, torch.Size([2, 5, 5]))
>>> )
>>> output = m(input)
>>> output.size()
torch.Size([2, 2, 5, 5])
>>> # With namedshape (tuple of tuples)
>>> input = torch.randn(2, 50, names=("N", "features"))
>>> unflatten = nn.Unflatten("features", (("C", 2), ("H", 5), ("W", 5)))
>>> output = unflatten(input)
>>> output.size()
torch.Size([2, 2, 5, 5])

```

NamedShape 
:   alias of [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ]]

