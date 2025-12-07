torch.triu_indices 
=========================================================================

torch. triu_indices ( *row*  , *col*  , *offset = 0*  , *** , *dtype = torch.long*  , *device = 'cpu'*  , *layout = torch.strided* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns the indices of the upper triangular part of a `row`  by `col`  matrix in a 2-by-N Tensor, where the first row contains row
coordinates of all indices and the second row contains column coordinates.
Indices are ordered based on rows and then columns. 

The upper triangular part of the matrix is defined as the elements on and
above the diagonal. 

The argument `offset`  controls which diagonal to consider. If `offset`  = 0, all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the main
diagonal, and similarly a negative value includes just as many diagonals below
the main diagonal. The main diagonal are the set of indices <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            {
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            i
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            i
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            }
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           lbrace (i, i) rbrace
          </annotation>
</semantics>
</math> -->{ ( i , i ) } lbrace (i, i) rbrace{( i , i )}  for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
<mo>
            ∈
           </mo>
<mo stretchy="false">
            [
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            min
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            {
           </mo>
<msub>
<mi>
             d
            </mi>
<mn>
             1
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             d
            </mi>
<mn>
             2
            </mn>
</msub>
<mo stretchy="false">
            }
           </mo>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            ]
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           i in [0, min{d_{1}, d_{2}} - 1]
          </annotation>
</semantics>
</math> -->i ∈ [ 0 , min ⁡ { d 1 , d 2 } − 1 ] i in [0, min{d_{1}, d_{2}} - 1]i ∈ [ 0 , min { d 1 ​ , d 2 ​ } − 1 ]  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             d
            </mi>
<mn>
             1
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             d
            </mi>
<mn>
             2
            </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           d_{1}, d_{2}
          </annotation>
</semantics>
</math> -->d 1 , d 2 d_{1}, d_{2}d 1 ​ , d 2 ​  are the dimensions of the matrix. 

Note 

When running on CUDA, `row * col`  must be less than <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mn>
              2
             </mn>
<mn>
              59
             </mn>
</msup>
</mrow>
<annotation encoding="application/x-tex">
            2^{59}
           </annotation>
</semantics>
</math> -->2 59 2^{59}2 59  to
prevent overflow during calculation.

Parameters
:   * **row** ( `int`  ) – number of rows in the 2-D matrix.
* **col** ( `int`  ) – number of columns in the 2-D matrix.
* **offset** ( `int`  ) – diagonal offset from the main diagonal.
Default: if not provided, 0.

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor,
only support `torch.int`  , `torch.long`  . Default: if `None`  , `torch.long`  .
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , uses the current device for the default tensor type
(see [`torch.set_default_device()`](torch.set_default_device.html#torch.set_default_device "torch.set_default_device")  ). [`device`](../tensor_attributes.html#torch.device "torch.device")  will be the CPU
for CPU tensor types and the current CUDA device for CUDA tensor types.
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – currently only support `torch.strided`  .

Example: 

```
>>> a = torch.triu_indices(3, 3)
>>> a
tensor([[0, 0, 0, 1, 1, 2],
        [0, 1, 2, 1, 2, 2]])

>>> a = torch.triu_indices(4, 3, -1)
>>> a
tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3],
        [0, 1, 2, 0, 1, 2, 1, 2, 2]])

>>> a = torch.triu_indices(4, 3, 1)
>>> a
tensor([[0, 0, 1],
        [1, 2, 2]])

```

