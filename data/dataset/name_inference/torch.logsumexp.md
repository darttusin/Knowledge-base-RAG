torch.logsumexp 
==================================================================

torch. logsumexp ( *input*  , *dim*  , *keepdim = False*  , *** , *out = None* ) 
:   Returns the log of summed exponentials of each row of the `input`  tensor in the given dimension `dim`  . The computation is numerically
stabilized. 

For summation index <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            j
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           j
          </annotation>
</semantics>
</math> -->j jj  given by *dim* and other indices <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           i
          </annotation>
</semantics>
</math> -->i ii  , the result is 

> <!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mtext>
> logsumexp
> </mtext>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> x
> </mi>
> <msub>
> <mo stretchy="false">
> )
> </mo>
> <mi>
> i
> </mi>
> </msub>
> <mo>
> =
> </mo>
> <mi>
> log
> </mi>
> <mo>
> ⁡
> </mo>
> <munder>
> <mo>
> ∑
> </mo>
> <mi>
> j
> </mi>
> </munder>
> <mi>
> exp
> </mi>
> <mo>
> ⁡
> </mo>
> <mo stretchy="false">
> (
> </mo>
> <msub>
> <mi>
> x
> </mi>
> <mrow>
> <mi>
> i
> </mi>
> <mi>
> j
> </mi>
> </mrow>
> </msub>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> text{logsumexp}(x)_{i} = log sum_j exp(x_{ij})
> </annotation>
> </semantics>
> </math> -->
> logsumexp ( x ) i = log ⁡ ∑ j exp ⁡ ( x i j ) text{logsumexp}(x)_{i} = log sum_j exp(x_{ij})
> 
> logsumexp ( x ) i ​ = lo g j ∑ ​ exp ( x ij ​ )

If `keepdim`  is `True`  , the output tensor is of the same size
as `input`  except in the dimension(s) `dim`  where it is of size 1.
Otherwise, `dim`  is squeezed (see [`torch.squeeze()`](torch.squeeze.html#torch.squeeze "torch.squeeze")  ), resulting in the
output tensor having 1 (or `len(dim)`  ) fewer dimension(s). 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *of* *ints*  ) – the dimension or dimensions to reduce.
* **keepdim** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether the output tensor has `dim`  retained or not. Default: `False`  .

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(3, 3)
>>> torch.logsumexp(a, 1)
tensor([1.4907, 1.0593, 1.5696])
>>> torch.dist(torch.logsumexp(a, 1), torch.log(torch.sum(torch.exp(a), 1)))
tensor(1.6859e-07)

```

