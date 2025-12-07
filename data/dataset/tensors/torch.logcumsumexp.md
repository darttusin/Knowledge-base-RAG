torch.logcumsumexp 
========================================================================

torch. logcumsumexp ( *input*  , *dim*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns the logarithm of the cumulative summation of the exponentiation of
elements of `input`  in the dimension `dim`  . 

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
> logcumsumexp
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
> <mrow>
> <mi>
> i
> </mi>
> <mi>
> j
> </mi>
> </mrow>
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
> <munderover>
> <mo>
> ∑
> </mo>
> <mrow>
> <mi>
> k
> </mi>
> <mo>
> =
> </mo>
> <mn>
> 0
> </mn>
> </mrow>
> <mi>
> j
> </mi>
> </munderover>
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
> k
> </mi>
> </mrow>
> </msub>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> text{logcumsumexp}(x)_{ij} = log sumlimits_{k=0}^{j} exp(x_{ik})
> </annotation>
> </semantics>
> </math> -->
> logcumsumexp ( x ) i j = log ⁡ ∑ k = 0 j exp ⁡ ( x i k ) text{logcumsumexp}(x)_{ij} = log sumlimits_{k=0}^{j} exp(x_{ik})
> 
> logcumsumexp ( x ) ij ​ = lo g k = 0 ∑ j ​ exp ( x ik ​ )

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the dimension to do the operation over

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(10)
>>> torch.logcumsumexp(a, dim=0)
tensor([-0.42296738, -0.04462666,  0.86278635,  0.94622083,  1.05277811,
         1.39202815,  1.83525007,  1.84492621,  2.06084887,  2.06844475]))

```

