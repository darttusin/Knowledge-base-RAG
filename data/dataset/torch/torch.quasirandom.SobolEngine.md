SobolEngine 
==========================================================

*class* torch.quasirandom. SobolEngine ( *dimension*  , *scramble = False*  , *seed = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/quasirandom.py#L7) 
:   The [`torch.quasirandom.SobolEngine`](#torch.quasirandom.SobolEngine "torch.quasirandom.SobolEngine")  is an engine for generating
(scrambled) Sobol sequences. Sobol sequences are an example of low
discrepancy quasi-random sequences. 

This implementation of an engine for Sobol sequences is capable of
sampling sequences up to a maximum dimension of 21201. It uses direction
numbers from [https://web.maths.unsw.edu.au/~fkuo/sobol/](https://web.maths.unsw.edu.au/~fkuo/sobol/)  obtained using the
search criterion D(6) up to the dimension 21201. This is the recommended
choice by the authors. 

References 

* Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
Journal of Complexity, 14(4):466-489, December 1998.
* I. M. Sobol. The distribution of points in a cube and the accurate
evaluation of integrals.
Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

Parameters
:   * **dimension** ( *Int*  ) – The dimensionality of the sequence to be drawn
* **scramble** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Setting this to `True`  will produce
scrambled Sobol sequences. Scrambling is
capable of producing better Sobol
sequences. Default: `False`  .
* **seed** ( *Int* *,* *optional*  ) – This is the seed for the scrambling. The seed
of the random number generator is set to this,
if specified. Otherwise, it uses a random seed.
Default: `None`

Examples: 

```
>>> soboleng = torch.quasirandom.SobolEngine(dimension=5)
>>> soboleng.draw(3)
tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
        [0.7500, 0.2500, 0.2500, 0.2500, 0.7500]])

```

draw ( *n = 1*  , *out = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/quasirandom.py#L78) 
:   Function to draw a sequence of `n`  points from a Sobol sequence.
Note that the samples are dependent on the previous samples. The size
of the result is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              n
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              d
             </mi>
<mi>
              i
             </mi>
<mi>
              m
             </mi>
<mi>
              e
             </mi>
<mi>
              n
             </mi>
<mi>
              s
             </mi>
<mi>
              i
             </mi>
<mi>
              o
             </mi>
<mi>
              n
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (n, dimension)
            </annotation>
</semantics>
</math> -->( n , d i m e n s i o n ) (n, dimension)( n , d im e n s i o n )  . 

Parameters
:   * **n** ( *Int* *,* *optional*  ) – The length of sequence of points to draw.
Default: 1
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – The output tensor
* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of the
returned tensor.
Default: `None`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

draw_base2 ( *m*  , *out = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/quasirandom.py#L131) 
:   Function to draw a sequence of `2**m`  points from a Sobol sequence.
Note that the samples are dependent on the previous samples. The size
of the result is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mn>
              2
             </mn>
<mo>
              ∗
             </mo>
<mo>
              ∗
             </mo>
<mi>
              m
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              d
             </mi>
<mi>
              i
             </mi>
<mi>
              m
             </mi>
<mi>
              e
             </mi>
<mi>
              n
             </mi>
<mi>
              s
             </mi>
<mi>
              i
             </mi>
<mi>
              o
             </mi>
<mi>
              n
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (2**m, dimension)
            </annotation>
</semantics>
</math> -->( 2 ∗ ∗ m , d i m e n s i o n ) (2**m, dimension)( 2 ∗ ∗ m , d im e n s i o n )  . 

Parameters
:   * **m** ( *Int*  ) – The (base2) exponent of the number of points to draw.
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – The output tensor
* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of the
returned tensor.
Default: `None`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

fast_forward ( *n* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/quasirandom.py#L169) 
:   Function to fast-forward the state of the `SobolEngine`  by `n`  steps. This is equivalent to drawing `n`  samples
without using the samples. 

Parameters
: **n** ( *Int*  ) – The number of steps to fast-forward by.

reset ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/quasirandom.py#L161) 
:   Function to reset the `SobolEngine`  to base state.

