torch.poisson 
==============================================================

torch. poisson ( *input*  , *generator = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a tensor of the same size as `input`  with each element
sampled from a Poisson distribution with rate parameter given by the corresponding
element in `input`  i.e., 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            ∼
           </mo>
<mtext>
            Poisson
           </mtext>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             input
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i sim text{Poisson}(text{input}_i)
          </annotation>
</semantics>
</math> -->
out i ∼ Poisson ( input i ) text{out}_i sim text{Poisson}(text{input}_i)

out i ​ ∼ Poisson ( input i ​ )

`input`  must be non-negative. 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor containing the rates of the Poisson distribution

Keyword Arguments
: **generator** ( [`torch.Generator`](torch.Generator.html#torch.Generator "torch.Generator")  , optional) – a pseudorandom number generator for sampling

Example: 

```
>>> rates = torch.rand(4, 4) * 5  # rate parameter between 0 and 5
>>> torch.poisson(rates)
tensor([[9., 1., 3., 5.],
        [8., 6., 6., 0.],
        [0., 4., 5., 3.],
        [2., 1., 4., 2.]])

```

