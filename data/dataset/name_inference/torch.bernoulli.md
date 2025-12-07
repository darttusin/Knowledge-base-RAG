torch.bernoulli 
==================================================================

torch. bernoulli ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *generator : Optional [ [Generator](torch.Generator.html#torch.Generator "torch.Generator") ]*  , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Draws binary random numbers (0 or 1) from a Bernoulli distribution. 

The `input`  tensor should be a tensor containing probabilities
to be used for drawing the binary random number.
Hence, all values in `input`  have to be in the range: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            0
           </mn>
<mo>
            ≤
           </mo>
<msub>
<mtext>
             input
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            ≤
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           0 leq text{input}_i leq 1
          </annotation>
</semantics>
</math> -->0 ≤ input i ≤ 1 0 leq text{input}_i leq 10 ≤ input i ​ ≤ 1  . 

The <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
             i
            </mtext>
<mrow>
<mi>
              t
             </mi>
<mi>
              h
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           text{i}^{th}
          </annotation>
</semantics>
</math> -->i t h text{i}^{th}i t h  element of the output tensor will draw a
value <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           1
          </annotation>
</semantics>
</math> -->1 11  according to the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
             i
            </mtext>
<mrow>
<mi>
              t
             </mi>
<mi>
              h
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           text{i}^{th}
          </annotation>
</semantics>
</math> -->i t h text{i}^{th}i t h  probability value given
in `input`  . 

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
<mrow>
<mi mathvariant="normal">
             B
            </mi>
<mi mathvariant="normal">
             e
            </mi>
<mi mathvariant="normal">
             r
            </mi>
<mi mathvariant="normal">
             n
            </mi>
<mi mathvariant="normal">
             o
            </mi>
<mi mathvariant="normal">
             u
            </mi>
<mi mathvariant="normal">
             l
            </mi>
<mi mathvariant="normal">
             l
            </mi>
<mi mathvariant="normal">
             i
            </mi>
</mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            p
           </mi>
<mo>
            =
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
           text{out}_{i} sim mathrm{Bernoulli}(p = text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i ∼ B e r n o u l l i ( p = input i ) text{out}_{i} sim mathrm{Bernoulli}(p = text{input}_{i})

out i ​ ∼ Bernoulli ( p = input i ​ )

The returned `out`  tensor only has values 0 or 1 and is of the same
shape as `input`  . 

`out`  can have integral `dtype`  , but `input`  must have floating
point `dtype`  . 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor of probability values for the Bernoulli distribution

Keyword Arguments
:   * **generator** ( [`torch.Generator`](torch.Generator.html#torch.Generator "torch.Generator")  , optional) – a pseudorandom number generator for sampling
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
>>> a
tensor([[ 0.1737,  0.0950,  0.3609],
        [ 0.7148,  0.0289,  0.2676],
        [ 0.9456,  0.8937,  0.7202]])
>>> torch.bernoulli(a)
tensor([[ 1.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.]])

>>> a = torch.ones(3, 3) # probability of drawing "1" is 1
>>> torch.bernoulli(a)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
>>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
>>> torch.bernoulli(a)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])

```

