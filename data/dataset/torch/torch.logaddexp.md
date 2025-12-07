torch.logaddexp 
==================================================================

torch. logaddexp ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Logarithm of the sum of exponentiations of the inputs. 

Calculates pointwise <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mrow>
<mo fence="true">
             (
            </mo>
<msup>
<mi>
              e
             </mi>
<mi>
              x
             </mi>
</msup>
<mo>
             +
            </mo>
<msup>
<mi>
              e
             </mi>
<mi>
              y
             </mi>
</msup>
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           logleft(e^x + e^yright)
          </annotation>
</semantics>
</math> -->log ⁡ ( e x + e y ) logleft(e^x + e^yright)lo g ( e x + e y )  . This function is useful
in statistics where the calculated probabilities of events may be so small as to
exceed the range of normal floating point numbers. In such cases the logarithm
of the calculated probability is stored. This function allows adding
probabilities stored in such a fashion. 

This op should be disambiguated with [`torch.logsumexp()`](torch.logsumexp.html#torch.logsumexp "torch.logsumexp")  which performs a
reduction on a single tensor. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second input tensor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.logaddexp(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))
tensor([-0.3069, -0.6867, -0.8731])
>>> torch.logaddexp(torch.tensor([-100.0, -200, -300]), torch.tensor([-1.0, -2, -3]))
tensor([-1., -2., -3.])
>>> torch.logaddexp(torch.tensor([1.0, 2000, 30000]), torch.tensor([-1.0, -2, -3]))
tensor([1.1269e+00, 2.0000e+03, 3.0000e+04])

```

