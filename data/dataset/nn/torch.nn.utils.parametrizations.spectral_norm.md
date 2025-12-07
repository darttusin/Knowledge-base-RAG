torch.nn.utils.parametrizations.spectral_norm 
===============================================================================================================================

torch.nn.utils.parametrizations. spectral_norm ( *module*  , *name = 'weight'*  , *n_power_iterations = 1*  , *eps = 1e-12*  , *dim = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/parametrizations.py#L527) 
:   Apply spectral normalization to a parameter in the given module. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi mathvariant="bold">
             W
            </mi>
<mrow>
<mi>
              S
             </mi>
<mi>
              N
             </mi>
</mrow>
</msub>
<mo>
            =
           </mo>
<mfrac>
<mi mathvariant="bold">
             W
            </mi>
<mrow>
<mi>
              σ
             </mi>
<mo stretchy="false">
              (
             </mo>
<mi mathvariant="bold">
              W
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
<mo separator="true">
            ,
           </mo>
<mi>
            σ
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi mathvariant="bold">
            W
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<munder>
<mrow>
<mi>
              max
             </mi>
<mo>
              ⁡
             </mo>
</mrow>
<mrow>
<mi mathvariant="bold">
              h
             </mi>
<mo>
              :
             </mo>
<mi mathvariant="bold">
              h
             </mi>
<mo mathvariant="normal">
              ≠
             </mo>
<mn>
              0
             </mn>
</mrow>
</munder>
<mfrac>
<mrow>
<mi mathvariant="normal">
              ∥
             </mi>
<mi mathvariant="bold">
              W
             </mi>
<mi mathvariant="bold">
              h
             </mi>
<msub>
<mi mathvariant="normal">
               ∥
              </mi>
<mn>
               2
              </mn>
</msub>
</mrow>
<mrow>
<mi mathvariant="normal">
              ∥
             </mi>
<mi mathvariant="bold">
              h
             </mi>
<msub>
<mi mathvariant="normal">
               ∥
              </mi>
<mn>
               2
              </mn>
</msub>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           mathbf{W}_{SN} = dfrac{mathbf{W}}{sigma(mathbf{W})},
sigma(mathbf{W}) = max_{mathbf{h}: mathbf{h} ne 0} dfrac{|mathbf{W} mathbf{h}|_2}{|mathbf{h}|_2}
          </annotation>
</semantics>
</math> -->
W S N = W σ ( W ) , σ ( W ) = max ⁡ h : h ≠ 0 ∥ W h ∥ 2 ∥ h ∥ 2 mathbf{W}_{SN} = dfrac{mathbf{W}}{sigma(mathbf{W})},
sigma(mathbf{W}) = max_{mathbf{h}: mathbf{h} ne 0} dfrac{|mathbf{W} mathbf{h}|_2}{|mathbf{h}|_2}

W SN ​ = σ ( W ) W ​ , σ ( W ) = h : h  = 0 max ​ ∥ h ∥ 2 ​ ∥ Wh ∥ 2 ​ ​

When applied on a vector, it simplifies to 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi mathvariant="bold">
             x
            </mi>
<mrow>
<mi>
              S
             </mi>
<mi>
              N
             </mi>
</mrow>
</msub>
<mo>
            =
           </mo>
<mfrac>
<mi mathvariant="bold">
             x
            </mi>
<mrow>
<mi mathvariant="normal">
              ∥
             </mi>
<mi mathvariant="bold">
              x
             </mi>
<msub>
<mi mathvariant="normal">
               ∥
              </mi>
<mn>
               2
              </mn>
</msub>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           mathbf{x}_{SN} = dfrac{mathbf{x}}{|mathbf{x}|_2}
          </annotation>
</semantics>
</math> -->
x S N = x ∥ x ∥ 2 mathbf{x}_{SN} = dfrac{mathbf{x}}{|mathbf{x}|_2}

x SN ​ = ∥ x ∥ 2 ​ x ​

Spectral normalization stabilizes the training of discriminators (critics)
in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant
of the model. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            σ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           sigma
          </annotation>
</semantics>
</math> -->σ sigmaσ  is approximated performing one iteration of the [power method](https://en.wikipedia.org/wiki/Power_iteration)  every time the weight is accessed. If the dimension of the
weight tensor is greater than 2, it is reshaped to 2D in power iteration
method to get spectral norm. 

See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)  . 

Note 

This function is implemented using the parametrization functionality
in [`register_parametrization()`](torch.nn.utils.parametrize.register_parametrization.html#torch.nn.utils.parametrize.register_parametrization "torch.nn.utils.parametrize.register_parametrization")  . It is a
reimplementation of [`torch.nn.utils.spectral_norm()`](torch.nn.utils.spectral_norm.html#torch.nn.utils.spectral_norm "torch.nn.utils.spectral_norm")  .

Note 

When this constraint is registered, the singular vectors associated to the largest
singular value are estimated rather than sampled at random. These are then updated
performing `n_power_iterations`  of the [power method](https://en.wikipedia.org/wiki/Power_iteration)  whenever the tensor
is accessed with the module on *training* mode.

Note 

If the *_SpectralNorm* module, i.e., *module.parametrization.weight[idx]* ,
is in training mode on removal, it will perform another power iteration.
If you’d like to avoid this iteration, set the module to eval mode
before its removal.

Parameters
:   * **module** ( [*nn.Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  ) – containing module
* **name** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – name of weight parameter. Default: `"weight"`  .
* **n_power_iterations** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – number of power iterations to
calculate spectral norm. Default: `1`  .
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – epsilon for numerical stability in
calculating norms. Default: `1e-12`  .
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – dimension corresponding to number of outputs.
Default: `0`  , except for modules that are instances of
ConvTranspose{1,2,3}d, when it is `1`

Returns
:   The original module with a new parametrization registered to the specified
weight

Return type
:   [*Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")

Example: 

```
>>> snm = spectral_norm(nn.Linear(20, 40))
>>> snm
ParametrizedLinear(
  in_features=20, out_features=40, bias=True
  (parametrizations): ModuleDict(
    (weight): ParametrizationList(
      (0): _SpectralNorm()
    )
  )
)
>>> torch.linalg.matrix_norm(snm.weight, 2)
tensor(1.0081, grad_fn=<AmaxBackward0>)

```

