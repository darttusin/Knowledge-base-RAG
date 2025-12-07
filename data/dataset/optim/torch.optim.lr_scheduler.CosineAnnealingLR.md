CosineAnnealingLR 
======================================================================

*class* torch.optim.lr_scheduler. CosineAnnealingLR ( *optimizer*  , *T_max*  , *eta_min = 0.0*  , *last_epoch = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1023) 
:   Set the learning rate of each parameter group using a cosine annealing schedule. 

The learning rate is updated recursively using: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             η
            </mi>
<mrow>
<mi>
              t
             </mi>
<mo>
              +
             </mo>
<mn>
              1
             </mn>
</mrow>
</msub>
<mo>
            =
           </mo>
<msub>
<mi>
             η
            </mi>
<mi>
             min
            </mi>
<mo>
             ⁡
            </mo>
</msub>
<mo>
            +
           </mo>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             η
            </mi>
<mi>
             t
            </mi>
</msub>
<mo>
            −
           </mo>
<msub>
<mi>
             η
            </mi>
<mi>
             min
            </mi>
<mo>
             ⁡
            </mo>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            ⋅
           </mo>
<mfrac>
<mrow>
<mn>
              1
             </mn>
<mo>
              +
             </mo>
<mi>
              cos
             </mi>
<mo>
              ⁡
             </mo>
<mrow>
<mo fence="true">
               (
              </mo>
<mfrac>
<mrow>
<mo stretchy="false">
                 (
                </mo>
<msub>
<mi>
                  T
                 </mi>
<mrow>
<mi>
                   c
                  </mi>
<mi>
                   u
                  </mi>
<mi>
                   r
                  </mi>
</mrow>
</msub>
<mo>
                 +
                </mo>
<mn>
                 1
                </mn>
<mo stretchy="false">
                 )
                </mo>
<mi>
                 π
                </mi>
</mrow>
<msub>
<mi>
                 T
                </mi>
<mrow>
<mi>
                  m
                 </mi>
<mi>
                  a
                 </mi>
<mi>
                  x
                 </mi>
</mrow>
</msub>
</mfrac>
<mo fence="true">
               )
              </mo>
</mrow>
</mrow>
<mrow>
<mn>
              1
             </mn>
<mo>
              +
             </mo>
<mi>
              cos
             </mi>
<mo>
              ⁡
             </mo>
<mrow>
<mo fence="true">
               (
              </mo>
<mfrac>
<mrow>
<msub>
<mi>
                  T
                 </mi>
<mrow>
<mi>
                   c
                  </mi>
<mi>
                   u
                  </mi>
<mi>
                   r
                  </mi>
</mrow>
</msub>
<mi>
                 π
                </mi>
</mrow>
<msub>
<mi>
                 T
                </mi>
<mrow>
<mi>
                  m
                 </mi>
<mi>
                  a
                 </mi>
<mi>
                  x
                 </mi>
</mrow>
</msub>
</mfrac>
<mo fence="true">
               )
              </mo>
</mrow>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           eta_{t+1} = eta_{min} + (eta_t - eta_{min}) cdot
frac{1 + cosleft(frac{(T_{cur}+1) pi}{T_{max}}right)}
    {1 + cosleft(frac{T_{cur} pi}{T_{max}}right)}
          </annotation>
</semantics>
</math> -->
η t + 1 = η min ⁡ + ( η t − η min ⁡ ) ⋅ 1 + cos ⁡ ( ( T c u r + 1 ) π T m a x ) 1 + cos ⁡ ( T c u r π T m a x ) eta_{t+1} = eta_{min} + (eta_t - eta_{min}) cdot
frac{1 + cosleft(frac{(T_{cur}+1) pi}{T_{max}}right)}
 {1 + cosleft(frac{T_{cur} pi}{T_{max}}right)}

η t + 1 ​ = η m i n ​ + ( η t ​ − η m i n ​ ) ⋅ 1 + cos ( T ma x ​ T c u r ​ π ​ ) 1 + cos ( T ma x ​ ( T c u r ​ + 1 ) π ​ ) ​

This implements a recursive approximation of the closed-form schedule proposed in [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)  : 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             η
            </mi>
<mi>
             t
            </mi>
</msub>
<mo>
            =
           </mo>
<msub>
<mi>
             η
            </mi>
<mi>
             min
            </mi>
<mo>
             ⁡
            </mo>
</msub>
<mo>
            +
           </mo>
<mfrac>
<mn>
             1
            </mn>
<mn>
             2
            </mn>
</mfrac>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             η
            </mi>
<mi>
             max
            </mi>
<mo>
             ⁡
            </mo>
</msub>
<mo>
            −
           </mo>
<msub>
<mi>
             η
            </mi>
<mi>
             min
            </mi>
<mo>
             ⁡
            </mo>
</msub>
<mo stretchy="false">
            )
           </mo>
<mrow>
<mo fence="true">
             (
            </mo>
<mn>
             1
            </mn>
<mo>
             +
            </mo>
<mi>
             cos
            </mi>
<mo>
             ⁡
            </mo>
<mrow>
<mo fence="true">
              (
             </mo>
<mfrac>
<mrow>
<msub>
<mi>
                 T
                </mi>
<mrow>
<mi>
                  c
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  r
                 </mi>
</mrow>
</msub>
<mi>
                π
               </mi>
</mrow>
<msub>
<mi>
                T
               </mi>
<mrow>
<mi>
                 m
                </mi>
<mi>
                 a
                </mi>
<mi>
                 x
                </mi>
</mrow>
</msub>
</mfrac>
<mo fence="true">
              )
             </mo>
</mrow>
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           eta_t = eta_{min} + frac{1}{2}(eta_{max} - eta_{min}) left(
    1 + cosleft(frac{T_{cur} pi}{T_{max}}right) right)
          </annotation>
</semantics>
</math> -->
η t = η min ⁡ + 1 2 ( η max ⁡ − η min ⁡ ) ( 1 + cos ⁡ ( T c u r π T m a x ) ) eta_t = eta_{min} + frac{1}{2}(eta_{max} - eta_{min}) left(
 1 + cosleft(frac{T_{cur} pi}{T_{max}}right) right)

η t ​ = η m i n ​ + 2 1 ​ ( η m a x ​ − η m i n ​ ) ( 1 + cos ( T ma x ​ T c u r ​ π ​ ) )

where: 

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
               η
              </mi>
<mi>
               t
              </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
             eta_t
            </annotation>
</semantics>
</math> -->η t eta_tη t ​  is the learning rate at step <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              t
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             t
            </annotation>
</semantics>
</math> -->t tt

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
               T
              </mi>
<mrow>
<mi>
                c
               </mi>
<mi>
                u
               </mi>
<mi>
                r
               </mi>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
             T_{cur}
            </annotation>
</semantics>
</math> -->T c u r T_{cur}T c u r ​  is the number of epochs since the last restart

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
               T
              </mi>
<mrow>
<mi>
                m
               </mi>
<mi>
                a
               </mi>
<mi>
                x
               </mi>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
             T_{max}
            </annotation>
</semantics>
</math> -->T m a x T_{max}T ma x ​  is the maximum number of epochs in a cycle

Note 

Although SGDR includes periodic restarts, this implementation performs cosine annealing **without restarts** , so <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              T
             </mi>
<mrow>
<mi>
               c
              </mi>
<mi>
               u
              </mi>
<mi>
               r
              </mi>
</mrow>
</msub>
<mo>
             =
            </mo>
<mi>
             t
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            T_{cur} = t
           </annotation>
</semantics>
</math> -->T c u r = t T_{cur} = tT c u r ​ = t  and increases monotonically with each call
to [`step()`](#torch.optim.lr_scheduler.CosineAnnealingLR.step "torch.optim.lr_scheduler.CosineAnnealingLR.step")  .

Parameters
:   * **optimizer** ( [*Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  ) – Wrapped optimizer.
* **T_max** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Maximum number of iterations.
* **eta_min** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – Minimum learning rate. Default: 0.
* **last_epoch** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The index of the last epoch. Default: -1.

Example 

```
>>> num_epochs = 100
>>> scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
>>> for epoch in range(num_epochs):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()

```

![../_images/CosineAnnealingLR.png](../_images/CosineAnnealingLR.png)

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1084) 
:   Retrieve the learning rate of each parameter group. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L159) 
:   Load the scheduler’s state. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – scheduler state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.lr_scheduler.CosineAnnealingLR.state_dict "torch.optim.lr_scheduler.CosineAnnealingLR.state_dict")  .

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L149) 
:   Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains an entry for every variable in self.__dict__ which
is not the optimizer. 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( *epoch = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L176) 
:   Perform a step.

