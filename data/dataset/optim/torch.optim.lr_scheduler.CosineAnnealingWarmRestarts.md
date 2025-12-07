CosineAnnealingWarmRestarts 
==========================================================================================

*class* torch.optim.lr_scheduler. CosineAnnealingWarmRestarts ( *optimizer*  , *T_0*  , *T_mult = 1*  , *eta_min = 0.0*  , *last_epoch = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1712) 
:   Set the learning rate of each parameter group using a cosine annealing schedule. 

The <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             η
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
           eta_{max}
          </annotation>
</semantics>
</math> -->η m a x eta_{max}η ma x ​  is set to the initial lr, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->T c u r T_{cur}T c u r ​  is the number of epochs since the last restart and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             T
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           T_{i}
          </annotation>
</semantics>
</math> -->T i T_{i}T i ​  is the number
of epochs between two warm restarts in SGDR: 

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
<mrow>
<mi>
              m
             </mi>
<mi>
              i
             </mi>
<mi>
              n
             </mi>
</mrow>
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
<mo>
            −
           </mo>
<msub>
<mi>
             η
            </mi>
<mrow>
<mi>
              m
             </mi>
<mi>
              i
             </mi>
<mi>
              n
             </mi>
</mrow>
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
<msub>
<mi>
                T
               </mi>
<mi>
                i
               </mi>
</msub>
</mfrac>
<mi>
              π
             </mi>
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
           eta_t = eta_{min} + frac{1}{2}(eta_{max} - eta_{min})left(1 +
cosleft(frac{T_{cur}}{T_{i}}piright)right)
          </annotation>
</semantics>
</math> -->
η t = η m i n + 1 2 ( η m a x − η m i n ) ( 1 + cos ⁡ ( T c u r T i π ) ) eta_t = eta_{min} + frac{1}{2}(eta_{max} - eta_{min})left(1 +
cosleft(frac{T_{cur}}{T_{i}}piright)right)

η t ​ = η min ​ + 2 1 ​ ( η ma x ​ − η min ​ ) ( 1 + cos ( T i ​ T c u r ​ ​ π ) )

When <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msub>
<mi>
             T
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           T_{cur}=T_{i}
          </annotation>
</semantics>
</math> -->T c u r = T i T_{cur}=T_{i}T c u r ​ = T i ​  , set <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mrow>
<mi>
              m
             </mi>
<mi>
              i
             </mi>
<mi>
              n
             </mi>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           eta_t = eta_{min}
          </annotation>
</semantics>
</math> -->η t = η m i n eta_t = eta_{min}η t ​ = η min ​  .
When <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mn>
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           T_{cur}=0
          </annotation>
</semantics>
</math> -->T c u r = 0 T_{cur}=0T c u r ​ = 0  after restart, set <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
           eta_t=eta_{max}
          </annotation>
</semantics>
</math> -->η t = η m a x eta_t=eta_{max}η t ​ = η ma x ​  . 

It has been proposed in [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)  . 

Parameters
:   * **optimizer** ( [*Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  ) – Wrapped optimizer.
* **T_0** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of iterations until the first restart.
* **T_mult** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – A factor by which <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 T
                </mi>
<mi>
                 i
                </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
               T_{i}
              </annotation>
</semantics>
</math> -->T i T_{i}T i ​  increases after a restart. Default: 1.

* **eta_min** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Minimum learning rate. Default: 0.
* **last_epoch** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – The index of the last epoch. Default: -1.

Example 

```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
>>> scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
...     optimizer, T_0=20
... )
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()

```

![../_images/CosineAnnealingWarmRestarts.png](../_images/CosineAnnealingWarmRestarts.png)

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1776) 
:   Compute the initial learning rate. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L159) 
:   Load the scheduler’s state. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – scheduler state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict")  .

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L149) 
:   Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains an entry for every variable in self.__dict__ which
is not the optimizer. 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( *epoch = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1789) 
:   Step could be called after every batch update. 

Example 

```
>>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
>>> iters = len(dataloader)
>>> for epoch in range(20):
>>>     for i, sample in enumerate(dataloader):
>>>         inputs, labels = sample['inputs'], sample['labels']
>>>         optimizer.zero_grad()
>>>         outputs = net(inputs)
>>>         loss = criterion(outputs, labels)
>>>         loss.backward()
>>>         optimizer.step()
>>>         scheduler.step(epoch + i / iters)

```

This function can be called in an interleaved way. 

Example 

```
>>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
>>> for epoch in range(20):
>>>     scheduler.step()
>>> scheduler.step(26)
>>> scheduler.step()  # scheduler.step(27), instead of scheduler(20)

```

