SyncBatchNorm 
==============================================================

*class* torch.nn. SyncBatchNorm ( *num_features*  , *eps = 1e-05*  , *momentum = 0.1*  , *affine = True*  , *track_running_stats = True*  , *process_group = None*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/batchnorm.py#L600) 
:   Applies Batch Normalization over a N-Dimensional input. 

The N-D input is a mini-batch of [N-2]D inputs with additional channel dimension) as described in the paper [Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift](https://arxiv.org/abs/1502.03167)  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            =
           </mo>
<mfrac>
<mrow>
<mi>
              x
             </mi>
<mo>
              −
             </mo>
<mi mathvariant="normal">
              E
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              ]
             </mo>
</mrow>
<msqrt>
<mrow>
<mrow>
<mi mathvariant="normal">
                V
               </mi>
<mi mathvariant="normal">
                a
               </mi>
<mi mathvariant="normal">
                r
               </mi>
</mrow>
<mo stretchy="false">
               [
              </mo>
<mi>
               x
              </mi>
<mo stretchy="false">
               ]
              </mo>
<mo>
               +
              </mo>
<mi>
               ϵ
              </mi>
</mrow>
</msqrt>
</mfrac>
<mo>
            ∗
           </mo>
<mi>
            γ
           </mi>
<mo>
            +
           </mo>
<mi>
            β
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y = frac{x - mathrm{E}[x]}{ sqrt{mathrm{Var}[x] + epsilon}} * gamma + beta
          </annotation>
</semantics>
</math> -->
y = x − E [ x ] V a r [ x ] + ϵ ∗ γ + β y = frac{x - mathrm{E}[x]}{ sqrt{mathrm{Var}[x] + epsilon}} * gamma + beta

y = Var [ x ] + ϵ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ x − E [ x ] ​ ∗ γ + β

The mean and standard-deviation are calculated per-dimension over all
mini-batches of the same process groups. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            γ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           gamma
          </annotation>
</semantics>
</math> -->γ gammaγ  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            β
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           beta
          </annotation>
</semantics>
</math> -->β betaβ  are learnable parameter vectors of size *C* (where *C* is the input size).
By default, the elements of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            γ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           gamma
          </annotation>
</semantics>
</math> -->γ gammaγ  are sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
            U
           </mi>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{U}(0, 1)
          </annotation>
</semantics>
</math> -->U ( 0 , 1 ) mathcal{U}(0, 1)U ( 0 , 1 )  and the elements of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            β
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           beta
          </annotation>
</semantics>
</math> -->β betaβ  are set to 0.
The standard-deviation is calculated via the biased estimator, equivalent to *torch.var(input, unbiased=False)* . 

Also by default, during training this layer keeps running estimates of its
computed mean and variance, which are then used for normalization during
evaluation. The running estimates are kept with a default `momentum`  of 0.1. 

If `track_running_stats`  is set to `False`  , this layer then does not
keep running estimates, and batch statistics are instead used during
evaluation time as well. 

Note 

This `momentum`  argument is different from one used in optimizer
classes and the conventional notion of momentum. Mathematically, the
update rule for running statistics here is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mover accent="true">
<mi>
               x
              </mi>
<mo>
               ^
              </mo>
</mover>
<mtext>
              new
             </mtext>
</msub>
<mo>
             =
            </mo>
<mo stretchy="false">
             (
            </mo>
<mn>
             1
            </mn>
<mo>
             −
            </mo>
<mtext>
             momentum
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             ×
            </mo>
<mover accent="true">
<mi>
              x
             </mi>
<mo>
              ^
             </mo>
</mover>
<mo>
             +
            </mo>
<mtext>
             momentum
            </mtext>
<mo>
             ×
            </mo>
<msub>
<mi>
              x
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            hat{x}_text{new} = (1 - text{momentum}) times hat{x} + text{momentum} times x_t
           </annotation>
</semantics>
</math> -->x ^ new = ( 1 − momentum ) × x ^ + momentum × x t hat{x}_text{new} = (1 - text{momentum}) times hat{x} + text{momentum} times x_tx ^ new ​ = ( 1 − momentum ) × x ^ + momentum × x t ​  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mover accent="true">
<mi>
              x
             </mi>
<mo>
              ^
             </mo>
</mover>
</mrow>
<annotation encoding="application/x-tex">
            hat{x}
           </annotation>
</semantics>
</math> -->x ^ hat{x}x ^  is the estimated statistic and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              x
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            x_t
           </annotation>
</semantics>
</math> -->x t x_tx t ​  is the
new observed value.

Because the Batch Normalization is done for each channel in the `C`  dimension, computing
statistics on `(N, +)`  slices, it’s common terminology to call this Volumetric Batch
Normalization or Spatio-temporal Batch Normalization. 

Currently [`SyncBatchNorm`](#torch.nn.SyncBatchNorm "torch.nn.SyncBatchNorm")  only supports `DistributedDataParallel`  (DDP) with single GPU per process. Use [`torch.nn.SyncBatchNorm.convert_sync_batchnorm()`](#torch.nn.SyncBatchNorm.convert_sync_batchnorm "torch.nn.SyncBatchNorm.convert_sync_batchnorm")  to convert `BatchNorm*D`  layer to [`SyncBatchNorm`](#torch.nn.SyncBatchNorm "torch.nn.SyncBatchNorm")  before wrapping
Network with DDP. 

Parameters
:   * **num_features** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                C
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               C
              </annotation>
</semantics>
</math> -->C CC  from an expected input of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mo>
                +
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, +)
              </annotation>
</semantics>
</math> -->( N , C , + ) (N, C, +)( N , C , + )

* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – a value added to the denominator for numerical stability.
Default: `1e-5`
* **momentum** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]*  ) – the value used for the running_mean and running_var
computation. Can be set to `None`  for cumulative moving average
(i.e. simple average). Default: 0.1
* **affine** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this module has
learnable affine parameters. Default: `True`
* **track_running_stats** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this
module tracks the running mean and variance, and when set to `False`  ,
this module does not track such statistics, and initializes statistics
buffers `running_mean`  and `running_var`  as `None`  .
When these buffers are `None`  , this module always uses batch statistics.
in both training and eval modes. Default: `True`
* **process_group** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]*  ) – synchronization of stats happen within each process group
individually. Default behavior is synchronization across the whole
world

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mo>
                +
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, +)
              </annotation>
</semantics>
</math> -->( N , C , + ) (N, C, +)( N , C , + )

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mo>
                +
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, +)
              </annotation>
</semantics>
</math> -->( N , C , + ) (N, C, +)( N , C , + )  (same shape as input)

Note 

Synchronization of batchnorm statistics occurs only while training, i.e.
synchronization is disabled when `model.eval()`  is set or if `self.training`  is otherwise `False`  .

Examples: 

```
>>> # With Learnable Parameters
>>> m = nn.SyncBatchNorm(100)
>>> # creating process group (optional)
>>> # ranks is a list of int identifying rank ids.
>>> ranks = list(range(8))
>>> r1, r2 = ranks[:4], ranks[4:]
>>> # Note: every rank calls into new_group for every
>>> # process group created, even if that rank is not
>>> # part of the group.
>>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
>>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
>>> # Without Learnable Parameters
>>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
>>> input = torch.randn(20, 100, 35, 45, 10)
>>> output = m(input)

>>> # network is nn.BatchNorm layer
>>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
>>> # only single gpu per process is currently supported
>>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
>>>                         sync_bn_network,
>>>                         device_ids=[args.local_rank],
>>>                         output_device=args.local_rank)

```

*classmethod* convert_sync_batchnorm ( *module*  , *process_group = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/batchnorm.py#L823) 
:   Converts all `BatchNorm*D`  layers in the model to [`torch.nn.SyncBatchNorm`](#torch.nn.SyncBatchNorm "torch.nn.SyncBatchNorm")  layers. 

Parameters
:   * **module** ( [*nn.Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  ) – module containing one or more `BatchNorm*D`  layers
* **process_group** ( *optional*  ) – process group to scope synchronization,
default is the whole world

Returns
:   The original `module`  with the converted [`torch.nn.SyncBatchNorm`](#torch.nn.SyncBatchNorm "torch.nn.SyncBatchNorm")  layers. If the original `module`  is a `BatchNorm*D`  layer,
a new [`torch.nn.SyncBatchNorm`](#torch.nn.SyncBatchNorm "torch.nn.SyncBatchNorm")  layer object will be returned
instead.

Example: 

```
>>> # Network with nn.BatchNorm layer
>>> module = torch.nn.Sequential(
>>>            torch.nn.Linear(20, 100),
>>>            torch.nn.BatchNorm1d(100),
>>>          ).cuda()
>>> # creating process group (optional)
>>> # ranks is a list of int identifying rank ids.
>>> ranks = list(range(8))
>>> r1, r2 = ranks[:4], ranks[4:]
>>> # Note: every rank calls into new_group for every
>>> # process group created, even if that rank is not
>>> # part of the group.
>>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
>>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
>>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

```

