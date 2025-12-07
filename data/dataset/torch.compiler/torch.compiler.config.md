torch.compiler.config 
=====================================================================================

This is the top-level configuration module for the compiler, containing
cross-cutting configuration options that affect all parts of the compiler
stack. 

You may also be interested in the per-component configuration modules, which
contain configuration options that affect only a specific part of the compiler: 

* `torch._dynamo.config`
* `torch._inductor.config`
* `torch._functorch.config`
* `torch.fx.experimental.config`

torch.compiler.config. job_id *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") ]* *= None* 
:   Semantically, this should be an identifier that uniquely identifies, e.g., a
training job. You might have multiple attempts of the same job, e.g., if it was
preempted or needed to be restarted, but each attempt should be running
substantially the same workload with the same distributed topology. You can
set this by environment variable with `TORCH_COMPILE_JOB_ID`  . 

Operationally, this controls the effect of profile-guided optimization related
persistent state. PGO state can affect how we perform compilation across
multiple invocations of PyTorch, e.g., the first time you run your program we
may compile twice as we discover what inputs are dynamic, and then PGO will
save this state so subsequent invocations only need to compile once, because
they remember it is dynamic. This profile information, however, is sensitive
to what workload you are running, so we require you to tell us that two jobs
are *related*  (i.e., are the same workload) before we are willing to reuse
this information. Notably, PGO does nothing (even if explicitly enabled)
unless a valid `job_id`  is available. In some situations, PyTorch can
configured to automatically compute a `job_id`  based on the environment it
is running in. 

Profiles are always collected on a per rank basis, so different ranks may have
different profiles. If you know your workload is truly SPMD, you can run with `torch._dynamo.config.enable_compiler_collectives`  to ensure nodes get
consistent profiles across all ranks.

