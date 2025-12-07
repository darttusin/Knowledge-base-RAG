Dealing with Recompilations 
==========================================================================================

Recompilations are necessary for `torch.compile`  soundness, but can result in significantly increased compile time.
Thus, minimizing recompilations while preserving soundness is essential for reducing compile time. 

You can view recompilations and their reasons using tlparse or `TORCH_LOGS=recompiles`  . 

Is Dynamic Shapes Enabled? 
---------------------------------------------------------------------------------------

In the below example, we recompile due to mismatched shapes: 

```
@torch.compile
def fn(x):
    return x + 1
fn(torch.ones(3))
fn(torch.ones(4))

```

```
Recompiling function fn in /tmp/ipykernel_231501/2479206322.py:1
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 3, actual 4

```

```
tensor([2., 2., 2., 2.])

```

Make sure that the dynamic option of `torch.compile`  is not set to `False`  .
The default option, `dynamic=None`  , will only attempt dynamic shapes after the first compilation.
You can set `dynamic=True`  to upfront compile as dynamic as possible: 

```
@torch.compile(dynamic=True)
def gn(x):
    return x + 1
gn(torch.ones(3))
gn(torch.ones(4))

```

```
tensor([2., 2., 2., 2.])

```

For more information on dynamic shapes, including dealing with errors/recompilations due to
dynamic shapes, see [the dynamic shapes manual](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng)  .

Wrapping Constants with Tensors 
--------------------------------------------------------------------------------------------------

By default, `int`  / `float`  variables are treated as constants and are guarded on their exact value.
In the below example, we have a recompilation for each function call. 

```
@torch.compile
def fn(x, c):
    return x + c
for i in range(5):
    fn(torch.ones(i), 0.5 + i)

```

```
Recompiling function fn in /tmp/ipykernel_231501/3647755280.py:1
    triggered by the following guard failure(s):
    - 2/0: c == 0.5                                               
Recompiling function fn in /tmp/ipykernel_231501/3647755280.py:1
    triggered by the following guard failure(s):
    - 2/1: tensor 'x' size mismatch at index 0. expected 1, actual 2
    - 2/0: c == 0.5                                               

```

In particular, for LR schedulers, initializing with a constant can lead to recompilations: 

```
mod = torch.nn.Linear(3, 3)
opt = torch.optim.Adam(mod.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)
@torch.compile
def gn(inp):
    opt.zero_grad(True)
    out = mod(inp).sum()
    out.backward()
    opt.step()
    sched.step()
for i in range(5):
    gn(torch.ones(3, 3))

```

```
Profiler function <class 'torch.autograd.profiler.record_function'> will be ignored
Recompiling function step in /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/optim/adam.py:213
    triggered by the following guard failure(s):
    - 7/0: self.param_groups[0]['lr'] == 0.01                     

```

In both examples, we can wrap `float`  variables in tensors in order to prevent recompilations. 

```
# first example
for i in range(5):
    fn(torch.ones(i), torch.tensor(0.5 + i))
# second example
opt = torch.optim.Adam(mod.parameters(), lr=torch.tensor(0.01))
sched = torch.optim.lr_scheduler.ExponentialLR(opt, torch.tensor(0.9))
for i in range(5):
    gn(torch.ones(3, 3))

```

```
Recompiling function fn in /tmp/ipykernel_231501/3647755280.py:1
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 0, actual 1
Recompiling function fn in /tmp/ipykernel_231501/3647755280.py:1
    triggered by the following guard failure(s):
    - 0/1: tensor 'x' size mismatch at index 0. expected 1, actual 2
    - 0/0: tensor 'x' size mismatch at index 0. expected 0, actual 2

```

Changing the Cache Size Limit 
----------------------------------------------------------------------------------------------

There is a limit to how many times a function can be recompiled,
determined by `torch._dynamo.config.cache_size_limit`  and `torch._dynamo.config.accumulated_cache_size_limit`  (The exact difference between these 2 values is detailed in [`torch/_dynamo/cache_size.py`](https://github.com/pytorch/pytorch/blob/4ce6e6ec8890a3f6ee604c9efb3ff153825ce575/torch/_dynamo/cache_size.py#L14)  ).
If the Dynamo cache limit is hit, then all future compilation attempts **will result in the function being skipped (run eagerly)** .
Dynamo will still attempt to use previously compiled bytecode for future function calls, if the guards pass.
Note that in the case of a recompilation limit hit, **all nested function calls WILL be skipped** (Dynamo will try to use previously compiled bytecode for the nested functions).
Dynamo will also issue a warning containing the affected function and which limit was hit.
In the example below, each function call results in a recompile attempt.
When we hit the cache size limit (by default, 8), we stop attempting to recompile.
(Note that we set `dynamic=False`  for demonstration purposes to force recompilation every time). 

```
@torch.compile(dynamic=False)
def fn(x):
    return x + 1
for i in range(1, 10):
    # recompile every time due to dynamic=False
    fn(torch.ones(i))

```

```
Recompiling function fn in /tmp/ipykernel_231501/3054308037.py:1
    triggered by the following guard failure(s):
    - 8/0: tensor 'x' size mismatch at index 0. expected 1, actual 2
Recompiling function fn in /tmp/ipykernel_231501/3054308037.py:1
    triggered by the following guard failure(s):
    - 8/1: tensor 'x' size mismatch at index 0. expected 2, actual 3
    - 8/0: tensor 'x' size mismatch at index 0. expected 1, actual 3
Recompiling function fn in /tmp/ipykernel_231501/3054308037.py:1
    triggered by the following guard failure(s):
    - 8/2: tensor 'x' size mismatch at index 0. expected 3, actual 4
    - 8/1: tensor 'x' size mismatch at index 0. expected 2, actual 4
    - 8/0: tensor 'x' size mismatch at index 0. expected 1, actual 4
Recompiling function fn in /tmp/ipykernel_231501/3054308037.py:1
    triggered by the following guard failure(s):
    - 8/3: tensor 'x' size mismatch at index 0. expected 4, actual 5
    - 8/2: tensor 'x' size mismatch at index 0. expected 3, actual 5
    - 8/1: tensor 'x' size mismatch at index 0. expected 2, actual 5
    - 8/0: tensor 'x' size mismatch at index 0. expected 1, actual 5
Recompiling function fn in /tmp/ipykernel_231501/3054308037.py:1
    triggered by the following guard failure(s):
    - 8/4: tensor 'x' size mismatch at index 0. expected 5, actual 6
    - 8/3: tensor 'x' size mismatch at index 0. expected 4, actual 6
    - 8/2: tensor 'x' size mismatch at index 0. expected 3, actual 6
    - 8/1: tensor 'x' size mismatch at index 0. expected 2, actual 6
    - 8/0: tensor 'x' size mismatch at index 0. expected 1, actual 6
Recompiling function fn in /tmp/ipykernel_231501/3054308037.py:1
    triggered by the following guard failure(s):
    - 8/5: tensor 'x' size mismatch at index 0. expected 6, actual 7
    - 8/4: tensor 'x' size mismatch at index 0. expected 5, actual 7
    - 8/3: tensor 'x' size mismatch at index 0. expected 4, actual 7
    - 8/2: tensor 'x' size mismatch at index 0. expected 3, actual 7
    - 8/1: tensor 'x' size mismatch at index 0. expected 2, actual 7
    - 8/0: tensor 'x' size mismatch at index 0. expected 1, actual 7
Recompiling function fn in /tmp/ipykernel_231501/3054308037.py:1
    triggered by the following guard failure(s):
    - 8/6: tensor 'x' size mismatch at index 0. expected 7, actual 8
    - 8/5: tensor 'x' size mismatch at index 0. expected 6, actual 8
    - 8/4: tensor 'x' size mismatch at index 0. expected 5, actual 8
    - 8/3: tensor 'x' size mismatch at index 0. expected 4, actual 8
    - 8/2: tensor 'x' size mismatch at index 0. expected 3, actual 8
    - 8/1: tensor 'x' size mismatch at index 0. expected 2, actual 8
    - 8/0: tensor 'x' size mismatch at index 0. expected 1, actual 8
Recompiling function fn in /tmp/ipykernel_231501/3054308037.py:1
    triggered by the following guard failure(s):
    - 8/7: tensor 'x' size mismatch at index 0. expected 8, actual 9
    - 8/6: tensor 'x' size mismatch at index 0. expected 7, actual 9
    - 8/5: tensor 'x' size mismatch at index 0. expected 6, actual 9
    - 8/4: tensor 'x' size mismatch at index 0. expected 5, actual 9
    - 8/3: tensor 'x' size mismatch at index 0. expected 4, actual 9
    - 8/2: tensor 'x' size mismatch at index 0. expected 3, actual 9
    - 8/1: tensor 'x' size mismatch at index 0. expected 2, actual 9
    - 8/0: tensor 'x' size mismatch at index 0. expected 1, actual 9
torch._dynamo hit config.recompile_limit (8)
   function: 'fn' (/tmp/ipykernel_231501/3054308037.py:1)
   last reason: 8/7: tensor 'x' size mismatch at index 0. expected 8, actual 9
To log all recompilation reasons, use TORCH_LOGS="recompiles".
To diagnose recompilation issues, see https://localhost:8000/docs/main/torch.compiler_troubleshooting.html.

```

If you know that the number of recompilations has a reasonable constant upper bound, you can raise the cache size limit.
If the cost of recompilation outweighs the benefit of compilation, then you can consider lowering the cache size limit. 

```
torch._dynamo.config.cache_size_limit = 16
@torch.compile(dynamic=False)
def gn(x):
    return x + 1
for i in range(1, 10):
    gn(torch.ones(i))

```

```
Recompiling function gn in /tmp/ipykernel_231501/887097224.py:2
    triggered by the following guard failure(s):
    - 9/0: tensor 'x' size mismatch at index 0. expected 1, actual 2
Recompiling function gn in /tmp/ipykernel_231501/887097224.py:2
    triggered by the following guard failure(s):
    - 9/1: tensor 'x' size mismatch at index 0. expected 2, actual 3
    - 9/0: tensor 'x' size mismatch at index 0. expected 1, actual 3
Recompiling function gn in /tmp/ipykernel_231501/887097224.py:2
    triggered by the following guard failure(s):
    - 9/2: tensor 'x' size mismatch at index 0. expected 3, actual 4
    - 9/1: tensor 'x' size mismatch at index 0. expected 2, actual 4
    - 9/0: tensor 'x' size mismatch at index 0. expected 1, actual 4
Recompiling function gn in /tmp/ipykernel_231501/887097224.py:2
    triggered by the following guard failure(s):
    - 9/3: tensor 'x' size mismatch at index 0. expected 4, actual 5
    - 9/2: tensor 'x' size mismatch at index 0. expected 3, actual 5
    - 9/1: tensor 'x' size mismatch at index 0. expected 2, actual 5
    - 9/0: tensor 'x' size mismatch at index 0. expected 1, actual 5
Recompiling function gn in /tmp/ipykernel_231501/887097224.py:2
    triggered by the following guard failure(s):
    - 9/4: tensor 'x' size mismatch at index 0. expected 5, actual 6
    - 9/3: tensor 'x' size mismatch at index 0. expected 4, actual 6
    - 9/2: tensor 'x' size mismatch at index 0. expected 3, actual 6
    - 9/1: tensor 'x' size mismatch at index 0. expected 2, actual 6
    - 9/0: tensor 'x' size mismatch at index 0. expected 1, actual 6
Recompiling function gn in /tmp/ipykernel_231501/887097224.py:2
    triggered by the following guard failure(s):
    - 9/5: tensor 'x' size mismatch at index 0. expected 6, actual 7
    - 9/4: tensor 'x' size mismatch at index 0. expected 5, actual 7
    - 9/3: tensor 'x' size mismatch at index 0. expected 4, actual 7
    - 9/2: tensor 'x' size mismatch at index 0. expected 3, actual 7
    - 9/1: tensor 'x' size mismatch at index 0. expected 2, actual 7
    - 9/0: tensor 'x' size mismatch at index 0. expected 1, actual 7
Recompiling function gn in /tmp/ipykernel_231501/887097224.py:2
    triggered by the following guard failure(s):
    - 9/6: tensor 'x' size mismatch at index 0. expected 7, actual 8
    - 9/5: tensor 'x' size mismatch at index 0. expected 6, actual 8
    - 9/4: tensor 'x' size mismatch at index 0. expected 5, actual 8
    - 9/3: tensor 'x' size mismatch at index 0. expected 4, actual 8
    - 9/2: tensor 'x' size mismatch at index 0. expected 3, actual 8
    - 9/1: tensor 'x' size mismatch at index 0. expected 2, actual 8
    - 9/0: tensor 'x' size mismatch at index 0. expected 1, actual 8
Recompiling function gn in /tmp/ipykernel_231501/887097224.py:2
    triggered by the following guard failure(s):
    - 9/7: tensor 'x' size mismatch at index 0. expected 8, actual 9
    - 9/6: tensor 'x' size mismatch at index 0. expected 7, actual 9
    - 9/5: tensor 'x' size mismatch at index 0. expected 6, actual 9
    - 9/4: tensor 'x' size mismatch at index 0. expected 5, actual 9
    - 9/3: tensor 'x' size mismatch at index 0. expected 4, actual 9
    - 9/2: tensor 'x' size mismatch at index 0. expected 3, actual 9
    - 9/1: tensor 'x' size mismatch at index 0. expected 2, actual 9
    - 9/0: tensor 'x' size mismatch at index 0. expected 1, actual 9

```

Graph Breaking to Reduce Recompilation Costs 
----------------------------------------------------------------------------------------------------------------------------

If a large graph is recompiling and causing high compile time, you can intentionally introduce
a graph break in order to reduce recompilation costs, at the expense of introducing a performance hit. 

```
def very_large_function(x):
    return x + 1

@torch.compile(dynamic=False)
def fn(x, c):
    y = very_large_function(x)  # recompiled every time
    return y + c

for i in range(1, 5):
    fn(torch.ones(3), i)

@torch.compile(dynamic=False)
def gn(x, c):
    y = very_large_function(x)  # compiled only once
    torch._dynamo.graph_break()
    return y + c  # recompiled every time

for i in range(1, 5):
    gn(torch.ones(3), i)

```

```
Recompiling function fn in /tmp/ipykernel_231501/2876112129.py:4
    triggered by the following guard failure(s):
    - 10/0: c == 1                                                 
Recompiling function fn in /tmp/ipykernel_231501/2876112129.py:4
    triggered by the following guard failure(s):
    - 10/1: c == 2                                                 
    - 10/0: c == 1                                                 
Recompiling function fn in /tmp/ipykernel_231501/2876112129.py:4
    triggered by the following guard failure(s):
    - 10/2: c == 3                                                 
    - 10/1: c == 2                                                 
    - 10/0: c == 1                                                 
Recompiling function torch_dynamo_resume_in_gn_at_15 in /tmp/ipykernel_231501/2876112129.py:15
    triggered by the following guard failure(s):
    - 12/0: c == 1                                                 
Recompiling function torch_dynamo_resume_in_gn_at_15 in /tmp/ipykernel_231501/2876112129.py:15
    triggered by the following guard failure(s):
    - 12/1: c == 2                                                 
    - 12/0: c == 1                                                 
Recompiling function torch_dynamo_resume_in_gn_at_15 in /tmp/ipykernel_231501/2876112129.py:15
    triggered by the following guard failure(s):
    - 12/2: c == 3                                                 
    - 12/1: c == 2                                                 
    - 12/0: c == 1                                                 

```

