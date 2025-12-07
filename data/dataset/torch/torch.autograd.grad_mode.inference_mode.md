inference_mode 
=================================================================

*class* torch.autograd.grad_mode. inference_mode ( *mode = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/grad_mode.py#L212) 
:   Context-manager that enables or disables inference mode. 

InferenceMode is a context manager analogous to [`no_grad`](torch.no_grad.html#torch.no_grad "torch.autograd.grad_mode.no_grad")  to be used when you are certain your operations will have no interactions
with autograd (e.g., model training). Code run under this mode gets better
performance by disabling view tracking and version counter bumps. Note that
unlike some other mechanisms that locally enable or disable grad,
entering inference_mode also disables to [forward-mode AD](../autograd.html#forward-mode-ad)  . 

This context manager is thread local; it will not affect computation
in other threads. 

Also functions as a decorator. 

Note 

Inference mode is one of several mechanisms that can enable or
disable gradients locally see [Locally disabling gradient computation](../notes/autograd.html#locally-disable-grad-doc)  for
more information on how they compare.

Parameters
: **mode** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *or* *function*  ) – Either a boolean flag whether to enable or
disable inference mode or a Python function to decorate with
inference mode enabled

Example::
:   ```
>>> import torch
>>> x = torch.ones(1, 2, 3, requires_grad=True)
>>> with torch.inference_mode():
...     y = x * x
>>> y.requires_grad
False
>>> y._version
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
RuntimeError: Inference tensors do not track version counter.
>>> @torch.inference_mode()
... def func(x):
...     return x * x
>>> out = func(x)
>>> out.requires_grad
False
>>> @torch.inference_mode()
... def doubler(x):
...     return x * 2
>>> out = doubler(x)
>>> out.requires_grad
False

```

clone ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/grad_mode.py#L282) 
:   Create a copy of this class 

Return type
:   [*inference_mode*](#torch.autograd.grad_mode.inference_mode "torch.autograd.grad_mode.inference_mode")

