TorchScript Unsupported PyTorch Constructs 
========================================================================================================================

Torch and Tensor Unsupported Attributes 
------------------------------------------------------------------------------------------------------------------

TorchScript supports most methods defined on `torch`  and `torch.Tensor`  , but we do not have full coverage.
Here are specific known ops and categories of ops which have diverging behavior between
Python and TorchScript. If you encounter something else that is not supported please
file a GitHub issue. Deprecated ops are not listed below. 

### Unsupported Tensor Methods 

### Unsupported Tensor Properties 

### Functions Not Correctly Bound on Torch 

The following functions will fail if used in TorchScript, either because they
are not bound on *torch* or because Python expects a different schema than
TorchScript. 

> * [`torch.tensordot()`](generated/torch.tensordot.html#torch.tensordot "torch.tensordot")
> * [`torch.nn.init.calculate_gain()`](nn.init.html#torch.nn.init.calculate_gain "torch.nn.init.calculate_gain")
> * [`torch.nn.init.eye_()`](nn.init.html#torch.nn.init.eye_ "torch.nn.init.eye_")
> * [`torch.nn.init.dirac_()`](nn.init.html#torch.nn.init.dirac_ "torch.nn.init.dirac_")
> * [`torch.nn.init.kaiming_normal_()`](nn.init.html#torch.nn.init.kaiming_normal_ "torch.nn.init.kaiming_normal_")
> * [`torch.nn.init.orthogonal_()`](nn.init.html#torch.nn.init.orthogonal_ "torch.nn.init.orthogonal_")
> * `torch.nn.init.sparse()`

### Ops With Divergent Schemas Between Torch & Python 

The following categories of ops have divergent schemas: 

Functions which construct tensors from non-tensor inputs do not support the *requires_grad* argument, except for *torch.tensor* . This covers the following ops: 

> * [`torch.norm()`](generated/torch.norm.html#torch.norm "torch.norm")
> * [`torch.bartlett_window()`](generated/torch.bartlett_window.html#torch.bartlett_window "torch.bartlett_window")
> * [`torch.blackman_window()`](generated/torch.blackman_window.html#torch.blackman_window "torch.blackman_window")
> * [`torch.empty()`](generated/torch.empty.html#torch.empty "torch.empty")
> * [`torch.empty_like()`](generated/torch.empty_like.html#torch.empty_like "torch.empty_like")
> * [`torch.empty_strided()`](generated/torch.empty_strided.html#torch.empty_strided "torch.empty_strided")
> * [`torch.eye()`](generated/torch.eye.html#torch.eye "torch.eye")
> * [`torch.full()`](generated/torch.full.html#torch.full "torch.full")
> * [`torch.full_like()`](generated/torch.full_like.html#torch.full_like "torch.full_like")
> * [`torch.hamming_window()`](generated/torch.hamming_window.html#torch.hamming_window "torch.hamming_window")
> * [`torch.hann_window()`](generated/torch.hann_window.html#torch.hann_window "torch.hann_window")
> * [`torch.linspace()`](generated/torch.linspace.html#torch.linspace "torch.linspace")
> * [`torch.logspace()`](generated/torch.logspace.html#torch.logspace "torch.logspace")
> * [`torch.normal()`](generated/torch.normal.html#torch.normal "torch.normal")
> * [`torch.ones()`](generated/torch.ones.html#torch.ones "torch.ones")
> * [`torch.rand()`](generated/torch.rand.html#torch.rand "torch.rand")
> * [`torch.rand_like()`](generated/torch.rand_like.html#torch.rand_like "torch.rand_like")
> * [`torch.randint_like()`](generated/torch.randint_like.html#torch.randint_like "torch.randint_like")
> * [`torch.randn()`](generated/torch.randn.html#torch.randn "torch.randn")
> * [`torch.randn_like()`](generated/torch.randn_like.html#torch.randn_like "torch.randn_like")
> * [`torch.randperm()`](generated/torch.randperm.html#torch.randperm "torch.randperm")
> * [`torch.tril_indices()`](generated/torch.tril_indices.html#torch.tril_indices "torch.tril_indices")
> * [`torch.triu_indices()`](generated/torch.triu_indices.html#torch.triu_indices "torch.triu_indices")
> * [`torch.vander()`](generated/torch.vander.html#torch.vander "torch.vander")
> * [`torch.zeros()`](generated/torch.zeros.html#torch.zeros "torch.zeros")
> * [`torch.zeros_like()`](generated/torch.zeros_like.html#torch.zeros_like "torch.zeros_like")

The following functions require *dtype* , *layout* , *device* as parameters in TorchScript,
but these parameters are optional in Python. 

> * [`torch.randint()`](generated/torch.randint.html#torch.randint "torch.randint")
> * [`torch.sparse_coo_tensor()`](generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor "torch.sparse_coo_tensor")
> * [`to()`](generated/torch.Tensor.to.html#torch.Tensor.to "torch.Tensor.to")

PyTorch Unsupported Modules and Classes 
------------------------------------------------------------------------------------------------------------------

TorchScript cannot currently compile a number of other commonly used PyTorch
constructs. Below are listed the modules that TorchScript does not support, and
an incomplete list of PyTorch classes that are not supported. For unsupported modules
we suggest using [`torch.jit.trace()`](generated/torch.jit.trace.html#torch.jit.trace "torch.jit.trace")  . 

> * [`torch.nn.RNN`](generated/torch.nn.RNN.html#torch.nn.RNN "torch.nn.RNN")
> * [`torch.nn.AdaptiveLogSoftmaxWithLoss`](generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss "torch.nn.AdaptiveLogSoftmaxWithLoss")
> * [`torch.autograd.Function`](autograd.html#torch.autograd.Function "torch.autograd.Function")
> * `torch.autograd.enable_grad`

