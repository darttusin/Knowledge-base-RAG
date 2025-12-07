torch.distributed.tensor 
====================================================================================

Note 

`torch.distributed.tensor`  is currently in alpha state and under
development, we are committing backward compatibility for the most APIs listed
in the doc, but there might be API changes if necessary.

PyTorch DTensor (Distributed Tensor) 
----------------------------------------------------------------------------------------------------------

PyTorch DTensor offers simple and flexible tensor sharding primitives that transparently handles distributed
logic, including sharded storage, operator computation and collective communications across devices/hosts. `DTensor`  could be used to build different parallelism solutions and support sharded state_dict representation
when working with multi-dimensional sharding. 

Please see examples from the PyTorch native parallelism solutions that are built on top of `DTensor`  : 

* [Tensor Parallel](https://localhost:8000/docs/main/distributed.tensor.parallel.html)
* [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)

[`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  follows the SPMD (single program, multiple data) programming model to empower users to
write distributed program as if it’s a **single-device program with the same convergence property** . It
provides a uniform tensor sharding layout (DTensor Layout) through specifying the `DeviceMesh`  and `Placement`  : 

* `DeviceMesh`  represents the device topology and the communicators of the cluster using
an n-dimensional array.
* `Placement`  describes the sharding layout of the logical tensor on the `DeviceMesh`  .
DTensor supports three types of placements: `Shard`  , `Replicate`  and `Partial`  .

### DTensor Class APIs 

[`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  is a `torch.Tensor`  subclass. This means once a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  is created, it could be
used in very similar way to `torch.Tensor`  , including running different types of PyTorch operators as if
running them in a single device, allowing proper distributed computation for PyTorch operators. 

In addition to existing `torch.Tensor`  methods, it also offers a set of additional methods to interact with `torch.Tensor`  , `redistribute`  the DTensor Layout to a new DTensor, get the full tensor content
on all devices, etc. 

*class* torch.distributed.tensor. DTensor ( *local_tensor*  , *spec*  , *** , *requires_grad* ) 
:   `DTensor`  (Distributed Tensor) is a subclass of `torch.Tensor`  that provides single-device like
abstraction to program with multi-device `torch.Tensor`  . It describes the distributed tensor sharding
layout (DTensor Layout) through the `DeviceMesh`  and following types of `Placement`  : 

* `Shard`  : Tensor sharded on the tensor dimension `dim`  on the devices of the `DeviceMesh`  dimension
* `Replicate`  : Tensor replicated on the devices of the `DeviceMesh`  dimension
* `Partial`  : Tensor is pending reduction on the devices of the `DeviceMesh`  dimension

When calling PyTorch operators, `DTensor`  overrides the PyTorch operators to perform sharded computation and issue
communications whenever necessary. Along with the operator computation, `DTensor`  will transform or propagate the
placements (DTensor Layout) properly (based on the operator semantic itself) and generate new `DTensor`  outputs. 

To ensure numerical correctness of the `DTensor`  sharded computation when calling PyTorch operators, `DTensor`  requires every Tensor argument of the operator be DTensor. 

Note 

Directly using the Tensor subclass constructor here is not the recommended way to create a `DTensor`  (i.e. it does not handle autograd correctly hence is not the public API). Please refer to the [create_dtensor](#create-dtensor)  section to see how to create a `DTensor`  .

Return type
:   [DTensor](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

__create_chunk_list__ ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L622) 
:   Return a list of ChunkStorageMetadata, which is a dataclass that describes the size/offset of the local shard/replica
on current rank. For DTensor, each rank will have a single local shard/replica, so the returned list usually only
has one element. 

This dunder method is primariy used for distributed checkpoint purpose. 

Returns
:   A List[ `ChunkStorageMetadata`  ] object that represents the shard size/offset on the current rank.

*static* from_local ( *local_tensor*  , *device_mesh = None*  , *placements = None*  , *** , *run_check = False*  , *shape = None*  , *stride = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L356) 
:   Create a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  from a local torch.Tensor on each rank
according to the `device_mesh`  and `placements`  specified. 

Parameters
:   * **local_tensor** ( [*torch.Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – local torch.Tensor on each rank.
* **device_mesh** ( `DeviceMesh`  , optional) – DeviceMesh to place the
tensor, if not specified, must be called under a DeviceMesh
context manager, default: None
* **placements** (List[ `Placement`  ], optional) – the placements that
describes how to place the local torch.Tensor on DeviceMesh, must
have the same number of elements as `device_mesh.ndim`  .

Keyword Arguments
:   * **run_check** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – at a cost of extra communications, perform
sanity check across ranks to check each local tensor’s meta information
to ensure correctness. If have `Replicate`  in `placements`  , the
data on first rank of the device mesh dimension will be broadcasted
to other ranks. default: False
* **shape** ( [*torch.Size*](size.html#torch.Size "torch.Size") *,* *optional*  ) – A List of int which specifies the size of
DTensor which build on top of *local_tensor* . Note this needs to be
provided if the shape of `local_tensor`  are different across the ranks.
If not provided, `shape`  will be computed assuming the given distributed
tensor is evenly sharded across ranks. default: None
* **stride** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – A List of int which specifies the stride of DTensor.
If not provided, `stride`  will be computed assuming the given distributed
tensor is evenly sharded across ranks. default: None

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  object

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

Note 

When `run_check=False`  , it is the user’s responsibility to ensure the
local tensor passed in is correct across ranks (i.e. the tensor is sharded for
the `Shard(dim)`  placement or replicated for the `Replicate()`  placement).
If not, the behavior of the created DTensor is undefined.

Note 

`from_local`  is differentiable, the *requires_grad* of the created *DTensor* object will depend on if *local_tensor* requires_grad or not.

full_tensor ( *** , *grad_placements = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L560) 
:   Return the full tensor of this DTensor. It will perform necessary collectives
to gather the local tensors from other ranks in its DeviceMesh and concatenate
them together. It’s a syntatic sugar of the following code: 

`dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()` 

Keyword Arguments
: **grad_placements** (List[ `Placement`  ], optional) – the placements describes
the future layout of any gradient layout of the full Tensor returned from this
function. *full_tensor* converts DTensor to a full torch.Tensor and the returned torch.tensor
might not be used as the original replicated DTensor layout later in the code. This
argument is the hint that user can give to autograd in case the gradient
layout of the returned tensor does not match the original replicated DTensor layout.
If not specified, we will assume the gradient layout of the full tensor be replicated.

Returns
:   A [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")  object that represents the full tensor of this DTensor.

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Note 

`full_tensor`  is differentiable.

redistribute ( *device_mesh = None*  , *placements = None*  , *** , *async_op = False*  , *forward_dtype = None*  , *backward_dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L478) 
:   `redistribute`  performs necessary collective operations that redistribute the current
DTensor from its current placements to a new placements, or from its current DeviceMesh
to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
specifying a Replicate placement for each dimension of the DeviceMesh. 

When redistributing from current to the new placements on one device mesh dimension, we
will perform the following operations including communication collective or local operation: 

1. `Shard(dim)`  -> `Replicate()`  : `all_gather`
2. `Shard(src_dim)`  -> `Shard(dst_dim)`  : `all_to_all`
3. `Replicate()`  -> `Shard(dim)`  : local chunking (i.e. `torch.chunk`  )
4. `Partial()`  -> `Replicate()`  : `all_reduce`
5. `Partial()`  -> `Shard(dim)`  : `reduce_scatter`

`redistribute`  would correctly figure out the necessary redistribute steps for DTensors
that are created either on 1-D or N-D DeviceMesh. 

Parameters
:   * **device_mesh** ( `DeviceMesh`  , optional) – DeviceMesh to place the
DTensor. If not specified, it would use the current DTensor’s DeviceMesh.
default: None
* **placements** (List[ `Placement`  ], optional) – the new placements that
describes how to place the DTensor into the DeviceMesh, must
have the same number of elements as `device_mesh.ndim`  .
default: replicate on all mesh dimensions

Keyword Arguments
:   * **async_op** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to perform the DTensor redistribute operation
asynchronously or not. Default: False
* **forward_dtype** ( [*torch.dtype*](tensor_attributes.html#torch.dtype "torch.dtype") *,* *optional*  ) – the local tensor datatype can be converted to `forward_dtype`  before redistributing the local tensor in its forward.
The result DTensor will be in `forward_dtype`  Default: None.
* **backward_dtype** ( [*torch.dtype*](tensor_attributes.html#torch.dtype "torch.dtype") *,* *optional*  ) – the local tensor datatype can be converted to `backward_dtype`  before redistributing the local tensor in its backward.
The result DTensor gradient would be converted back to the current DTensor dtype. Default: None

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  object

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

Note 

`redistribute`  is differentiable, which means user do not need to worry about
the backward formula of the redistribute operation.

Note 

`redistribute`  currently only supports redistributing DTensor on the same DeviceMesh,
Please file an issue if you need to redistribute DTensor to different DeviceMesh.

to_local ( *** , *grad_placements = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L441) 
:   Get the local tensor of this DTensor on its current rank. For sharding it returns
a local shard of the logical tensor view, for replication it returns the replica on
its current rank. 

Keyword Arguments
: **grad_placements** (List[ `Placement`  ], optional) – the placements describes
the future layout of any gradient layout of the Tensor returned from this
function. *to_local* converts DTensor to local tensor and the returned local tensor
might not be used as the original DTensor layout later in the code. This
argument is the hint that user can give to autograd in case the gradient
layout of the returned tensor does not match the original DTensor layout.
If not specified, we will assume the gradient layout remains the same
as the original DTensor and use that for gradient computation.

Returns
:   A [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")  or `AsyncCollectiveTensor`  object. it represents the
local tensor on its current rank. When an `AsyncCollectiveTensor`  object is returned,
it means the local tensor is not ready yet (i.e. communication is not finished). In this
case, user needs to call `wait`  to wait the local tensor to be ready.

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

Note 

`to_local`  is differentiable, the `requires_grad`  of the local tensor returned
will depend on if the *DTensor* requires_grad or not.

*property* device_mesh *: [DeviceMesh](distributed.html#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh")* 
:   The `DeviceMesh`  attribute that associates with this DTensor object. 

Note 

`device_mesh`  is a read-only property, it can not be set.

*property* placements *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") [ [torch.distributed.tensor.placement_types.Placement](#torch.distributed.tensor.placement_types.Placement "torch.distributed.tensor.placement_types.Placement") , ... ]* 
:   The placements attribute of this DTensor that describes the layout of this
DTensor on the its DeviceMesh. 

Note 

`placements`  is a read-only property, it can not be set.

### DeviceMesh as the distributed communicator 

[`DeviceMesh`](distributed.html#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh")  was built from DTensor as the abstraction to describe cluster’s device topology and represent
multi-dimensional communicators (on top of `ProcessGroup`  ). To see the details of how to create/use a DeviceMesh,
please refer to the [DeviceMesh recipe](https://localhost:8000/tutorials/recipes/distributed_device_mesh.html)  .

### DTensor Placement Types 

DTensor supports the following types of [`Placement`](#torch.distributed.tensor.placement_types.Placement "torch.distributed.tensor.placement_types.Placement")  on each `DeviceMesh`  dimension: 

*class* torch.distributed.tensor.placement_types. Shard ( *dim* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/placement_types.py#L48) 
:   The `Shard(dim)`  placement describes the DTensor sharding on tensor dimension `dim`  over a corresponding `DeviceMesh`  dimension, where each rank on the
DeviceMesh dimension only holds a shard/piece of the global Tensor. The `Shard(dim)`  placement follows the `torch.chunk(dim)`  semantic, where the
last few shards on the DeviceMesh dimension might be empty when the tensor dimension
is not evenly divisible on the DeviceMesh dimension. The `Shard`  placement can be
used by all DTensor APIs (i.e. distribute_tensor, from_local, etc.) 

Parameters
: **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The tensor dimension that describes the DTensor is sharded over its
corresponding DeviceMesh dimension.

Warning 

sharding on a tensor dimension where the tensor dimension size is not
evenly divisible on a DeviceMesh dimension is currently experimental and subject to change.

dim *: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")* 
:

*class* torch.distributed.tensor.placement_types. Replicate [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/placement_types.py#L600) 
:   The `Replicate()`  placement describes the DTensor replicating on a corresponding `DeviceMesh`  dimension, where each rank on the DeviceMesh dimension holds a
replica of the global Tensor. The `Replicate`  placement can be used by all
DTensor APIs (i.e. `distribute_tensor`  , `DTensor.from_local`  , etc.)

*class* torch.distributed.tensor.placement_types. Partial ( *reduce_op = 'sum'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/placement_types.py#L652) 
:   The `Partial(reduce_op)`  placement describes the DTensor that is pending
reduction on a specified `DeviceMesh`  dimension, where each rank on the
DeviceMesh dimension holds the partial value of the global Tensor. User can
redistribute the `Partial`  DTensor to a `Replicate`  or `Shard(dim)`  placement on the specified `DeviceMesh`  dimension using `redistribute`  ,
which would trigger necessary communication operations under the hood (i.e. `allreduce`  , `reduce_scatter`  ). 

Parameters
: **reduce_op** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – The reduction op to be used for the partial DTensor
to produce Replicated/Sharded DTensor. Only element-wise reduction operations
are supported, including: “sum”, “avg”, “product”, “max”, “min”, default: “sum”.

Note 

The `Partial`  placement can be generated as a result of the DTensor operators,
and can only be used by the `DTensor.from_local`  API.

reduce_op *: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")* *= 'sum'* 
:

*class* torch.distributed.tensor.placement_types. Placement [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/placement_types.py#L23) 
:   The base class for the Placement type, where it describes how a DTensor is placed onto the `DeviceMesh`  . `Placement`  and `DeviceMesh`  together could describe the DTensor Layout.
It is the base class of the three main DTensor Placement types: `Shard`  , `Replicate`  ,
and `Partial`  . 

This class is not meant to be used directly, mainly served as a typing stub. 

is_partial ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/placement_types.py#L44) 
:   Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

is_replicate ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/placement_types.py#L41) 
:   Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

is_shard ( *dim = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/placement_types.py#L34) 
:   Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

Different ways to create a DTensor 
--------------------------------------------------------------------------------------------------------

There’re three ways to construct a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  :
:   * [`distribute_tensor()`](#torch.distributed.tensor.distribute_tensor "torch.distributed.tensor.distribute_tensor")  creates a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  from a logical or “global” `torch.Tensor`  on
each rank. This could be used to shard the leaf `torch.Tensor`  s (i.e. model parameters/buffers
and inputs).
* [`DTensor.from_local()`](#torch.distributed.tensor.DTensor.from_local "torch.distributed.tensor.DTensor.from_local")  creates a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  from a local `torch.Tensor`  on each rank, which can
be used to create [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  from a non-leaf `torch.Tensor`  s (i.e. intermediate activation
tensors during forward/backward).
* DTensor provides dedicated tensor factory functions (e.g. [`empty()`](#torch.distributed.tensor.empty "torch.distributed.tensor.empty")  , [`ones()`](#torch.distributed.tensor.ones "torch.distributed.tensor.ones")  , [`randn()`](#torch.distributed.tensor.randn "torch.distributed.tensor.randn")  , etc.)
to allow different [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  creations by directly specifying the `DeviceMesh`  and `Placement`  . Compare to [`distribute_tensor()`](#torch.distributed.tensor.distribute_tensor "torch.distributed.tensor.distribute_tensor")  , this could directly materializing the sharded memory
on device, instead of performing sharding after initializing the logical Tensor memory.

### Create DTensor from a logical torch.Tensor 

The SPMD (single program, multiple data) programming model in `torch.distributed`  launches multiple processes
(i.e. via `torchrun`  ) to execute the same program, this means that the model inside the program would be
initialized on different processes first (i.e. the model might be initialized on CPU, or meta device, or directly
on GPU if enough memory). 

`DTensor`  offers a [`distribute_tensor()`](#torch.distributed.tensor.distribute_tensor "torch.distributed.tensor.distribute_tensor")  API that could shard the model weights or Tensors to `DTensor`  s,
where it would create a DTensor from the “logical” Tensor on each process. This would empower the created `DTensor`  s to comply with the single device semantic, which is critical for **numerical correctness** . 

torch.distributed.tensor. distribute_tensor ( *tensor*  , *device_mesh = None*  , *placements = None*  , *** , *src_data_rank = 0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L653) 
:   Distribute a leaf `torch.Tensor`  (i.e. nn.Parameter/buffers) to the `device_mesh`  according
to the `placements`  specified. The rank of `device_mesh`  and `placements`  must be the
same. The `tensor`  to distribute is the logical or “global” tensor, and the API would use
the `tensor`  from first rank of the DeviceMesh dimension as the source of truth to preserve
the single-device semantic. If you want to construct a DTensor in the middle of the Autograd
computation, please use [`DTensor.from_local()`](#torch.distributed.tensor.DTensor.from_local "torch.distributed.tensor.DTensor.from_local")  instead. 

Parameters
:   * **tensor** ( [*torch.Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – torch.Tensor to be distributed. Note that if you
want to shard a tensor on a dimension that is not evenly divisible by
the number of devices in that mesh dimension, we use `torch.chunk`  semantic to shard the tensor and scatter the shards. The uneven sharding
behavior is experimental and subject to change.
* **device_mesh** ( `DeviceMesh`  , optional) – DeviceMesh to distribute the
tensor, if not specified, must be called under a DeviceMesh context
manager, default: None
* **placements** (List[ `Placement`  ], optional) – the placements that
describes how to place the tensor on DeviceMesh, must have the same
number of elements as `device_mesh.ndim`  . If not specified, we will
by default replicate the tensor across the `device_mesh`  from the
first rank of each dimension of the *device_mesh* .

Keyword Arguments
: **src_data_rank** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the rank of the source data for the logical/global tensor, it is
used by [`distribute_tensor()`](#torch.distributed.tensor.distribute_tensor "torch.distributed.tensor.distribute_tensor")  to scatter/broadcast the shards/replicas to other ranks.
By default, we use `group_rank=0`  on each DeviceMesh dimension as the source data to preserve
the single-device semantic. If passing `None`  explicitly, [`distribute_tensor()`](#torch.distributed.tensor.distribute_tensor "torch.distributed.tensor.distribute_tensor")  simply uses
its local data instead of trying to preserve the single-device semantic via scatter/broadcast.
Default: 0

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  or `XLAShardedTensor`  object.

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

Note 

When initialize the DeviceMesh with the `xla`  device_type, `distribute_tensor`  return *XLAShardedTensor* instead. see [this issue](https://github.com/pytorch/pytorch/issues/92909)  for more details. The XLA integration is experimental and subject to change.

Along with [`distribute_tensor()`](#torch.distributed.tensor.distribute_tensor "torch.distributed.tensor.distribute_tensor")  , DTensor also offers a [`distribute_module()`](#torch.distributed.tensor.distribute_module "torch.distributed.tensor.distribute_module")  API to allow easier
sharding on the `nn.Module`  level 

torch.distributed.tensor. distribute_module ( *module*  , *device_mesh = None*  , *partition_fn = None*  , *input_fn = None*  , *output_fn = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L837) 
:   This function expose three functions to control the parameters/inputs/outputs of the module: 

1. To perform sharding on the module before runtime execution by specifying the `partition_fn`  (i.e. allow user to convert Module parameters to [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  parameters according to the *partition_fn* specified).
2. To control the inputs or outputs of the module during runtime execution by
specifying the `input_fn`  and `output_fn`  . (i.e. convert the input to [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  , convert the output back to `torch.Tensor`  ) 

Parameters
:   * **module** ( `nn.Module`  ) – user module to be partitioned.
* **device_mesh** ( `DeviceMesh`  ) – the device mesh to place the module.
* **partition_fn** ( *Callable*  ) – the function to partition parameters (i.e. shard certain
parameters across the `device_mesh`  ). If `partition_fn`  is not specified,
by default we replicate all module parameters of `module`  across the mesh.
* **input_fn** ( *Callable*  ) – specify the input distribution, i.e. could control how the
input of the module is sharded. `input_fn`  will be installed as a module `forward_pre_hook`  (pre forward hook).
* **output_fn** ( *Callable*  ) – specify the output distribution, i.e. could control how the
output is sharded, or convert it back to torch.Tensor. `output_fn`  will be
installed as a module `forward_hook`  (post forward hook).

Returns
:   A module that contains parameters/buffers that are all `DTensor`  s.

Return type
:   [*Module*](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")

Note 

When initialize the DeviceMesh with the `xla`  device_type, `distribute_module`  return nn.Module with PyTorch/XLA SPMD annotated parameters. See [this issue](https://github.com/pytorch/pytorch/issues/92909)  for more details. The XLA integration is experimental and subject to change.

### DTensor Factory Functions 

DTensor also provides dedicated tensor factory functions to allow creating [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  directly
using torch.Tensor like factory function APIs (i.e. torch.ones, torch.empty, etc), by additionally
specifying the `DeviceMesh`  and `Placement`  for the [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  created: 

torch.distributed.tensor. zeros ( ** size*  , *requires_grad = False*  , *dtype = None*  , *layout = torch.strided*  , *device_mesh = None*  , *placements = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L1277) 
:   Returns a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  filled with the scalar value 0. 

Parameters
: **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – a sequence of integers defining the shape of the output [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Can be a variable number of arguments or a collection like a list or tuple.
E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..))

Keyword Arguments
:   * **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  . Default: `False`  .
* **dtype** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](generated/torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Default: `torch.strided`  .
* **device_mesh** – `DeviceMesh`  type, contains the mesh info of ranks
* **placements** – a sequence of `Placement`  type: `Shard`  , `Replicate`

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  object on each rank

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

torch.distributed.tensor. ones ( ** size*  , *dtype = None*  , *layout = torch.strided*  , *requires_grad = False*  , *device_mesh = None*  , *placements = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L1056) 
:   Returns a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  filled with the scalar value 1, with the shape defined
by the variable argument `size`  . 

Parameters
: **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – a sequence of integers defining the shape of the output [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Can be a variable number of arguments or a collection like a list or tuple.
E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](generated/torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned DTensor.
Default: `torch.strided`  .
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  . Default: `False`  .
* **device_mesh** – `DeviceMesh`  type, contains the mesh info of ranks
* **placements** – a sequence of `Placement`  type: `Shard`  , `Replicate`

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  object on each rank

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

torch.distributed.tensor. empty ( ** size*  , *dtype = None*  , *layout = torch.strided*  , *requires_grad = False*  , *device_mesh = None*  , *placements = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L1099) 
:   Returns a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  filled with uninitialized data. The shape of the [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  is defined by the variable argument `size`  . 

Parameters
: **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – a sequence of integers defining the shape of the output [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Can be a variable number of arguments or a collection like a list or tuple.
E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..))

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](generated/torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ). layout ( [`torch.layout`](tensor_attributes.html#torch.layout "torch.layout")  , optional): the desired layout of returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Default: `torch.strided`  .
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  . Default: `False`  .
* **device_mesh** – `DeviceMesh`  type, contains the mesh info of ranks
* **placements** – a sequence of `Placement`  type: `Shard`  , `Replicate`

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  object on each rank

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

torch.distributed.tensor. full ( *size*  , *fill_value*  , *** , *dtype = None*  , *layout = torch.strided*  , *requires_grad = False*  , *device_mesh = None*  , *placements = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L1142) 
:   Returns a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  filled with `fill_value`  according to `device_mesh`  and `placements`  , with the shape defined by the argument `size`  . 

Parameters
:   * **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – a sequence of integers defining the shape of the output [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Can be a variable number of arguments or a collection like a list or tuple.
E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))
* **fill_value** ( *Scalar*  ) – the value to fill the output tensor with.

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](generated/torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned DTensor.
Default: `torch.strided`  .
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  . Default: `False`  .
* **device_mesh** – `DeviceMesh`  type, contains the mesh info of ranks.
* **placements** – a sequence of `Placement`  type: `Shard`  , `Replicate`

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  object on each rank

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

torch.distributed.tensor. rand ( ** size*  , *requires_grad = False*  , *dtype = None*  , *layout = torch.strided*  , *device_mesh = None*  , *placements = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L1189) 
:   Returns a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  filled with random numbers from a uniform distribution
on the interval `[0, 1)`  . The shape of the tensor is defined by the variable
argument `size`  . 

Parameters
: **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – a sequence of integers defining the shape of the output [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Can be a variable number of arguments or a collection like a list or tuple.
E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](generated/torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned DTensor.
Default: `torch.strided`  .
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  . Default: `False`  .
* **device_mesh** – `DeviceMesh`  type, contains the mesh info of ranks.
* **placements** – a sequence of `Placement`  type: `Shard`  , `Replicate`

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  object on each rank

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

torch.distributed.tensor. randn ( ** size*  , *requires_grad = False*  , *dtype = None*  , *layout = torch.strided*  , *device_mesh = None*  , *placements = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/_api.py#L1233) 
:   Returns a [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  filled with random numbers from a normal distribution
with mean 0 and variance 1. The shape of the tensor is defined by the variable
argument `size`  . 

Parameters
: **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – a sequence of integers defining the shape of the output [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Can be a variable number of arguments or a collection like a list or tuple.
E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  .
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](generated/torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned DTensor.
Default: `torch.strided`  .
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  . Default: `False`  .
* **device_mesh** – `DeviceMesh`  type, contains the mesh info of ranks.
* **placements** – a sequence of `Placement`  type: `Shard`  , `Replicate`

Returns
:   A [`DTensor`](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")  object on each rank

Return type
:   [*DTensor*](#torch.distributed.tensor.DTensor "torch.distributed.tensor.DTensor")

Debugging 
----------------------------------------------------------------------------------

### Logging 

When launching the program, you can turn on additional logging using the `TORCH_LOGS`  environment variable from [torch._logging](https://localhost:8000/docs/main/logging.html#module-torch._logging)  : 

* `TORCH_LOGS=+dtensor`  will display `logging.DEBUG`  messages and all levels above it.
* `TORCH_LOGS=dtensor`  will display `logging.INFO`  messages and above.
* `TORCH_LOGS=-dtensor`  will display `logging.WARNING`  messages and above.

### Debugging Tools 

To debug the program that applied DTensor, and understand more details about what collectives happened under the
hood, DTensor provides a [`CommDebugMode`](#torch.distributed.tensor.debug.CommDebugMode "torch.distributed.tensor.debug.CommDebugMode")  : 

*class* torch.distributed.tensor.debug. CommDebugMode 
:   [`CommDebugMode`](#torch.distributed.tensor.debug.CommDebugMode "torch.distributed.tensor.debug.CommDebugMode")  is a context manager that counts the number of
functional collectives within its context. It does this using a `TorchDispatchMode`  . 

Note 

Not all collectives are supported yet.

Example usage 

```
mod = ...
comm_mode = CommDebugMode()
with comm_mode:
    mod.sum().backward()
print(comm_mode.get_comm_counts())

```

generate_comm_debug_tracing_table ( *noise_level = 3* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/debug/_comm_mode.py#L402) 
:   Generates detailed table displaying operations and collective tracing information
on a module level. Amount of information is dependent on noise_level 

0. prints module-level collective counts
1. prints dTensor operations not included in trivial operations, module information
2. prints operations not included in trivial operations
3. prints all operations

generate_json_dump ( *file_name = 'comm_mode_log.json'*  , *noise_level = 3* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/debug/_comm_mode.py#L254) 
:   Creates json file used to build browser visual
0. prints module-level collective counts
1. prints dTensor operations not included in trivial operations
2. prints operations not included in trivial operations
3. prints all operations

get_comm_counts ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/debug/_comm_mode.py#L570) 
:   Returns the communication counts as a dictionary. 

Returns
:   The communication counts as a dictionary.

Return type
:   Dict[Any, [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ]

get_parameter_info ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/debug/_comm_mode.py#L578) 
:   Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]]

get_sharding_info ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/debug/_comm_mode.py#L581) 
:   Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]]

get_total_counts ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/debug/_comm_mode.py#L567) 
:   Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

log_comm_debug_tracing_table_to_file ( *file_name = 'comm_mode_log.txt'*  , *noise_level = 3* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/debug/_comm_mode.py#L601) 
:   Alternative to console CommDebugMode output, writes to file specified by the user

To visualize the sharding of a DTensor that have less than 3 dimensions, DTensor provides [`visualize_sharding()`](#torch.distributed.tensor.debug.visualize_sharding "torch.distributed.tensor.debug.visualize_sharding")  : 

torch.distributed.tensor.debug. visualize_sharding ( *dtensor*  , *header = ''*  , *use_rich = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/debug/_visualize_sharding.py#L155) 
:   Visualizes sharding in the terminal for `DTensor`  that are 1D or 2D. 

Note 

This requires the `tabulate`  package, or `rich`  and `matplotlib`  .
No sharding info will be printed for empty tensors

Experimental Features 
------------------------------------------------------------------------------

`DTensor`  also provides a set of experimental features. These features are either in prototyping stage, or the basic
functionality is done and but looking for user feedbacks. Please submit a issue to PyTorch if you have feedbacks to
these features. 

torch.distributed.tensor.experimental. context_parallel ( *mesh*  , *** , *buffers = None*  , *buffer_seq_dims = None*  , *no_restore_buffers = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/experimental/_attention.py#L1340) 
:   `context_parallel`  is an experimental API to enable context
parallelism (CP). This API performs two actions: 1) patch the SDPA
( `torch.nn.functional.scaled_dot_product_attention`  ) with the CP-enabled
one, 2) shard `buffers`  along the sequence dimension and each rank will
preserve the corresponding shard according `mesh`  . 

Parameters
:   * **mesh** ( `DeviceMesh`  ) – the device mesh for the context parallelism.
* **buffers** ( *Optional* *[* *List* *[* [*torch.Tensor*](tensors.html#torch.Tensor "torch.Tensor") *]* *]*  ) – buffers that the usage depend
on the sequence dimension. Examples are input batch, labels and
positional embedding buffers. These buffers must be sharded along
the sequence dimension to ensure the accuracy. The sharding will
happen in-place, the buffer’s shape will change within the context.
The buffers will be restored after the context finishes. `no_restore_buffers`  can be used to specify which buffers don’t
need to be restored. Note that `buffers`  should not contain any
nn.Parameter.
* **buffer_seq_dims** ( *Optional* *[* *List* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the sequence dimensions of `buffers`  .
* **no_restore_buffers** ( *Optional* *[* *Set* *[* [*torch.Tensor*](tensors.html#torch.Tensor "torch.Tensor") *]* *]*  ) – buffers in these set
won’t be restored after the context exits. This set must be a subset
of `buffers`  . If the buffers won’t be used after the context exits,
these buffers can be put in this list to avoid extra restore time.

Return type
:   [*Generator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator "(in Python v3.13)")  [None, None, None]

Warning 

*torch.distributed.tensor.experimental.context_parallel* is a
prototype feature in PyTorch. The API is subject to change.

torch.distributed.tensor.experimental. local_map ( *func*  , *out_placements*  , *in_placements = None*  , *in_grad_placements = None*  , *device_mesh = None*  , *** , *redistribute_inputs = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/experimental/_func_map.py#L26) 
:   [`local_map()`](#torch.distributed.tensor.experimental.local_map "torch.distributed.tensor.experimental.local_map")  is an experimental API that allows users to pass `DTensor`  s
to a function that is written to be applied on `torch.Tensor`  s. It is done by extracting
the local components of `DTensor`  , call the function, and wrap the outputs to `DTensor`  according to the `out_placements`  . 

Parameters
:   * **func** ( *Callable*  ) – the function to be applied on each local shard of `DTensor`  s.
* **out_placements** (Union[ *PlacementType* , Tuple[ *PlacementType* , …]]) – the desired placements of the `DTensor`  s in `func`  ’s flattened output.
If the flattened `output`  is a single value, the `out_placements`  should be
of type *PlacementType* . Otherwise if the flattened `output`  has multiple
values, the `out_placements`  should be a tuple of *PlacementType* values 1:1
mapping to the flattened `output`  .
Besides, for `Tensor`  output, we use *PlacementType* as its
placements (a *Tuple[Placement]* value). For non-Tensor output, the *PlacementType* should be *None* .
Note that the only exception is when no `DTensor`  argument is passed
in. In this case, even if *out_placements* is not *None* , the result function
should ignore the desired placements because the function is not running with `DTensor`  s.
* **in_placements** (Tuple[ *PlacementType* , …], optional) – the required placements of the `DTensor`  s in the flattened inputs of `func`  .
If `in_placements`  is specified, [`local_map()`](#torch.distributed.tensor.experimental.local_map "torch.distributed.tensor.experimental.local_map")  would examine whether the
placements of each `DTensor`  argument is the same as the required
placements or not. If the placements are not the same and `redistribute_inputs`  is `False`  , an exception will be raised. Otherwise if `redistribute_inputs`  is `True`  , the argument will be first redistributed to
the required sharding placements before passing its local tensor to `func`  .
The only exception is when required placements are not `None`  and the
argument is a [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")  . In this case, the placements examination
will be skipped and the argument will be directly passed to `func`  .
If `in_placements`  is `None`  , no placements examination will be performed.
Default: None
* **in_grad_placements** (Tuple[ *PlacementType* , …], optional) – the placements hint of the `DTensor`  s gradient corresponds
to the flattened input DTensor. This argument is the hint that user
can give to `to_local()`  in case the gradient layout of the
local tensor input does not match its `DTensor`  input layout.
If not specified, we will assume the gradient layout of the local
tensor input remains the same as the original `DTensor`  input
and use that for gradient computation. Default: None.
* **device_mesh** ( `DeviceMesh`  , optional) – the device mesh that all the `DTensor`  s are placed on. If not
specified, this will be inferred from the input `DTensor`  s’ device
mesh. *local_map* requires every `DTensor`  s to be placed on the same
device mesh. Default: None.
* **redistribute_inputs** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – the bool value indicating whether to reshard the input `DTensor`  s when
their placements are different from the required input placements. If this
value is `False`  and some `DTensor`  input has a different placement,
an exception will be raised. Default: False.

Returns
:   A `Callable`  that applies `func`  to each local shard of the input `DTensor`  and returns a `DTensor`  constructed from the return value of `func`  .

Raises
:   * [**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError "(in Python v3.13)")  – If the input `DTensor`  is not placed on the same device
 mesh, or if they are placed on a different device mesh than the `device_mesh`  argument passed in.
* [**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError "(in Python v3.13)")  – For any non-DTensor output, we require its corresponding
 output placement in `out_placements`  be None. An AssertionError will be raised
 if this is not the case.
* [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.13)")  – If `redistribute_inputs=False`  but the input `DTensor`  needs
 a redistribution according to `in_placements`  .

Example 

```
>>> def mm_allreduce_forward(device_mesh, W, X):
>>>     partial_sum_tensor = torch.mm(W, X)
>>>     reduced_tensor = funcol.all_reduce(partial_sum_tensor, "sum", device_mesh)
>>>     return reduced_tensor
>>>
>>> W = torch.randn(12, 8, requires_grad=False)
>>> X = torch.randn(8, 16, requires_grad=False)
>>> Y = torch.mm(W, X)
>>> row_wise = [Shard(0)]  # row-wise sharding placements on 1-d mesh
>>> col_wise = [Shard(1)]  # col-wise sharding placements on 1-d mesh
>>>
>>> # local_mm_allreduce_forward is the function wrapped with DTensor/Tensor convertion
>>> local_mm_allreduce_forward = local_map(
>>>     mm_allreduce_forward,
>>>     out_placements=[Replicate()],
>>>     in_placements=[col_wise, row_wise],
>>>     device_mesh=device_mesh,
>>> )
>>>
>>> W_dt = distribute_tensor(
...     W, device_mesh, (col_wise)
... )  # col-wisely sharded W tensor
>>> X_dt = distribute_tensor(
...     X, device_mesh, (row_wise)
... )  # row-wisely sharded X tensor
>>> Y_dt = local_mm_allreduce_forward(
...     device_mesh, W_dt, X_dt
... )  # apply local_mm_allreduce_forward to DTensors

```

Note 

This API is currently experimental and subject to change

torch.distributed.tensor.experimental. register_sharding ( *op* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/experimental/_register_sharding.py#L24) 
:   [`register_sharding()`](#torch.distributed.tensor.experimental.register_sharding "torch.distributed.tensor.experimental.register_sharding")  is an experimental API that allows users to register sharding
strategies for an operator when the tensor inputs and outputs are DTensor.
It can be useful when: (1) there doesn’t exist a default sharding strategy for `op`  ,
e.g. when `op`  is a custom operator that is not supported by `DTensor`  ; (2)
when users would like to overwrite default sharding strategies of existing operators. 

Parameters
: **op** ( *Union* *[* *OpOverload* *,* *List* *[* *OpOverload* *]* *]*  ) – An op or a list of ops to register the customized sharding function.

Returns
:   A function decorator which can be used to wrap a function that defines the sharding
strategy for the operator specified in `op`  . The defined sharding strategy will be
registered to DTensor and will override the default sharding strategy if DTensor has
already implemented the operator. The customized sharding function takes the same inputs
as the original op (except that if an arg is a [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")  , it will be
replaced by a tensor-like object that DTensor uses internally). The function should
return a sequence of 2-tuples, each specifying acceptable output placements and its
corresponding intput placements.

Example 

```
>>> @register_sharding(aten._softmax.default)
>>> def custom_softmax_sharding(x, dim, half_to_float):
>>>     softmax_dim = dim if dim >= 0 else dim + x.ndim
>>>     acceptable_shardings = []
>>>
>>>     all_replicate = ([Replicate()], [Replicate(), None, None])
>>>     acceptable_shardings.append(all_replicate)
>>>
>>>     for sharding_dim in range(x.ndim):
>>>         if sharding_dim != softmax_dim:
>>>             all_sharded = (
>>>                 [Shard(sharding_dim)],
>>>                 [Shard(sharding_dim), None, None],
>>>             )
>>>             acceptable_shardings.append(all_sharded)
>>>
>>>     return acceptable_shardings

```

Note 

This API is currently experimental and subject to change

