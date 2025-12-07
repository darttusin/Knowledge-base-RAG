PackedSequence 
================================================================

*class* torch.nn.utils.rnn. PackedSequence ( *data*  , *batch_sizes = None*  , *sorted_indices = None*  , *unsorted_indices = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/rnn.py#L38) 
:   Holds the data and list of [`batch_sizes`](#torch.nn.utils.rnn.PackedSequence.batch_sizes "torch.nn.utils.rnn.PackedSequence.batch_sizes")  of a packed sequence. 

All RNN modules accept packed sequences as inputs. 

Note 

Instances of this class should never be created manually. They are meant
to be instantiated by functions like [`pack_padded_sequence()`](torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence "torch.nn.utils.rnn.pack_padded_sequence")  . 

Batch sizes represent the number elements at each sequence step in
the batch, not the varying sequence lengths passed to [`pack_padded_sequence()`](torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence "torch.nn.utils.rnn.pack_padded_sequence")  . For instance, given data `abc`  and `x`  the [`PackedSequence`](#torch.nn.utils.rnn.PackedSequence "torch.nn.utils.rnn.PackedSequence")  would contain data `axbc`  with `batch_sizes=[2,1,1]`  .

Variables
:   * **data** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Tensor containing packed sequence
* **batch_sizes** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Tensor of integers holding
information about the batch size at each sequence step
* **sorted_indices** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – Tensor of integers holding how this [`PackedSequence`](#torch.nn.utils.rnn.PackedSequence "torch.nn.utils.rnn.PackedSequence")  is constructed from sequences.
* **unsorted_indices** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – Tensor of integers holding how this
to recover the original sequences with correct order.

Return type
:   [*Self*](https://docs.python.org/3/library/typing.html#typing.Self "(in Python v3.13)")

Note 

[`data`](#torch.nn.utils.rnn.PackedSequence.data "torch.nn.utils.rnn.PackedSequence.data")  can be on arbitrary device and of arbitrary dtype. [`sorted_indices`](#torch.nn.utils.rnn.PackedSequence.sorted_indices "torch.nn.utils.rnn.PackedSequence.sorted_indices")  and [`unsorted_indices`](#torch.nn.utils.rnn.PackedSequence.unsorted_indices "torch.nn.utils.rnn.PackedSequence.unsorted_indices")  must be `torch.int64`  tensors on the same device as [`data`](#torch.nn.utils.rnn.PackedSequence.data "torch.nn.utils.rnn.PackedSequence.data")  . 

However, [`batch_sizes`](#torch.nn.utils.rnn.PackedSequence.batch_sizes "torch.nn.utils.rnn.PackedSequence.batch_sizes")  should always be a CPU `torch.int64`  tensor. 

This invariant is maintained throughout [`PackedSequence`](#torch.nn.utils.rnn.PackedSequence "torch.nn.utils.rnn.PackedSequence")  class,
and all functions that construct a [`PackedSequence`](#torch.nn.utils.rnn.PackedSequence "torch.nn.utils.rnn.PackedSequence")  in PyTorch
(i.e., they only pass in tensors conforming to this constraint).

batch_sizes *: [Tensor](../tensors.html#torch.Tensor "torch.Tensor")* 
:   Alias for field number 1

count ( *value*  , */* ) 
:   Return number of occurrences of value.

data *: [Tensor](../tensors.html#torch.Tensor "torch.Tensor")* 
:   Alias for field number 0

index ( *value*  , *start = 0*  , *stop = 9223372036854775807*  , */* ) 
:   Return first index of value. 

Raises ValueError if the value is not present.

*property* is_cuda *: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")* 
:   Return true if *self.data* stored on a gpu.

is_pinned ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/rnn.py#L206) 
:   Return true if *self.data* stored on in pinned memory. 

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

sorted_indices *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* 
:   Alias for field number 2

to ( *dtype : [torch.dtype](../tensor_attributes.html#torch.dtype "torch.dtype")*  , *non_blocking : [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = ...*  , *copy : [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = ...* ) → Self [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/rnn.py#L127) 
to ( *device : Optional [ Union [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") , [torch.device](../tensor_attributes.html#torch.device "torch.device") , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") ] ] = ...*  , *dtype : Optional [ [torch.dtype](../tensor_attributes.html#torch.dtype "torch.dtype") ] = ...*  , *non_blocking : [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = ...*  , *copy : [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = ...* ) → Self
to ( *other : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *non_blocking : [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = ...*  , *copy : [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = ...* ) → Self
:   Perform dtype and/or device conversion on *self.data* . 

It has similar signature as [`torch.Tensor.to()`](torch.Tensor.to.html#torch.Tensor.to "torch.Tensor.to")  , except optional
arguments like *non_blocking* and *copy* should be passed as kwargs,
not args, or they will not apply to the index tensors. 

Note 

If the `self.data`  Tensor already has the correct [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  and [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , then `self`  is returned.
Otherwise, returns a copy with the desired configuration.

unsorted_indices *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* 
:   Alias for field number 3

