torch.nn.utils.rnn.unpack_sequence 
=========================================================================================================

torch.nn.utils.rnn. unpack_sequence ( *packed_sequences* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/rnn.py#L570) 
:   Unpack PackedSequence into a list of variable length Tensors. 

`packed_sequences`  should be a PackedSequence object. 

Example 

```
>>> from torch.nn.utils.rnn import pack_sequence, unpack_sequence
>>> a = torch.tensor([1, 2, 3])
>>> b = torch.tensor([4, 5])
>>> c = torch.tensor([6])
>>> sequences = [a, b, c]
>>> print(sequences)
[tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]
>>> packed_sequences = pack_sequence(sequences)
>>> print(packed_sequences)
PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)
>>> unpacked_sequences = unpack_sequence(packed_sequences)
>>> print(unpacked_sequences)
[tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]

```

Parameters
: **packed_sequences** ( [*PackedSequence*](torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence "torch.nn.utils.rnn.PackedSequence")  ) – A PackedSequence object.

Returns
:   a list of `Tensor`  objects

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  ]

