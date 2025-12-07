torch.cuda.get_stream_from_external 
=============================================================================================================

torch.cuda. get_stream_from_external ( *data_ptr*  , *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L1137) 
:   Return a [`Stream`](torch.cuda.Stream.html#torch.cuda.Stream "torch.cuda.Stream")  from an externally allocated CUDA stream. 

This function is used to wrap streams allocated in other libraries in order
to facilitate data exchange and multi-library interactions. 

Note 

This function doesn’t manage the stream life-cycle, it is the user
responsibility to keep the referenced stream alive while this returned
stream is being used.

Parameters
:   * **data_ptr** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Integer representation of the *cudaStream_t* value that
is allocated externally.
* **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the device where the stream
was originally allocated. If device is specified incorrectly,
subsequent launches using this stream may fail.

Return type
:   [*Stream*](torch.cuda.Stream.html#torch.cuda.Stream "torch.cuda.streams.Stream")

