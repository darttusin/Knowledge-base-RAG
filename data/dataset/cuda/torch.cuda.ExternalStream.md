ExternalStream 
================================================================

*class* torch.cuda. ExternalStream ( *stream_ptr*  , *device = None*  , *** kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/streams.py#L120) 
:   Wrapper around an externally allocated CUDA stream. 

This class is used to wrap streams allocated in other libraries in order
to facilitate data exchange and multi-library interactions. 

Note 

This class doesn’t manage the stream life-cycle, it is the user
responsibility to keep the referenced stream alive while this class is
being used.

Parameters
:   * **stream_ptr** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Integer representation of the *cudaStream_t* value.
allocated externally.
* **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the device where the stream
was originally allocated. If device is specified incorrectly,
subsequent launches using this stream may fail.

query ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/streams.py#L88) 
:   Check if all the work submitted has been completed. 

Returns
:   A boolean indicating if all kernels in this stream are completed.

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

record_event ( *event = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/streams.py#L73) 
:   Record an event. 

Parameters
: **event** ( [*torch.cuda.Event*](torch.cuda.Event.html#torch.cuda.Event "torch.cuda.Event") *,* *optional*  ) – event to record. If not given, a new one
will be allocated.

Returns
:   Recorded event.

synchronize ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/streams.py#L96) 
:   Wait for all the kernels in this stream to complete. 

Note 

This is a wrapper around `cudaStreamSynchronize()`  : see [CUDA Stream documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)  for more info.

wait_event ( *event* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/streams.py#L42) 
:   Make all future work submitted to the stream wait for an event. 

Parameters
: **event** ( [*torch.cuda.Event*](torch.cuda.Event.html#torch.cuda.Event "torch.cuda.Event")  ) – an event to wait for.

Note 

This is a wrapper around `cudaStreamWaitEvent()`  : see [CUDA Stream documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)  for more info. 

This function returns without waiting for `event`  : only future
operations are affected.

wait_stream ( *stream* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/streams.py#L59) 
:   Synchronize with another stream. 

All future work submitted to this stream will wait until all kernels
submitted to a given stream at the time of call complete. 

Parameters
: **stream** ( [*Stream*](torch.cuda.Stream.html#torch.cuda.Stream "torch.cuda.Stream")  ) – a stream to synchronize.

Note 

This function returns without waiting for currently enqueued
kernels in [`stream`](torch.cuda.stream.html#torch.cuda.stream "torch.cuda.stream")  : only future operations are affected.

