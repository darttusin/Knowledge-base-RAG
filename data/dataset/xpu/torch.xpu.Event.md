Event 
==============================================

*class* torch.xpu. Event ( *enable_timing = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/streams.py#L103) 
:   Wrapper around a XPU event. 

XPU events are synchronization markers that can be used to monitor the
device’s progress, and to synchronize XPU streams. 

The underlying XPU events are lazily initialized when the event is first
recorded. After creation, only streams on the same device may record the
event. However, streams on any device can wait on the event. 

Parameters
: **enable_timing** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – indicates if the event should measure time
(default: `False`  )

elapsed_time ( *end_event* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/streams.py#L149) 
:   Return the time elapsed. 

Time reported in milliseconds after the event was recorded and
before the end_event was recorded.

query ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/streams.py#L140) 
:   Check if all work currently captured by event has completed. 

Returns
:   A boolean indicating if all work currently captured by event has
completed.

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

record ( *stream = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/streams.py#L121) 
:   Record the event in a given stream. 

Uses `torch.xpu.current_stream()`  if no stream is specified. The
stream’s device must match the event’s device.

synchronize ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/streams.py#L157) 
:   Wait for the event to complete. 

Waits until the completion of all work currently captured in this event.
This prevents the CPU thread from proceeding until the event completes.

wait ( *stream = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/streams.py#L131) 
:   Make all future work submitted to the given stream wait for this event. 

Use `torch.xpu.current_stream()`  if no stream is specified.

