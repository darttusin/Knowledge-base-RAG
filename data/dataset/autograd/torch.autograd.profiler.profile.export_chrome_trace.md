torch.autograd.profiler.profile.export_chrome_trace 
============================================================================================================================================

profile. export_chrome_trace ( *path* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/profiler.py#L484) 
:   Export an EventList as a Chrome tracing tools file. 

The checkpoint can be later loaded and inspected under `chrome://tracing`  URL. 

Parameters
: **path** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – Path where the trace will be written.

