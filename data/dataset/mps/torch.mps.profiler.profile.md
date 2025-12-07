torch.mps.profiler.profile 
========================================================================================

torch.mps.profiler. profile ( *mode = 'interval'*  , *wait_until_completed = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mps/profiler.py#L46) 
:   Context Manager to enabling generating OS Signpost tracing from MPS backend. 

Parameters
:   * **mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – OS Signpost tracing mode could be “interval”, “event”,
or both “interval,event”.
The interval mode traces the duration of execution of the operations,
whereas event mode marks the completion of executions.
See document [Recording Performance Data](https://developer.apple.com/documentation/os/logging/recording_performance_data)  for more info.
* **wait_until_completed** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Waits until the MPS Stream complete
executing each encoded GPU operation. This helps generating single
dispatches on the trace’s timeline.
Note that enabling this option would affect the performance negatively.

