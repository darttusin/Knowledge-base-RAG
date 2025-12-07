torch.mtia 
========================================================

The MTIA backend is implemented out of the tree, only interfaces are be defined here. 

This package enables an interface for accessing MTIA backend in python 

| [`StreamContext`](generated/torch.mtia.StreamContext.html#torch.mtia.StreamContext "torch.mtia.StreamContext") | Context-manager that selects a given stream. |
| --- | --- |
| [`current_device`](generated/torch.mtia.current_device.html#torch.mtia.current_device "torch.mtia.current_device") | Return the index of a currently selected device. |
| [`current_stream`](generated/torch.mtia.current_stream.html#torch.mtia.current_stream "torch.mtia.current_stream") | Return the currently selected [`Stream`](generated/torch.mtia.Stream.html#torch.mtia.Stream "torch.mtia.Stream")  for a given device. |
| [`default_stream`](generated/torch.mtia.default_stream.html#torch.mtia.default_stream "torch.mtia.default_stream") | Return the default [`Stream`](generated/torch.mtia.Stream.html#torch.mtia.Stream "torch.mtia.Stream")  for a given device. |
| [`device_count`](generated/torch.mtia.device_count.html#torch.mtia.device_count "torch.mtia.device_count") | Return the number of MTIA devices available. |
| [`init`](generated/torch.mtia.init.html#torch.mtia.init "torch.mtia.init") |  |
| [`is_available`](generated/torch.mtia.is_available.html#torch.mtia.is_available "torch.mtia.is_available") | Return true if MTIA device is available |
| [`is_initialized`](generated/torch.mtia.is_initialized.html#torch.mtia.is_initialized "torch.mtia.is_initialized") | Return whether PyTorch's MTIA state has been initialized. |
| [`memory_stats`](generated/torch.mtia.memory_stats.html#torch.mtia.memory_stats "torch.mtia.memory_stats") | Return a dictionary of MTIA memory allocator statistics for a given device. |
| [`get_device_capability`](generated/torch.mtia.get_device_capability.html#torch.mtia.get_device_capability "torch.mtia.get_device_capability") | Return capability of a given device as a tuple of (major version, minor version). |
| [`empty_cache`](generated/torch.mtia.empty_cache.html#torch.mtia.empty_cache "torch.mtia.empty_cache") | Empty the MTIA device cache. |
| [`record_memory_history`](generated/torch.mtia.record_memory_history.html#torch.mtia.record_memory_history "torch.mtia.record_memory_history") | Enable/Disable the memory profiler on MTIA allocator |
| [`snapshot`](generated/torch.mtia.snapshot.html#torch.mtia.snapshot "torch.mtia.snapshot") | Return a dictionary of MTIA memory allocator history |
| [`attach_out_of_memory_observer`](generated/torch.mtia.attach_out_of_memory_observer.html#torch.mtia.attach_out_of_memory_observer "torch.mtia.attach_out_of_memory_observer") | Attach an out-of-memory observer to MTIA memory allocator |
| [`set_device`](generated/torch.mtia.set_device.html#torch.mtia.set_device "torch.mtia.set_device") | Set the current device. |
| [`set_stream`](generated/torch.mtia.set_stream.html#torch.mtia.set_stream "torch.mtia.set_stream") | Set the current stream.This is a wrapper API to set the stream. |
| [`stream`](generated/torch.mtia.stream.html#torch.mtia.stream "torch.mtia.stream") | Wrap around the Context-manager StreamContext that selects a given stream. |
| [`synchronize`](generated/torch.mtia.synchronize.html#torch.mtia.synchronize "torch.mtia.synchronize") | Waits for all jobs in all streams on a MTIA device to complete. |
| [`device`](generated/torch.mtia.device.html#torch.mtia.device "torch.mtia.device") | Context-manager that changes the selected device. |
| [`set_rng_state`](generated/torch.mtia.set_rng_state.html#torch.mtia.set_rng_state "torch.mtia.set_rng_state") | Sets the random number generator state. |
| [`get_rng_state`](generated/torch.mtia.get_rng_state.html#torch.mtia.get_rng_state "torch.mtia.get_rng_state") | Returns the random number generator state as a ByteTensor. |
| [`DeferredMtiaCallError`](generated/torch.mtia.DeferredMtiaCallError.html#torch.mtia.DeferredMtiaCallError "torch.mtia.DeferredMtiaCallError") |  |

Streams and events 
------------------------------------------------------------------------

| [`Event`](generated/torch.mtia.Event.html#torch.mtia.Event "torch.mtia.Event") | Query and record Stream status to identify or control dependencies across Stream and measure timing. |
| --- | --- |
| [`Stream`](generated/torch.mtia.Stream.html#torch.mtia.Stream "torch.mtia.Stream") | An in-order queue of executing the respective tasks asynchronously in first in first out (FIFO) order. |

