torch.xpu 
=============================================================

This package introduces support for the XPU backend, specifically tailored for
Intel GPU optimization. 

This package is lazily initialized, so you can always import it, and use [`is_available()`](generated/torch.xpu.is_available.html#torch.xpu.is_available "torch.xpu.is_available")  to determine if your system supports XPU. 

| [`StreamContext`](generated/torch.xpu.StreamContext.html#torch.xpu.StreamContext "torch.xpu.StreamContext") | Context-manager that selects a given stream. |
| --- | --- |
| [`current_device`](generated/torch.xpu.current_device.html#torch.xpu.current_device "torch.xpu.current_device") | Return the index of a currently selected device. |
| [`current_stream`](generated/torch.xpu.current_stream.html#torch.xpu.current_stream "torch.xpu.current_stream") | Return the currently selected [`Stream`](generated/torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.Stream")  for a given device. |
| [`device`](generated/torch.xpu.device.html#torch.xpu.device "torch.xpu.device") | Context-manager that changes the selected device. |
| [`device_count`](generated/torch.xpu.device_count.html#torch.xpu.device_count "torch.xpu.device_count") | Return the number of XPU device available. |
| [`device_of`](generated/torch.xpu.device_of.html#torch.xpu.device_of "torch.xpu.device_of") | Context-manager that changes the current device to that of given object. |
| [`get_arch_list`](generated/torch.xpu.get_arch_list.html#torch.xpu.get_arch_list "torch.xpu.get_arch_list") | Return list XPU architectures this library was compiled for. |
| [`get_device_capability`](generated/torch.xpu.get_device_capability.html#torch.xpu.get_device_capability "torch.xpu.get_device_capability") | Get the xpu capability of a device. |
| [`get_device_name`](generated/torch.xpu.get_device_name.html#torch.xpu.get_device_name "torch.xpu.get_device_name") | Get the name of a device. |
| [`get_device_properties`](generated/torch.xpu.get_device_properties.html#torch.xpu.get_device_properties "torch.xpu.get_device_properties") | Get the properties of a device. |
| [`get_gencode_flags`](generated/torch.xpu.get_gencode_flags.html#torch.xpu.get_gencode_flags "torch.xpu.get_gencode_flags") | Return XPU AOT(ahead-of-time) build flags this library was compiled with. |
| [`get_stream_from_external`](generated/torch.xpu.get_stream_from_external.html#torch.xpu.get_stream_from_external "torch.xpu.get_stream_from_external") | Return a [`Stream`](generated/torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.Stream")  from an external SYCL queue. |
| [`init`](generated/torch.xpu.init.html#torch.xpu.init "torch.xpu.init") | Initialize PyTorch's XPU state. |
| [`is_available`](generated/torch.xpu.is_available.html#torch.xpu.is_available "torch.xpu.is_available") | Return a bool indicating if XPU is currently available. |
| [`is_initialized`](generated/torch.xpu.is_initialized.html#torch.xpu.is_initialized "torch.xpu.is_initialized") | Return whether PyTorch's XPU state has been initialized. |
| [`set_device`](generated/torch.xpu.set_device.html#torch.xpu.set_device "torch.xpu.set_device") | Set the current device. |
| [`set_stream`](generated/torch.xpu.set_stream.html#torch.xpu.set_stream "torch.xpu.set_stream") | Set the current stream.This is a wrapper API to set the stream. |
| [`stream`](generated/torch.xpu.stream.html#torch.xpu.stream "torch.xpu.stream") | Wrap around the Context-manager StreamContext that selects a given stream. |
| [`synchronize`](generated/torch.xpu.synchronize.html#torch.xpu.synchronize "torch.xpu.synchronize") | Wait for all kernels in all streams on a XPU device to complete. |

Random Number Generator 
----------------------------------------------------------------------------------

| [`get_rng_state`](generated/torch.xpu.get_rng_state.html#torch.xpu.get_rng_state "torch.xpu.get_rng_state") | Return the random number generator state of the specified GPU as a ByteTensor. |
| --- | --- |
| [`get_rng_state_all`](generated/torch.xpu.get_rng_state_all.html#torch.xpu.get_rng_state_all "torch.xpu.get_rng_state_all") | Return a list of ByteTensor representing the random number states of all devices. |
| [`initial_seed`](generated/torch.xpu.initial_seed.html#torch.xpu.initial_seed "torch.xpu.initial_seed") | Return the current random seed of the current GPU. |
| [`manual_seed`](generated/torch.xpu.manual_seed.html#torch.xpu.manual_seed "torch.xpu.manual_seed") | Set the seed for generating random numbers for the current GPU. |
| [`manual_seed_all`](generated/torch.xpu.manual_seed_all.html#torch.xpu.manual_seed_all "torch.xpu.manual_seed_all") | Set the seed for generating random numbers on all GPUs. |
| [`seed`](generated/torch.xpu.seed.html#torch.xpu.seed "torch.xpu.seed") | Set the seed for generating random numbers to a random number for the current GPU. |
| [`seed_all`](generated/torch.xpu.seed_all.html#torch.xpu.seed_all "torch.xpu.seed_all") | Set the seed for generating random numbers to a random number on all GPUs. |
| [`set_rng_state`](generated/torch.xpu.set_rng_state.html#torch.xpu.set_rng_state "torch.xpu.set_rng_state") | Set the random number generator state of the specified GPU. |
| [`set_rng_state_all`](generated/torch.xpu.set_rng_state_all.html#torch.xpu.set_rng_state_all "torch.xpu.set_rng_state_all") | Set the random number generator state of all devices. |

Streams and events 
------------------------------------------------------------------------

| [`Event`](generated/torch.xpu.Event.html#torch.xpu.Event "torch.xpu.Event") | Wrapper around a XPU event. |
| --- | --- |
| [`Stream`](generated/torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.Stream") | Wrapper around a XPU stream. |

Memory management 
----------------------------------------------------------------------

| [`empty_cache`](generated/torch.xpu.memory.empty_cache.html#torch.xpu.memory.empty_cache "torch.xpu.memory.empty_cache") | Release all unoccupied cached memory currently held by the caching allocator so that those can be used in other XPU application. |
| --- | --- |
| [`max_memory_allocated`](generated/torch.xpu.memory.max_memory_allocated.html#torch.xpu.memory.max_memory_allocated "torch.xpu.memory.max_memory_allocated") | Return the maximum GPU memory occupied by tensors in bytes for a given device. |
| [`max_memory_reserved`](generated/torch.xpu.memory.max_memory_reserved.html#torch.xpu.memory.max_memory_reserved "torch.xpu.memory.max_memory_reserved") | Return the maximum GPU memory managed by the caching allocator in bytes for a given device. |
| [`mem_get_info`](generated/torch.xpu.memory.mem_get_info.html#torch.xpu.memory.mem_get_info "torch.xpu.memory.mem_get_info") | Return the global free and total GPU memory for a given device. |
| [`memory_allocated`](generated/torch.xpu.memory.memory_allocated.html#torch.xpu.memory.memory_allocated "torch.xpu.memory.memory_allocated") | Return the current GPU memory occupied by tensors in bytes for a given device. |
| [`memory_reserved`](generated/torch.xpu.memory.memory_reserved.html#torch.xpu.memory.memory_reserved "torch.xpu.memory.memory_reserved") | Return the current GPU memory managed by the caching allocator in bytes for a given device. |
| [`memory_stats`](generated/torch.xpu.memory.memory_stats.html#torch.xpu.memory.memory_stats "torch.xpu.memory.memory_stats") | Return a dictionary of XPU memory allocator statistics for a given device. |
| [`memory_stats_as_nested_dict`](generated/torch.xpu.memory.memory_stats_as_nested_dict.html#torch.xpu.memory.memory_stats_as_nested_dict "torch.xpu.memory.memory_stats_as_nested_dict") | Return the result of `memory_stats()`  as a nested dictionary. |
| [`reset_accumulated_memory_stats`](generated/torch.xpu.memory.reset_accumulated_memory_stats.html#torch.xpu.memory.reset_accumulated_memory_stats "torch.xpu.memory.reset_accumulated_memory_stats") | Reset the "accumulated" (historical) stats tracked by the XPU memory allocator. |
| [`reset_peak_memory_stats`](generated/torch.xpu.memory.reset_peak_memory_stats.html#torch.xpu.memory.reset_peak_memory_stats "torch.xpu.memory.reset_peak_memory_stats") | Reset the "peak" stats tracked by the XPU memory allocator. |

