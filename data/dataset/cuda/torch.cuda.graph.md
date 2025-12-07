graph 
==============================================

*class* torch.cuda. graph ( *cuda_graph*  , *pool = None*  , *stream = None*  , *capture_error_mode = 'global'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/graphs.py#L153) 
:   Context-manager that captures CUDA work into a [`torch.cuda.CUDAGraph`](torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph "torch.cuda.CUDAGraph")  object for later replay. 

See [CUDA Graphs](../notes/cuda.html#cuda-graph-semantics)  for a general introduction,
detailed use, and constraints. 

Parameters
:   * **cuda_graph** ( [*torch.cuda.CUDAGraph*](torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph "torch.cuda.CUDAGraph")  ) – Graph object used for capture.
* **pool** ( *optional*  ) – Opaque token (returned by a call to [`graph_pool_handle()`](torch.cuda.graph_pool_handle.html#torch.cuda.graph_pool_handle "torch.cuda.graph_pool_handle")  or [`other_Graph_instance.pool()`](torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.pool "torch.cuda.CUDAGraph.pool")  ) hinting this graph’s capture
may share memory from the specified pool. See [Graph memory management](../notes/cuda.html#graph-memory-management)  .
* **stream** ( [*torch.cuda.Stream*](torch.cuda.Stream.html#torch.cuda.Stream "torch.cuda.Stream") *,* *optional*  ) – If supplied, will be set as the current stream in the context.
If not supplied, `graph`  sets its own internal side stream as the current stream in the context.
* **capture_error_mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – specifies the cudaStreamCaptureMode for the graph capture stream.
Can be “global”, “thread_local” or “relaxed”. During cuda graph capture, some actions, such as cudaMalloc,
may be unsafe. “global” will error on actions in other threads, “thread_local” will only error for
actions in the current thread, and “relaxed” will not error on actions. Do NOT change this setting
unless you’re familiar with [cudaStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85)

Note 

For effective memory sharing, if you pass a `pool`  used by a previous capture and the previous capture
used an explicit `stream`  argument, you should pass the same `stream`  argument to this capture.

Warning 

This API is in beta and may change in future releases.

