torch.cuda.is_current_stream_capturing 
===================================================================================================================

torch.cuda. is_current_stream_capturing ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/graphs.py#L25) 
:   Return True if CUDA graph capture is underway on the current CUDA stream, False otherwise. 

If a CUDA context does not exist on the current device, returns False without initializing the context.

