Subprocess Handling 
==========================================================================================================================

Retrieve SubprocessHandler 
----------------------------------------------------------------------------------------

torch.distributed.elastic.multiprocessing.subprocess_handler.handlers. get_subprocess_handler ( *entrypoint*  , *args*  , *env*  , *stdout*  , *stderr*  , *local_rank_id* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/elastic/multiprocessing/subprocess_handler/handlers.py#L15) 
:   Return type
:   [*SubprocessHandler*](#torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler.SubprocessHandler "torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler.SubprocessHandler")

SubprocessHandler 
----------------------------------------------------------------------

*class* torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler. SubprocessHandler ( *entrypoint*  , *args*  , *env*  , *stdout*  , *stderr*  , *local_rank_id* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py#L28) 
:   Convenience wrapper around python’s `subprocess.Popen`  . Keeps track of
meta-objects associated to the process (e.g. stdout and stderr redirect fds).

