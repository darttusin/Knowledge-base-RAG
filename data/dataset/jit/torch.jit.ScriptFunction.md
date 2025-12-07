ScriptFunction 
================================================================

*class* torch.jit. ScriptFunction 
:   Functionally equivalent to a [`ScriptModule`](torch.jit.ScriptModule.html#torch.jit.ScriptModule "torch.jit.ScriptModule")  , but represents a single
function and does not have any attributes or Parameters. 

get_debug_state ( *self : torch._C.ScriptFunction* ) → torch._C.GraphExecutorState 
:

save ( *self : torch._C.ScriptFunction*  , *filename : [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")*  , *_extra_files : [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") , [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") ] = {}* ) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") 
:

save_to_buffer ( *self : torch._C.ScriptFunction*  , *_extra_files : [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") , [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") ] = {}* ) → [bytes](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.13)") 
:

