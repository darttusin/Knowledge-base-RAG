torch.jit.script_if_tracing 
============================================================================================

torch.jit. script_if_tracing ( *fn* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/jit/__init__.py#L177) 
:   Compiles `fn`  when it is first called during tracing. 

`torch.jit.script`  has a non-negligible start up time when it is first called due to
lazy-initializations of many compiler builtins. Therefore you should not use
it in library code. However, you may want to have parts of your library work
in tracing even if they use control flow. In these cases, you should use `@torch.jit.script_if_tracing`  to substitute for `torch.jit.script`  . 

Parameters
: **fn** – A function to compile.

Returns
:   If called during tracing, a [`ScriptFunction`](torch.jit.ScriptFunction.html#torch.jit.ScriptFunction "torch.jit.ScriptFunction")  created by *torch.jit.script* is returned.
Otherwise, the original function *fn* is returned.

