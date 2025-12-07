strict_fusion 
===============================================================

*class* torch.jit. strict_fusion [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/jit/__init__.py#L242) 
:   Give errors if not all nodes have been fused in inference, or symbolically differentiated in training. 

Example:
Forcing fusion of additions. 

```
@torch.jit.script
def foo(x):
    with torch.jit.strict_fusion():
        return x + x + x

```

