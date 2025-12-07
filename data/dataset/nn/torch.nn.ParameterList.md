ParameterList 
==============================================================

*class* torch.nn. ParameterList ( *values = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L632) 
:   Holds parameters in a list. 

[`ParameterList`](#torch.nn.ParameterList "torch.nn.ParameterList")  can be used like a regular Python
list, but Tensors that are `Parameter`  are properly registered,
and will be visible by all [`Module`](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  methods. 

Note that the constructor, assigning an element of the list, the [`append()`](#torch.nn.ParameterList.append "torch.nn.ParameterList.append")  method and the [`extend()`](#torch.nn.ParameterList.extend "torch.nn.ParameterList.extend")  method will convert any [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor")  into `Parameter`  . 

Parameters
: **parameters** ( *iterable* *,* *optional*  ) – an iterable of elements to add to the list.

Example: 

```
class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(10, 10)) for i in range(10)]
        )

    def forward(self, x):
        # ParameterList can act as an iterable, or be indexed using ints
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x

```

append ( *value* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L719) 
:   Append a given value at the end of the list. 

Parameters
: **value** ( *Any*  ) – value to append

Return type
:   [*Self*](https://docs.python.org/3/library/typing.html#typing.Self "(in Python v3.13)")

extend ( *values* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L730) 
:   Append values from a Python iterable to the end of the list. 

Parameters
: **values** ( *iterable*  ) – iterable of values to append

Return type
:   Self

