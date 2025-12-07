ParameterDict 
==============================================================

*class* torch.nn. ParameterDict ( *parameters = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L776) 
:   Holds parameters in a dictionary. 

ParameterDict can be indexed like a regular Python dictionary, but Parameters it
contains are properly registered, and will be visible by all Module methods.
Other objects are treated as would be done by a regular Python dictionary 

[`ParameterDict`](#torch.nn.ParameterDict "torch.nn.ParameterDict")  is an **ordered** dictionary. [`update()`](#torch.nn.ParameterDict.update "torch.nn.ParameterDict.update")  with other unordered mapping
types (e.g., Python’s plain `dict`  ) does not preserve the order of the
merged mapping. On the other hand, `OrderedDict`  or another [`ParameterDict`](#torch.nn.ParameterDict "torch.nn.ParameterDict")  will preserve their ordering. 

Note that the constructor, assigning an element of the dictionary and the [`update()`](#torch.nn.ParameterDict.update "torch.nn.ParameterDict.update")  method will convert any [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor")  into `Parameter`  . 

Parameters
: **values** ( *iterable* *,* *optional*  ) – a mapping (dictionary) of
(string : Any) or an iterable of key-value pairs
of type (string, Any)

Example: 

```
class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.params = nn.ParameterDict(
            {
                "left": nn.Parameter(torch.randn(5, 10)),
                "right": nn.Parameter(torch.randn(5, 10)),
            }
        )

    def forward(self, x, choice):
        x = self.params[choice].mm(x)
        return x

```

clear ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L886) 
:   Remove all items from the ParameterDict.

copy ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L862) 
:   Return a copy of this [`ParameterDict`](#torch.nn.ParameterDict "torch.nn.ParameterDict")  instance. 

Return type
:   [*ParameterDict*](#torch.nn.ParameterDict "torch.nn.modules.container.ParameterDict")

fromkeys ( *keys*  , *default = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L919) 
:   Return a new ParameterDict with the keys provided. 

Parameters
:   * **keys** ( *iterable* *,* *string*  ) – keys to make the new ParameterDict from
* **default** ( [*Parameter*](torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter "torch.nn.parameter.Parameter") *,* *optional*  ) – value to set for all keys

Return type
:   [ParameterDict](#torch.nn.ParameterDict "torch.nn.ParameterDict")

get ( *key*  , *default = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L910) 
:   Return the parameter associated with key if present. Otherwise return default if provided, None if not. 

Parameters
:   * **key** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – key to get from the ParameterDict
* **default** ( [*Parameter*](torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter "torch.nn.parameter.Parameter") *,* *optional*  ) – value to return if key not present

Return type
:   [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")

items ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L934) 
:   Return an iterable of the ParameterDict key/value pairs. 

Return type
:   Iterable[ [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , Any]]

keys ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L930) 
:   Return an iterable of the ParameterDict keys. 

Return type
:   [*KeysView*](https://docs.python.org/3/library/collections.abc.html#collections.abc.KeysView "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ]

pop ( *key* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L891) 
:   Remove key from the ParameterDict and return its parameter. 

Parameters
: **key** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – key to pop from the ParameterDict

Return type
:   [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")

popitem ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L901) 
:   Remove and return the last inserted *(key, parameter)* pair from the ParameterDict. 

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

setdefault ( *key*  , *default = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L871) 
:   Set the default for a key in the Parameterdict. 

If key is in the ParameterDict, return its value.
If not, insert *key* with a parameter *default* and return *default* . *default* defaults to *None* . 

Parameters
:   * **key** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – key to set default for
* **default** ( *Any*  ) – the parameter set to the key

Return type
:   [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")

update ( *parameters* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L942) 
:   Update the [`ParameterDict`](#torch.nn.ParameterDict "torch.nn.ParameterDict")  with key-value pairs from `parameters`  , overwriting existing keys. 

Note 

If `parameters`  is an `OrderedDict`  , a [`ParameterDict`](#torch.nn.ParameterDict "torch.nn.ParameterDict")  , or
an iterable of key-value pairs, the order of new elements in it is preserved.

Parameters
: **parameters** ( *iterable*  ) – a mapping (dictionary) from string to `Parameter`  , or an iterable of
key-value pairs of type (string, `Parameter`  )

values ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L938) 
:   Return an iterable of the ParameterDict values. 

Return type
:   Iterable[Any]

