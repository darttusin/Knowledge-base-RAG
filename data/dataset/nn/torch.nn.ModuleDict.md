ModuleDict 
========================================================

*class* torch.nn. ModuleDict ( *modules = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L492) 
:   Holds submodules in a dictionary. 

[`ModuleDict`](#torch.nn.ModuleDict "torch.nn.ModuleDict")  can be indexed like a regular Python dictionary,
but modules it contains are properly registered, and will be visible by all [`Module`](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  methods. 

[`ModuleDict`](#torch.nn.ModuleDict "torch.nn.ModuleDict")  is an **ordered** dictionary that respects 

* the order of insertion, and
* in [`update()`](#torch.nn.ModuleDict.update "torch.nn.ModuleDict.update")  , the order of the merged `OrderedDict`  , `dict`  (started from Python 3.6) or another [`ModuleDict`](#torch.nn.ModuleDict "torch.nn.ModuleDict")  (the argument to [`update()`](#torch.nn.ModuleDict.update "torch.nn.ModuleDict.update")  ).

Note that [`update()`](#torch.nn.ModuleDict.update "torch.nn.ModuleDict.update")  with other unordered mapping
types (e.g., Python’s plain `dict`  before Python version 3.6) does not
preserve the order of the merged mapping. 

Parameters
: **modules** ( *iterable* *,* *optional*  ) – a mapping (dictionary) of (string: module)
or an iterable of key-value pairs of type (string, module)

Example: 

```
class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.choices = nn.ModuleDict(
            {"conv": nn.Conv2d(10, 10, 3), "pool": nn.MaxPool2d(3)}
        )
        self.activations = nn.ModuleDict(
            [["lrelu", nn.LeakyReLU()], ["prelu", nn.PReLU()]]
        )

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

```

clear ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L563) 
:   Remove all items from the ModuleDict.

items ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L582) 
:   Return an iterable of the ModuleDict key/value pairs. 

Return type
:   [*ItemsView*](https://docs.python.org/3/library/collections.abc.html#collections.abc.ItemsView "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")  ]

keys ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L577) 
:   Return an iterable of the ModuleDict keys. 

Return type
:   [*KeysView*](https://docs.python.org/3/library/collections.abc.html#collections.abc.KeysView "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ]

pop ( *key* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L567) 
:   Remove key from the ModuleDict and return its module. 

Parameters
: **key** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – key to pop from the ModuleDict

Return type
:   [*Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")

update ( *modules* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L592) 
:   Update the [`ModuleDict`](#torch.nn.ModuleDict "torch.nn.ModuleDict")  with key-value pairs from a mapping, overwriting existing keys. 

Note 

If [`modules`](../nn.html#module-torch.nn.modules "torch.nn.modules")  is an `OrderedDict`  , a [`ModuleDict`](#torch.nn.ModuleDict "torch.nn.ModuleDict")  , or
an iterable of key-value pairs, the order of new elements in it is preserved.

Parameters
: **modules** ( *iterable*  ) – a mapping (dictionary) from string to [`Module`](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  ,
or an iterable of key-value pairs of type (string, [`Module`](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  )

values ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/container.py#L587) 
:   Return an iterable of the ModuleDict values. 

Return type
:   [*ValuesView*](https://docs.python.org/3/library/collections.abc.html#collections.abc.ValuesView "(in Python v3.13)")  [ [*Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")  ]

