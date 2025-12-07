torch.jit.load 
================================================================

torch.jit. load ( *f*  , *map_location = None*  , *_extra_files = None*  , *_restore_shapes = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/jit/_serialization.py#L90) 
:   Load a [`ScriptModule`](torch.jit.ScriptModule.html#torch.jit.ScriptModule "torch.jit.ScriptModule")  or [`ScriptFunction`](torch.jit.ScriptFunction.html#torch.jit.ScriptFunction "torch.jit.ScriptFunction")  previously saved with [`torch.jit.save`](torch.jit.save.html#torch.jit.save "torch.jit.save")  . 

All previously saved modules, no matter their device, are first loaded onto CPU,
and then are moved to the devices they were saved from. If this fails (e.g.
because the run time system doesn’t have certain devices), an exception is
raised. 

Parameters
:   * **f** – a file-like object (has to implement read, readline, tell, and seek),
or a string containing a file name
* **map_location** ( *string* *or* [*torch.device*](../tensor_attributes.html#torch.device "torch.device")  ) – A simplified version of `map_location`  in *torch.jit.save* used to dynamically remap
storages to an alternative set of devices.
* **_extra_files** ( *dictionary* *of* *filename to content*  ) – The extra
filenames given in the map would be loaded and their content
would be stored in the provided map.
* **_restore_shapes** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether or not to retrace the module on load using stored inputs

Returns
:   A [`ScriptModule`](torch.jit.ScriptModule.html#torch.jit.ScriptModule "torch.jit.ScriptModule")  object.

Warning 

It is possible to construct malicious pickle data which will execute arbitrary code
during func: *torch.jit.load* . Never load data that could have come from an untrusted
source, or that could have been tampered with. **Only load data you trust** .

Example:
.. testcode: 

```
import torch
import io

torch.jit.load('scriptmodule.pt')

# Load ScriptModule from io.BytesIO object
with open('scriptmodule.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())

# Load all tensors to the original device
torch.jit.load(buffer)

# Load all tensors onto CPU, using a device
buffer.seek(0)
torch.jit.load(buffer, map_location=torch.device('cpu'))

# Load all tensors onto CPU, using a string
buffer.seek(0)
torch.jit.load(buffer, map_location='cpu')

# Load with extra files.
extra_files = {'foo.txt': ''}  # values will be replaced with data
torch.jit.load('scriptmodule.pt', _extra_files=extra_files)
print(extra_files['foo.txt'])

```

