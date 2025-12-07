TorchDynamo-based ONNX Exporter 
===================================================================================================

* [Overview](#overview)
* [Dependencies](#dependencies)
* [A simple example](#a-simple-example)
* [Use the same model to compare with the TorchScript-enabled exporter](#use-the-same-model-to-compare-with-the-torchscript-enabled-exporter)
* [Inspecting the ONNX model using GUI](#inspecting-the-onnx-model-using-gui)
* [When the conversion fails](#when-the-conversion-fails)
* [Metadata](#metadata)
* [API Reference](#api-reference)
* [Deprecated](#deprecated)

[Overview](#id1) 
------------------------------------------------------------

The ONNX exporter leverages TorchDynamo engine to hook into Python’s frame evaluation API
and dynamically rewrite its bytecode into an FX Graph.
The resulting FX Graph is then polished before it is finally translated into an ONNX graph. 

The main advantage of this approach is that the [FX graph](https://localhost:8000/docs/stable/fx.html)  is captured using
bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques. 

In addition, during the export process, memory usage is significantly reduced compared to the TorchScript-enabled exporter.
See the [memory usage documentation](onnx_dynamo_memory_usage.html)  for more information.

[Dependencies](#id2) 
--------------------------------------------------------------------

The ONNX exporter depends on extra Python packages: 

* [ONNX](https://onnx.ai)
* [ONNX Script](https://microsoft.github.io/onnxscript)

They can be installed through [pip](https://pypi.org/project/pip/)  : 

```
  pip install --upgrade onnx onnxscript

```

[onnxruntime](https://onnxruntime.ai)  can then be used to execute the model
on a large variety of processors.

[A simple example](#id3) 
----------------------------------------------------------------------------

See below a demonstration of exporter API in action with a simple Multilayer Perceptron (MLP) as example: 

```
import torch
import torch.nn as nn

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = nn.Linear(8, 8, bias=True)
      self.fc1 = nn.Linear(8, 4, bias=True)
      self.fc2 = nn.Linear(4, 2, bias=True)
      self.fc3 = nn.Linear(2, 2, bias=True)
      self.fc_combined = nn.Linear(8 + 8 + 8, 8, bias=True)  # Combine all inputs

  def forward(self, tensor_x: torch.Tensor, input_dict: dict, input_list: list):
      """
      Forward method that requires all inputs:
      - tensor_x: A direct tensor input.
      - input_dict: A dictionary containing the tensor under the key 'tensor_x'.
      - input_list: A list where the first element is the tensor.
      """
      # Extract tensors from inputs
      dict_tensor = input_dict['tensor_x']
      list_tensor = input_list[0]

      # Combine all inputs into a single tensor
      combined_tensor = torch.cat([tensor_x, dict_tensor, list_tensor], dim=1)

      # Process the combined tensor through the layers
      combined_tensor = self.fc_combined(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc0(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc1(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc2(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      output = self.fc3(combined_tensor)
      return output

model = MLPModel()

# Example inputs
tensor_input = torch.rand((97, 8), dtype=torch.float32)
dict_input = {'tensor_x': torch.rand((97, 8), dtype=torch.float32)}
list_input = [torch.rand((97, 8), dtype=torch.float32)]

# The input_names and output_names are used to identify the inputs and outputs of the ONNX model
input_names = ['tensor_input', 'tensor_x', 'list_input_index_0']
output_names = ['output']

# Exporting the model with all required inputs
onnx_program = torch.onnx.export(model,(tensor_input, dict_input, list_input), dynamic_shapes=({0: "batch_size"},{"tensor_x": {0: "batch_size"}},[{0: "batch_size"}]), input_names=input_names, output_names=output_names, dynamo=True,)

# Check the exported ONNX model is dynamic
assert onnx_program.model.graph.inputs[0].shape == ("batch_size", 8)
assert onnx_program.model.graph.inputs[1].shape == ("batch_size", 8)
assert onnx_program.model.graph.inputs[2].shape == ("batch_size", 8)

```

As the code above shows, all you need is to provide [`torch.onnx.export()`](#torch.onnx.export "torch.onnx.export")  with an instance of the model and its input.
The exporter will then return an instance of [`torch.onnx.ONNXProgram`](#torch.onnx.ONNXProgram "torch.onnx.ONNXProgram")  that contains the exported ONNX graph along with extra information. 

The in-memory model available through `onnx_program.model_proto`  is an `onnx.ModelProto`  object in compliance with the [ONNX IR spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)  .
The ONNX model may then be serialized into a [Protobuf file](https://protobuf.dev/)  using the [`torch.onnx.ONNXProgram.save()`](#torch.onnx.ONNXProgram.save "torch.onnx.ONNXProgram.save")  API. 

```
  onnx_program.save("mlp.onnx")

```

[Use the same model to compare with the TorchScript-enabled exporter](#id4) 
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The biggest difference between the TorchScript-enabled exporter and the TorchDynamo-based exporter is that the latter
requires dynamic_shapes to be the same tree structure as the input, while the former
requires the dynamic_shapes to be a single and flatten dictionary. 

```
  torch.onnx.export(model,(tensor_input, dict_input, list_input), "mlp.onnx", dynamic_axes={"tensor_input":{0: "batch_size"}, "tensor_x": {0: "batch_size"}, "list_input_index_0": {0: "batch_size"}}, input_names=input_names, output_names=output_names)

```

[Inspecting the ONNX model using GUI](#id5) 
------------------------------------------------------------------------------------------------------------------

You can view the exported model using [Netron](https://netron.app/)  . 

[![MLP model as viewed using Netron](_images/onnx_dynamo_mlp_model.png)](_images/onnx_dynamo_mlp_model.png)

[When the conversion fails](#id6) 
----------------------------------------------------------------------------------------------

Function [`torch.onnx.export()`](#torch.onnx.export "torch.onnx.export")  should be called a second time with
parameter `report=True`  . A markdown report is generated to help the user
to resolve the issue.

[Metadata](#id7) 
------------------------------------------------------------

During ONNX export, each ONNX node is annotated with metadata that helps trace its origin and context from the original PyTorch model. This metadata is useful for debugging, model inspection, and understanding the mapping between PyTorch and ONNX graphs. 

The following metadata fields are added to each ONNX node: 

* **namespace**

    A string representing the hierarchical namespace of the node, consisting of a stack trace of modules/methods.

    *Example:* `__main__.SimpleAddModel/add: aten.add.Tensor`

* **pkg.torch.onnx.class_hierarchy**

    A list of class names representing the hierarchy of modules leading to this node.

    *Example:* `['__main__.SimpleAddModel', 'aten.add.Tensor']`

* **pkg.torch.onnx.fx_node**

    The string representation of the original FX node, including its name, number of consumers, the targeted torch op, arguments, and keyword arguments.

    *Example:* `%cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tensor_x, %input_dict_tensor_x, %input_list_0], 1), kwargs = {})`

* **pkg.torch.onnx.name_scopes**

    A list of name scopes (methods) representing the path to this node in the PyTorch model.

    *Example:* `['', 'add']`

* **pkg.torch.onnx.stack_trace**

    The stack trace from the original code where this node was created, if available.

    *Example:*

    ```
        File "simpleadd.py", line 7, in forward
            return torch.add(x, y)

    ```

These metadata fields are stored in the metadata_props attribute of each ONNX node and can be inspected using Netron or programmatically. 

The overall ONNX graph has the following `metadata_props`  : 

* **pkg.torch.export.ExportedProgram.graph_signature**

    This property contains a string representation of the graph_signature from the original PyTorch ExportedProgram. The graph signature describes the structure of the model’s inputs and outputs and how they map to the ONNX graph. The inputs are defined as `InputSpec`  objects, which include the kind of input (e.g., `InputKind.PARAMETER`  for parameters, `InputKind.USER_INPUT`  for user-defined inputs), the argument name, the target (which can be a specific node in the model), and whether the input is persistent. The outputs are defined as `OutputSpec`  objects, which specify the kind of output (e.g., `OutputKind.USER_OUTPUT`  ) and the argument name.

    To read more about the graph signature, please see the [torch.export](export.html)  for more information.

* **pkg.torch.export.ExportedProgram.range_constraints**

    This property contains a string representation of any range constraints that were present in the original PyTorch ExportedProgram. Range constraints specify valid ranges for symbolic shapes or values in the model, which can be important for models that use dynamic shapes or symbolic dimensions.

    *Example:* `s0: VR[2, int_oo]`  , which indicates that the size of the input tensor must be at least 2.

    To read more about range constraints, please see the [torch.export](export.html)  for more information.

Each input value in the ONNX graph may have the following metadata property: 

* **pkg.torch.export.graph_signature.InputSpec.kind**

    The kind of input, as defined by PyTorch’s InputKind enum.

    *Example values:*

    + “USER_INPUT”: A user-provided input to the model.
        + “PARAMETER”: A model parameter (e.g., weight).
        + “BUFFER”: A model buffer (e.g., running mean in BatchNorm).
        + “CONSTANT_TENSOR”: A constant tensor argument.
        + “CUSTOM_OBJ”: A custom object input.
        + “TOKEN”: A token input.
* **pkg.torch.export.graph_signature.InputSpec.persistent**

    Indicates whether the input is persistent (i.e., should be saved as part of the model’s state).

    *Example values:*

    + “True”
        + “False”

Each output value in the ONNX graph may have the following metadata property: 

* **pkg.torch.export.graph_signature.OutputSpec.kind**

    The kind of input, as defined by PyTorch’s OutputKind enum.

    *Example values:*

    + “USER_OUTPUT”: A user-visible output.
        + “LOSS_OUTPUT”: A loss value output.
        + “BUFFER_MUTATION”: Indicates a buffer was mutated.
        + “GRADIENT_TO_PARAMETER”: Gradient output for a parameter.
        + “GRADIENT_TO_USER_INPUT”: Gradient output for a user input.
        + “USER_INPUT_MUTATION”: Indicates a user input was mutated.
        + “TOKEN”: A token output.

Each initialized value, input, output has the following metadata: 

* **pkg.torch.onnx.original_node_name**

    The original name of the node in the PyTorch FX graph that produced this value in the case where the value was renamed. This helps trace initializers back to their source in the original model.

    *Example:* `fc1.weight`

[API Reference](#id8) 
----------------------------------------------------------------------

torch.onnx. export ( *model*  , *args=()*  , *f=None*  , *** , *kwargs=None*  , *export_params=True*  , *verbose=None*  , *input_names=None*  , *output_names=None*  , *opset_version=None*  , *dynamic_axes=None*  , *keep_initializers_as_inputs=False*  , *dynamo=False*  , *external_data=True*  , *dynamic_shapes=None*  , *custom_translation_table=None*  , *report=False*  , *optimize=True*  , *verify=False*  , *profile=False*  , *dump_exported_program=False*  , *artifacts_dir='.'*  , *fallback=False*  , *training=<TrainingMode.EVAL: 0>*  , *operator_export_type=<OperatorExportTypes.ONNX: 0>*  , *do_constant_folding=True*  , *custom_opsets=None*  , *export_modules_as_functions=False*  , *autograd_inlining=True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/__init__.py#L115) 
:   Exports a model into ONNX format. 

Setting `dynamo=True`  enables the new ONNX export logic
which is based on [`torch.export.ExportedProgram`](export.html#torch.export.ExportedProgram "torch.export.ExportedProgram")  and a more modern
set of translation logic. This is the recommended way to export models
to ONNX. 

When `dynamo=True`  : 

The exporter tries the following strategies to get an ExportedProgram for conversion to ONNX. 

1. If the model is already an ExportedProgram, it will be used as-is.
2. Use [`torch.export.export()`](export.html#torch.export.export "torch.export.export")  and set `strict=False`  .
3. Use [`torch.export.export()`](export.html#torch.export.export "torch.export.export")  and set `strict=True`  .
4. Use `draft_export`  which removes some soundness guarantees in data-dependent
operations to allow export to proceed. You will get a warning if the exporter
encounters any unsound data-dependent operation.
5. Use [`torch.jit.trace()`](generated/torch.jit.trace.html#torch.jit.trace "torch.jit.trace")  to trace the model then convert to ExportedProgram.
This is the most unsound strategy but may be useful for converting TorchScript
models to ONNX.

Parameters
:   * **model** ( [*torch.nn.Module*](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") *|* [*torch.export.ExportedProgram*](export.html#torch.export.ExportedProgram "torch.export.ExportedProgram") *|* [*torch.jit.ScriptModule*](generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule "torch.jit.ScriptModule") *|* [*torch.jit.ScriptFunction*](generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction "torch.jit.ScriptFunction")  ) – The model to be exported.
* **args** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* *Any* *,* *...* *]*  ) – Example positional inputs. Any non-Tensor arguments will be hard-coded into the
exported model; any Tensor arguments will become inputs of the exported model,
in the order they occur in the tuple.
* **f** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *|* [*os.PathLike*](https://docs.python.org/3/library/os.html#os.PathLike "(in Python v3.13)") *|* *None*  ) – Path to the output ONNX model file. E.g. “model.onnx”.
* **kwargs** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *Any* *]* *|* *None*  ) – Optional example keyword inputs.
* **export_params** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If false, parameters (weights) will not be exported.
* **verbose** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *|* *None*  ) – Whether to enable verbose logging.
* **input_names** ( *Sequence* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]* *|* *None*  ) – names to assign to the input nodes of the graph, in order.
* **output_names** ( *Sequence* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]* *|* *None*  ) – names to assign to the output nodes of the graph, in order.
* **opset_version** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *|* *None*  ) – The version of the [default (ai.onnx) opset](https://github.com/onnx/onnx/blob/master/docs/Operators.md)  to target. Must be >= 7.
* **dynamic_axes** ( *Mapping* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *Mapping* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]* *]* *|* *Mapping* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *Sequence* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]* *|* *None*  ) –

    By default the exported model will have the shapes of all input and output tensors
        set to exactly match those given in `args`  . To specify axes of tensors as
        dynamic (i.e. known only at run-time), set `dynamic_axes`  to a dict with schema:

    + KEY (str): an input or output name. Each name must also be provided in `input_names`  or
        :   `output_names`  .

        + VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
        :   list, each element is an axis index.
        For example:

    ```
        class SumModule(torch.nn.Module):
            def forward(self, x):
                return torch.sum(x, dim=1)

    torch.onnx.export(
            SumModule(),
            (torch.ones(2, 2),),
            "onnx.pb",
            input_names=["x"],
            output_names=["sum"],
        )

    ```

    Produces:

    ```
            input {
              name: "x"
              ...
                  shape {
                    dim {
                      dim_value: 2  # axis 0
                    }
                    dim {
                      dim_value: 2  # axis 1
            ...
            output {
              name: "sum"
              ...
                  shape {
                    dim {
                      dim_value: 2  # axis 0
            ...

    ```

    While:

    ```
            torch.onnx.export(
                SumModule(),
                (torch.ones(2, 2),),
                "onnx.pb",
                input_names=["x"],
                output_names=["sum"],
                dynamic_axes={
                    # dict value: manually named axes
                    "x": {0: "my_custom_axis_name"},
                    # list value: automatic names
                    "sum": [0],
                },
            )

    ```

    Produces:

    ```
            input {
              name: "x"
              ...
                  shape {
                    dim {
                      dim_param: "my_custom_axis_name"  # axis 0
                    }
                    dim {
                      dim_value: 2  # axis 1
            ...
            output {
              name: "sum"
              ...
                  shape {
                    dim {
                      dim_param: "sum_dynamic_axes_1"  # axis 0
            ...

    ```

* **keep_initializers_as_inputs** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) –

    If True, all the
        initializers (typically corresponding to model weights) in the
        exported graph will also be added as inputs to the graph. If False,
        then initializers are not added as inputs to the graph, and only
        the user inputs are added as inputs.

    Set this to True if you intend to supply model weights at runtime.
        Set it to False if the weights are static to allow for better optimizations
        (e.g. constant folding) by backends/runtimes.

* **dynamo** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to export the model with `torch.export`  ExportedProgram instead of TorchScript.
* **external_data** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to save the model weights as an external data file.
This is required for models with large weights that exceed the ONNX file size limit (2GB).
When False, the weights are saved in the ONNX file with the model architecture.
* **dynamic_shapes** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *Any* *]* *|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* *Any* *,* *...* *]* *|* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)") *[* *Any* *]* *|* *None*  ) – A dictionary or a tuple of dynamic shapes for the model inputs. Refer to [`torch.export.export()`](export.html#torch.export.export "torch.export.export")  for more details. This is only used (and preferred) when dynamo is True.
Note that dynamic_shapes is designed to be used when the model is exported with dynamo=True, while
dynamic_axes is used when dynamo=False.
* **custom_translation_table** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") *[* *Callable* *,* *Callable* *|* *Sequence* *[* *Callable* *]* *]* *|* *None*  ) – A dictionary of custom decompositions for operators in the model.
The dictionary should have the callable target in the fx Node as the key (e.g. `torch.ops.aten.stft.default`  ),
and the value should be a function that builds that graph using ONNX Script. This option
is only valid when dynamo is True.
* **report** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to generate a markdown report for the export process. This option
is only valid when dynamo is True.
* **optimize** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to optimize the exported model. This option
is only valid when dynamo is True. Default is True.
* **verify** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to verify the exported model using ONNX Runtime. This option
is only valid when dynamo is True.
* **profile** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to profile the export process. This option
is only valid when dynamo is True.
* **dump_exported_program** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to dump the [`torch.export.ExportedProgram`](export.html#torch.export.ExportedProgram "torch.export.ExportedProgram")  to a file.
This is useful for debugging the exporter. This option is only valid when dynamo is True.
* **artifacts_dir** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *|* [*os.PathLike*](https://docs.python.org/3/library/os.html#os.PathLike "(in Python v3.13)")  ) – The directory to save the debugging artifacts like the report and the serialized
exported program. This option is only valid when dynamo is True.
* **fallback** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to fallback to the TorchScript exporter if the dynamo exporter fails.
This option is only valid when dynamo is True. When fallback is enabled, It is
recommended to set dynamic_axes even when dynamic_shapes is provided.
* **training** ( *_C_onnx.TrainingMode*  ) – Deprecated option. Instead, set the training mode of the model before exporting.
* **operator_export_type** ( *_C_onnx.OperatorExportTypes*  ) – Deprecated option. Only ONNX is supported.
* **do_constant_folding** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Deprecated option.
* **custom_opsets** ( *Mapping* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *|* *None*  ) –

    Deprecated.
        A dictionary:

    + KEY (str): opset domain name
        + VALUE (int): opset version
        If a custom opset is referenced by `model`  but not mentioned in this dictionary,
        the opset version is set to 1. Only custom opset domain name and version should be
        indicated through this argument.

* **export_modules_as_functions** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *|* *Collection* *[* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.13)") *[* [*torch.nn.Module*](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") *]* *]*  ) –

    Deprecated option.

    Flag to enable
        exporting all `nn.Module`  forward calls as local functions in ONNX. Or a set to indicate the
        particular types of modules to export as local functions in ONNX.
        This feature requires `opset_version`  >= 15, otherwise the export will fail. This is because `opset_version`  < 15 implies IR version < 8, which means no local function support.
        Module variables will be exported as function attributes. There are two categories of function
        attributes.

    1. Annotated attributes: class variables that have type annotations via [PEP 526-style](https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations)  will be exported as attributes.
        Annotated attributes are not used inside the subgraph of ONNX local function because
        they are not created by PyTorch JIT tracing, but they may be used by consumers
        to determine whether or not to replace the function with a particular fused kernel.

    2. Inferred attributes: variables that are used by operators inside the module. Attribute names
        will have prefix “inferred::”. This is to differentiate from predefined attributes retrieved from
        python module annotations. Inferred attributes are used inside the subgraph of ONNX local function.

    + `False`  (default): export `nn.Module`  forward calls as fine grained nodes.
        + `True`  : export all `nn.Module`  forward calls as local function nodes.
        + Set of type of nn.Module: export `nn.Module`  forward calls as local function nodes,
        :   only if the type of the `nn.Module`  is found in the set.

* **autograd_inlining** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Deprecated.
Flag used to control whether to inline autograd functions.
Refer to [pytorch/pytorch#74765](https://github.com/pytorch/pytorch/pull/74765)  for more details.

Returns
:   [`torch.onnx.ONNXProgram`](#torch.onnx.ONNXProgram "torch.onnx.ONNXProgram")  if dynamo is True, otherwise None.

Return type
:   [ONNXProgram](#torch.onnx.ONNXProgram "torch.onnx.ONNXProgram")  | None

Changed in version 2.6: *training*  is now deprecated. Instead, set the training mode of the model before exporting. *operator_export_type*  is now deprecated. Only ONNX is supported. *do_constant_folding*  is now deprecated. It is always enabled. *export_modules_as_functions*  is now deprecated. *autograd_inlining*  is now deprecated.

Changed in version 2.7: *optimize*  is now True by default.

*class* torch.onnx. ONNXProgram ( *model*  , *exported_program* ) 
:   A class to represent an ONNX program that is callable with torch tensors. 

Variables
:   * **model** – The ONNX model as an ONNX IR model object.
* **exported_program** – The exported program that produced the ONNX model.

apply_weights ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/_internal/exporter/_onnx_program.py#L337) 
:   Apply the weights from the specified state dict to the ONNX model. 

Use this method to replace FakeTensors or other weights. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*torch.Tensor*](tensors.html#torch.Tensor "torch.Tensor") *]*  ) – The state dict containing the weights to apply to the ONNX model.

compute_values ( *value_names*  , *args = ()*  , *kwargs = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/_internal/exporter/_onnx_program.py#L225) 
:   Compute the values of the specified names in the ONNX model. 

This method is used to compute the values of the specified names in the ONNX model.
The values are returned as a dictionary mapping names to tensors. 

Parameters
: **value_names** ( *Sequence* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]*  ) – The names of the values to compute.

Returns
:   A dictionary mapping names to tensors.

Return type
:   Sequence[ [torch.Tensor](tensors.html#torch.Tensor "torch.Tensor")  ]

initialize_inference_session ( *initializer=<function _ort_session_initializer>* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/_internal/exporter/_onnx_program.py#L359) 
:   Initialize the ONNX Runtime inference session. 

Parameters
: **initializer** ( *Callable* *[* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *|* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.13)") *]* *,* *ort.InferenceSession* *]*  ) – The function to initialize the ONNX Runtime inference
session with the specified model. By default, it uses the `_ort_session_initializer()`  function.

*property* model_proto *: onnx.ModelProto* 
:   Return the ONNX `ModelProto`  object.

optimize ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/_internal/exporter/_onnx_program.py#L262) 
:   Optimize the ONNX model. 

This method optimizes the ONNX model by performing constant folding and
eliminating redundancies in the graph. The optimization is done in-place.

release ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/_internal/exporter/_onnx_program.py#L389) 
:   Release the inference session. 

You may call this method to release the resources used by the inference session.

save ( *destination*  , *** , *include_initializers = True*  , *keep_initializers_as_inputs = False*  , *external_data = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/_internal/exporter/_onnx_program.py#L270) 
:   Save the ONNX model to the specified destination. 

When `external_data`  is `True`  or the model is larger than 2GB,
the weights are saved as external data in a separate file. 

Initializer (model weights) serialization behaviors: 

* `include_initializers=True`  , `keep_initializers_as_inputs=False`  (default):
The initializers are included in the saved model.
* `include_initializers=True`  , `keep_initializers_as_inputs=True`  :
The initializers are included in the saved model and kept as model inputs.
Choose this option if you want the ability to override the model weights
during inference.
* `include_initializers=False`  , `keep_initializers_as_inputs=False`  :
The initializers are not included in the saved model and are not listed
as model inputs. Choose this option if you want to attach the initializers
to the ONNX model in a separate, post-processing, step.
* `include_initializers=False`  , `keep_initializers_as_inputs=True`  :
The initializers are not included in the saved model but are listed as model
inputs. Choose this option if you want to supply the initializers during
inference and want to minimize the size of the saved model.

Parameters
:   * **destination** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *|* [*os.PathLike*](https://docs.python.org/3/library/os.html#os.PathLike "(in Python v3.13)")  ) – The path to save the ONNX model to.
* **include_initializers** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to include the initializers in the saved model.
* **keep_initializers_as_inputs** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to keep the initializers as inputs in the saved model.
If *True* , the initializers are added as inputs to the model which means they can be overwritten.
by providing the initializers as model inputs.
* **external_data** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *|* *None*  ) – Whether to save the weights as external data in a separate file.

Raises
:   [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.13)")  – If `external_data`  is `True`  and `destination`  is not a file path.

torch.onnx. is_in_onnx_export ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/__init__.py#L548) 
:   Returns whether it is in the middle of ONNX export. 

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

*class* torch.onnx. OnnxExporterError 
:   Errors raised by the ONNX exporter. This is the base class for all exporter errors.

torch.onnx. enable_fake_mode ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/_internal/_exporter_legacy.py#L278) 
:   Enable fake mode for the duration of the context. 

Internally it instantiates a `torch._subclasses.fake_tensor.FakeTensorMode`  context manager
that converts user input and model parameters into `torch._subclasses.fake_tensor.FakeTensor`  . 

A `torch._subclasses.fake_tensor.FakeTensor`  is a [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")  with the ability to run PyTorch code without having to
actually do computation through tensors allocated on a `meta`  device. Because
there is no actual data being allocated on the device, this API allows for
initializing and exporting large models without the actual memory footprint needed for executing it. 

It is highly recommended to initialize the model in fake mode when exporting models that
are too large to fit into memory. 

Note 

This function does not support torch.onnx.export(…, dynamo=True, optimize=True).
Please call ONNXProgram.optimize() outside of the function after the model is exported.

Example: 

```
# xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
>>> import torch
>>> class MyModel(torch.nn.Module):  # Model with a parameter
...     def __init__(self) -> None:
...         super().__init__()
...         self.weight = torch.nn.Parameter(torch.tensor(42.0))
...     def forward(self, x):
...         return self.weight + x
>>> with torch.onnx.enable_fake_mode():
...     # When initialized in fake mode, the model's parameters are fake tensors
...     # They do not take up memory so we can initialize large models
...     my_nn_module = MyModel()
...     arg1 = torch.randn(2, 2, 2)
>>> onnx_program = torch.onnx.export(my_nn_module, (arg1,), dynamo=True, optimize=False)
>>> # Saving model WITHOUT initializers (only the architecture)
>>> onnx_program.save(
...     "my_model_without_initializers.onnx",
...     include_initializers=False,
...     keep_initializers_as_inputs=True,
... )
>>> # Saving model WITH initializers after applying concrete weights
>>> onnx_program.apply_weights({"weight": torch.tensor(42.0)})
>>> onnx_program.save("my_model_with_initializers.onnx")

```

Warning 

This API is experimental and is *NOT*  backward-compatible.

[Deprecated](#id9) 
----------------------------------------------------------------

The following classes and functions are deprecated and will be removed. 

torch.onnx. dynamo_export ( *model*  , */*  , ** model_args*  , *export_options = None*  , *** model_kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/__init__.py#L466) 
:   Export a torch.nn.Module to an ONNX graph. 

Deprecated since version 2.7:  Please use `torch.onnx.export(..., dynamo=True)`  instead.

Parameters
:   * **model** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*Module*](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module") *,* [*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.13)") *,* [*ExportedProgram*](export.html#torch.export.ExportedProgram "torch.export.exported_program.ExportedProgram") *]*  ) – The PyTorch model to be exported to ONNX.
* **model_args** – Positional inputs to `model`  .
* **model_kwargs** – Keyword inputs to `model`  .
* **export_options** ( [*torch.onnx.ExportOptions*](#torch.onnx.ExportOptions "torch.onnx.ExportOptions") *|* *None*  ) – Options to influence the export to ONNX.

Returns
:   An in-memory representation of the exported ONNX model.

Return type
:   [*ONNXProgram*](#torch.onnx.ONNXProgram "torch.onnx.ONNXProgram")

*class* torch.onnx. ExportOptions ( *** , *dynamic_shapes = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/__init__.py#L446) 
:   Options for dynamo_export. 

Deprecated since version 2.7:  Please use `torch.onnx.export(..., dynamo=True)`  instead.

Variables
: **dynamic_shapes** – Shape information hint for input/output tensors.
When `None`  , the exporter determines the most compatible setting.
When `True`  , all input shapes are considered dynamic.
When `False`  , all input shapes are considered static.

