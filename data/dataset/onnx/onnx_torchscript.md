TorchScript-based ONNX Exporter 
===================================================================================================

Note 

To export an ONNX model using TorchDynamo instead of TorchScript, please see [Learn more about the TorchDynamo-based ONNX Exporter](onnx_dynamo.html)

* [Example: AlexNet from PyTorch to ONNX](#example-alexnet-from-pytorch-to-onnx)
* [Tracing vs Scripting](#tracing-vs-scripting)
* [Avoiding Pitfalls](#avoiding-pitfalls)

    + [Avoid NumPy and built-in Python types](#avoid-numpy-and-built-in-python-types)
        + [Avoid Tensor.data](#avoid-tensor-data)
        + [Avoid in-place operations when using tensor.shape in tracing mode](#avoid-in-place-operations-when-using-tensor-shape-in-tracing-mode)
* [Limitations](#limitations)

    + [Types](#types)
        + [Differences in Operator Implementations](#differences-in-operator-implementations)
        + [Unsupported Tensor Indexing Patterns](#unsupported-tensor-indexing-patterns)

    - [Reads / Gets](#reads-gets)
                - [Writes / Sets](#writes-sets)
* [Adding support for operators](#adding-support-for-operators)

    + [ONNX exporter internals](#onnx-exporter-internals)
        + [ATen operators](#aten-operators)

    - [List of supported operators](#list-of-supported-operators)
                - [Adding support for an aten or quantized operator](#adding-support-for-an-aten-or-quantized-operator)
        + [torch.autograd.Functions](#torch-autograd-functions)

    - [Static Symbolic Method](#static-symbolic-method)
                - [Inline Autograd Function](#inline-autograd-function)
        + [Custom operators](#custom-operators)

    - [ONNX-script functions](#onnx-script-functions)
                - [C++ Operators](#c-operators)
        + [Discovering all unconvertible ATen ops at once](#discovering-all-unconvertible-aten-ops-at-once)
* [Frequently Asked Questions](#frequently-asked-questions)
* [Python API](#module-torch.onnx)

    + [Functions](#functions)
        + [Classes](#classes)

[Example: AlexNet from PyTorch to ONNX](#id3) 
---------------------------------------------------------------------------------------------------------------------

Here is a simple script which exports a pretrained AlexNet to an ONNX file named `alexnet.onnx`  .
The call to `torch.onnx.export`  runs the model once to trace its execution and then exports the
traced model to the specified file: 

```
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = torchvision.models.alexnet(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

```

The resulting `alexnet.onnx`  file contains a binary [protocol buffer](https://developers.google.com/protocol-buffers/)  which contains both the network structure and parameters of the model you exported
(in this case, AlexNet). The argument `verbose=True`  causes the
exporter to print out a human-readable representation of the model: 

```
# These are the inputs and parameters to the network, which have taken on
# the names we specified earlier.
graph(%actual_input_1 : Float(10, 3, 224, 224)
      %learned_0 : Float(64, 3, 11, 11)
      %learned_1 : Float(64)
      %learned_2 : Float(192, 64, 5, 5)
      %learned_3 : Float(192)
      # ---- omitted for brevity ----
      %learned_14 : Float(1000, 4096)
      %learned_15 : Float(1000)) {
  # Every statement consists of some output tensors (and their types),
  # the operator to be run (with its attributes, e.g., kernels, strides,
  # etc.), its input tensors (%actual_input_1, %learned_0, %learned_1)
  %17 : Float(10, 64, 55, 55) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4]](%actual_input_1, %learned_0, %learned_1), scope: AlexNet/Sequential[features]/Conv2d[0]
  %18 : Float(10, 64, 55, 55) = onnx::Relu(%17), scope: AlexNet/Sequential[features]/ReLU[1]
  %19 : Float(10, 64, 27, 27) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%18), scope: AlexNet/Sequential[features]/MaxPool2d[2]
  # ---- omitted for brevity ----
  %29 : Float(10, 256, 6, 6) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%28), scope: AlexNet/Sequential[features]/MaxPool2d[12]
  # Dynamic means that the shape is not known. This may be because of a
  # limitation of our implementation (which we would like to fix in a
  # future release) or shapes which are truly dynamic.
  %30 : Dynamic = onnx::Shape(%29), scope: AlexNet
  %31 : Dynamic = onnx::Slice[axes=[0], ends=[1], starts=[0]](%30), scope: AlexNet
  %32 : Long() = onnx::Squeeze[axes=[0]](%31), scope: AlexNet
  %33 : Long() = onnx::Constant[value={9216}](), scope: AlexNet
  # ---- omitted for brevity ----
  %output1 : Float(10, 1000) = onnx::Gemm[alpha=1, beta=1, broadcast=1, transB=1](%45, %learned_14, %learned_15), scope: AlexNet/Sequential[classifier]/Linear[6]
  return (%output1);
}

```

You can also verify the output using the [ONNX](https://github.com/onnx/onnx/)  library,
which you can install using `pip`  : 

```
pip install onnx

```

Then, you can run: 

```
import onnx

# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

```

You can also run the exported model with one of the many [runtimes that support ONNX](https://onnx.ai/supported-tools.html#deployModel)  .
For example after installing [ONNX Runtime](https://www.onnxruntime.ai)  , you can
load and run the model: 

```
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("alexnet.onnx")

outputs = ort_session.run(
    None,
    {"actual_input_1": np.random.randn(10, 3, 224, 224).astype(np.float32)},
)
print(outputs[0])

```

Here is a more involved [tutorial on exporting a model and running it with ONNX Runtime](https://localhost:8000/tutorials/advanced/super_resolution_with_onnxruntime.html)  .

[Tracing vs Scripting](#id4) 
------------------------------------------------------------------------------------

Internally, [`torch.onnx.export()`](onnx_dynamo.html#torch.onnx.export "torch.onnx.export")  requires a [`torch.jit.ScriptModule`](generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule "torch.jit.ScriptModule")  rather than
a [`torch.nn.Module`](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  . If the passed-in model is not already a `ScriptModule`  , `export()`  will use *tracing*  to convert it to one: 

* **Tracing** : If `torch.onnx.export()`  is called with a Module that is not already a `ScriptModule`  , it first does the equivalent of [`torch.jit.trace()`](generated/torch.jit.trace.html#torch.jit.trace "torch.jit.trace")  , which executes the model
once with the given `args`  and records all operations that happen during that execution. This
means that if your model is dynamic, e.g., changes behavior depending on input data, the exported
model will *not*  capture this dynamic behavior.
We recommend examining the exported model and making sure the operators look
reasonable. Tracing will unroll loops and if statements, exporting a static graph that is exactly
the same as the traced run. If you want to export your model with dynamic control flow, you will
need to use *scripting*  .
* **Scripting** : Compiling a model via scripting preserves dynamic control flow and is valid for inputs
of different sizes. To use scripting:

    + Use [`torch.jit.script()`](generated/torch.jit.script.html#torch.jit.script "torch.jit.script")  to produce a `ScriptModule`  .
        + Call `torch.onnx.export()`  with the `ScriptModule`  as the model. The `args`  are still required,
        but they will be used internally only to produce example outputs, so that the types and shapes of the
        outputs can be captured. No tracing will be performed.

See [Introduction to TorchScript](https://localhost:8000/tutorials/beginner/Intro_to_TorchScript_tutorial.html)  and [TorchScript](jit.html)  for more details, including how to compose tracing and scripting to suit the
particular requirements of different models.

[Avoiding Pitfalls](#id5) 
------------------------------------------------------------------------------

### [Avoid NumPy and built-in Python types](#id6) 

PyTorch models can be written using NumPy or Python types and functions, but
during [tracing](#tracing-vs-scripting)  , any variables of NumPy or Python
types (rather than torch.Tensor) are converted to constants, which will produce
the wrong result if those values should change depending on the inputs. 

For example, rather than using numpy functions on numpy.ndarrays: 

```
# Bad! Will be replaced with constants during tracing.
x, y = np.random.rand(1, 2), np.random.rand(1, 2)
np.concatenate((x, y), axis=1)

```

Use torch operators on torch.Tensors: 

```
# Good! Tensor operations will be captured during tracing.
x, y = torch.randn(1, 2), torch.randn(1, 2)
torch.cat((x, y), dim=1)

```

And rather than use [`torch.Tensor.item()`](generated/torch.Tensor.item.html#torch.Tensor.item "torch.Tensor.item")  (which converts a Tensor to a Python
built-in number): 

```
# Bad! y.item() will be replaced with a constant during tracing.
def forward(self, x, y):
    return x.reshape(y.item(), -1)

```

Use torch’s support for implicit casting of single-element tensors: 

```
# Good! y will be preserved as a variable during tracing.
def forward(self, x, y):
    return x.reshape(y, -1)

```

### [Avoid Tensor.data](#id7) 

Using the Tensor.data field can produce an incorrect trace and therefore an incorrect ONNX graph.
Use [`torch.Tensor.detach()`](generated/torch.Tensor.detach.html#torch.Tensor.detach "torch.Tensor.detach")  instead. (Work is ongoing to [remove Tensor.data entirely](https://github.com/pytorch/pytorch/issues/30987)  ).

### [Avoid in-place operations when using tensor.shape in tracing mode](#id8) 

In tracing mode, shapes obtained from `tensor.shape`  are traced as tensors,
and share the same memory. This might cause a mismatch the final output values.
As a workaround, avoid the use of inplace operations in these scenarios.
For example, in the model: 

```
class Model(torch.nn.Module):
  def forward(self, states):
      batch_size, seq_length = states.shape[:2]
      real_seq_length = seq_length
      real_seq_length += 2
      return real_seq_length + seq_length

```

`real_seq_length`  and `seq_length`  share the same memory in tracing mode.
This could be avoided by rewriting the inplace operation: 

```
real_seq_length = real_seq_length + 2

```

[Limitations](#id9) 
------------------------------------------------------------------

### [Types](#id10) 

* Only `torch.Tensors`  , numeric types that can be trivially converted to torch.Tensors (e.g. float, int),
and tuples and lists of those types are supported as model inputs or outputs. Dict and str inputs and
outputs are accepted in [tracing](#tracing-vs-scripting)  mode, but:

    + Any computation that depends on the value of a dict or a str input **will be replaced with the
        constant value** seen during the one traced execution.

        + Any output that is a dict will be silently replaced with a **flattened sequence of its values
        (keys will be removed)** . E.g. `{"foo": 1, "bar": 2}`  becomes `(1, 2)`  .

        + Any output that is a str will be silently removed.
* Certain operations involving tuples and lists are not supported in [scripting](#tracing-vs-scripting)  mode due to limited support in ONNX for nested sequences.
In particular appending a tuple to a list is not supported. In tracing mode, the nested sequences
will be flattened automatically during the tracing.

### [Differences in Operator Implementations](#id11) 

Due to differences in implementations of operators, running the exported model on different runtimes
may produce different results from each other or from PyTorch. Normally these differences are
numerically small, so this should only be a concern if your application is sensitive to these
small differences.

### [Unsupported Tensor Indexing Patterns](#id12) 

Tensor indexing patterns that cannot be exported are listed below.
If you are experiencing issues exporting a model that does not include any of
the unsupported patterns below, please double check that you are exporting with
the latest `opset_version`  . 

#### [Reads / Gets](#id13) 

When indexing into a tensor for reading, the following patterns are not supported: 

```
# Tensor indices that includes negative values.
data[torch.tensor([[1, 2], [2, -3]]), torch.tensor([-2, 3])]
# Workarounds: use positive index values.

```

#### [Writes / Sets](#id14) 

When indexing into a Tensor for writing, the following patterns are not supported: 

```
# Multiple tensor indices if any has rank >= 2
data[torch.tensor([[1, 2], [2, 3]]), torch.tensor([2, 3])] = new_data
# Workarounds: use single tensor index with rank >= 2,
#              or multiple consecutive tensor indices with rank == 1.

# Multiple tensor indices that are not consecutive
data[torch.tensor([2, 3]), :, torch.tensor([1, 2])] = new_data
# Workarounds: transpose `data` such that tensor indices are consecutive.

# Tensor indices that includes negative values.
data[torch.tensor([1, -2]), torch.tensor([-2, 3])] = new_data
# Workarounds: use positive index values.

# Implicit broadcasting required for new_data.
data[torch.tensor([[0, 2], [1, 1]]), 1:3] = new_data
# Workarounds: expand new_data explicitly.
# Example:
#   data shape: [3, 4, 5]
#   new_data shape: [5]
#   expected new_data shape after broadcasting: [2, 2, 2, 5]

```

[Adding support for operators](#id15) 
-----------------------------------------------------------------------------------------------------

When exporting a model that includes unsupported operators, you’ll see an error message like: 

```
RuntimeError: ONNX export failed: Couldn't export operator foo

```

When that happens, there are a few things you can do: 

1. Change the model to not use that operator.
2. Create a symbolic function to convert the operator and register it as a custom symbolic function.
3. Contribute to PyTorch to add the same symbolic function to [`torch.onnx`](#module-torch.onnx "torch.onnx")  itself.

If you decided to implement a symbolic function (we hope you will contribute it back to PyTorch!), here is how you can get started: 

### [ONNX exporter internals](#id16) 

A “symbolic function” is a function that decomposes a PyTorch operator into a
composition of a series of ONNX operators. 

During export, each node (which contains a PyTorch operator) in the TorchScript
graph is visited by the exporter in topological order.
Upon visiting a node, the exporter looks for a registered symbolic functions for
that operator. Symbolic functions are implemented in Python. A symbolic function for
an op named `foo`  would look something like: 

```
def foo(
  g,
  input_0: torch._C.Value,
  input_1: torch._C.Value) -> Union[None, torch._C.Value, List[torch._C.Value]]:
  """
  Adds the ONNX operations representing this PyTorch function by updating the
  graph g with `g.op()` calls.

  Args:
    g (Graph): graph to write the ONNX representation into.
    input_0 (Value): value representing the variables which contain
        the first input for this operator.
    input_1 (Value): value representing the variables which contain
        the second input for this operator.

  Returns:
    A Value or List of Values specifying the ONNX nodes that compute something
    equivalent to the original PyTorch operator with the given inputs.

    None if it cannot be converted to ONNX.
  """
  ...

```

The `torch._C`  types are Python wrappers around the types defined in C++ in [ir.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/ir/ir.h)  . 

The process for adding a symbolic function depends on the type of operator.

### [ATen operators](#id17) 

[ATen](https://localhost:8000/cppdocs/#aten)  is PyTorch’s built-in tensor library.
If the operator is an ATen operator (shows up in the TorchScript graph with the prefix `aten::`  ), make sure it is not supported already. 

#### [List of supported operators](#id18) 

Visit the auto generated [list of supported TorchScript operators](onnx_torchscript_supported_aten_ops.html)  for details on which operator are supported in each `opset_version`  .

#### [Adding support for an aten or quantized operator](#id19) 

If the operator is not in the list above: 

* Define the symbolic function in `torch/onnx/symbolic_opset<version>.py`  , for example [torch/onnx/symbolic_opset9.py](https://github.com/pytorch/pytorch/blob/main/torch/onnx/symbolic_opset9.py)  .
Make sure the function has the same name as the ATen function, which may be declared in `torch/_C/_VariableFunctions.pyi`  or `torch/nn/functional.pyi`  (these files are generated at
build time, so will not appear in your checkout until you build PyTorch).
* By default, the first arg is the ONNX graph.
Other arg names must EXACTLY match the names in the `.pyi`  file,
because dispatch is done with keyword arguments.
* In the symbolic function, if the operator is in the [ONNX standard operator set](https://github.com/onnx/onnx/blob/master/docs/Operators.md)  ,
we only need to create a node to represent the ONNX operator in the graph.
If not, we can compose several standard operators that have the
equivalent semantics to the ATen operator.

Here is an example of handling missing symbolic function for the `ELU`  operator. 

If we run the following code: 

```
print(
    torch.jit.trace(
        torch.nn.ELU(), # module
        torch.ones(1)   # example input
    ).graph
)

```

We see something like: 

```
graph(%self : __torch__.torch.nn.modules.activation.___torch_mangle_0.ELU,
      %input : Float(1, strides=[1], requires_grad=0, device=cpu)):
  %4 : float = prim::Constant[value=1.]()
  %5 : int = prim::Constant[value=1]()
  %6 : int = prim::Constant[value=1]()
  %7 : Float(1, strides=[1], requires_grad=0, device=cpu) = aten::elu(%input, %4, %5, %6)
  return (%7)

```

Since we see `aten::elu`  in the graph, we know this is an ATen operator. 

We check the [ONNX operator list](https://github.com/onnx/onnx/blob/master/docs/Operators.md)  ,
and confirm that `Elu`  is standardized in ONNX. 

We find a signature for `elu`  in `torch/nn/functional.pyi`  : 

```
def elu(input: Tensor, alpha: float = ..., inplace: bool = ...) -> Tensor: ...

```

We add the following lines to `symbolic_opset9.py`  : 

```
def elu(g, input: torch.Value, alpha: torch.Value, inplace: bool = False):
    return g.op("Elu", input, alpha_f=alpha)

```

Now PyTorch is able to export models containing the `aten::elu`  operator! 

See the `torch/onnx/symbolic_opset*.py`  files for more examples.

### [torch.autograd.Functions](#id20) 

If the operator is a sub-class of [`torch.autograd.Function`](autograd.html#torch.autograd.Function "torch.autograd.Function")  , there are three ways
to export it. 

#### [Static Symbolic Method](#id21) 

You can add a static method named `symbolic`  to your function class. It should return
ONNX operators that represent the function’s behavior in ONNX. For example: 

```
class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        return g.op("Clip", input, g.op("Constant", value_t=torch.tensor(0, dtype=torch.float)))

```

#### [Inline Autograd Function](#id22) 

In cases where a static symbolic method is not provided for its subsequent [`torch.autograd.Function`](autograd.html#torch.autograd.Function "torch.autograd.Function")  or
where a function to register `prim::PythonOp`  as custom symbolic functions is not provided, [`torch.onnx.export()`](onnx_dynamo.html#torch.onnx.export "torch.onnx.export")  tries to inline the graph that corresponds to that [`torch.autograd.Function`](autograd.html#torch.autograd.Function "torch.autograd.Function")  such that
this function is broken down into individual operators that were used within the function.
The export should be successful as long as these individual operators are supported. For example: 

```
class MyLogExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        h = input.exp()
        return h.log().log()

```

There is no static symbolic method present for this model, yet it is exported as follows: 

```
graph(%input : Float(1, strides=[1], requires_grad=0, device=cpu)):
    %1 : float = onnx::Exp[](%input)
    %2 : float = onnx::Log[](%1)
    %3 : float = onnx::Log[](%2)
    return (%3)

```

If you need to avoid inlining of [`torch.autograd.Function`](autograd.html#torch.autograd.Function "torch.autograd.Function")  , you should export models with `operator_export_type`  set to `ONNX_FALLTHROUGH`  or `ONNX_ATEN_FALLBACK`  .

### [Custom operators](#id23) 

You can export your model with custom operators that includes a combination of many standard ONNX ops,
or are driven by self-defined C++ backend. 

#### [ONNX-script functions](#id24) 

If an operator is not a standard ONNX op, but can be composed of multiple existing ONNX ops, you can utilize [ONNX-script](https://github.com/microsoft/onnx-script)  to create an external ONNX function to support the operator.
You can export it by following this example: 

```
import onnxscript
# There are three opset version needed to be aligned
# This is (1) the opset version in ONNX function
from onnxscript.onnx_opset import opset15 as op
opset_version = 15

x = torch.randn(1, 2, 3, 4, requires_grad=True)
model = torch.nn.SELU()

custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)

@onnxscript.script(custom_opset)
def Selu(X):
    alpha = 1.67326  # auto wrapped as Constants
    gamma = 1.0507
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)

# setType API provides shape/type to ONNX shape/type inference
def custom_selu(g: jit_utils.GraphContext, X):
    return g.onnxscript_op(Selu, X).setType(X.type())

# Register custom symbolic function
# There are three opset version needed to be aligned
# This is (2) the opset version in registry
torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::selu",
    symbolic_fn=custom_selu,
    opset_version=opset_version,
)

# There are three opset version needed to be aligned
# This is (2) the opset version in exporter
torch.onnx.export(
    model,
    x,
    "model.onnx",
    opset_version=opset_version,
    # only needed if you want to specify an opset version > 1.
    custom_opsets={"onnx-script": 2}
)

```

The example above exports it as a custom operator in the “onnx-script” opset.
When exporting a custom operator, you can specify the custom domain version using the `custom_opsets`  dictionary at export. If not specified, the custom opset version defaults to 1. 

NOTE: Be careful to align the opset version mentioned in the above example, and make sure they are consumed in exporter step.
The example usage of how to write a onnx-script function is a beta version in terms of the active development on onnx-script.
Please follow the latest [ONNX-script](https://github.com/microsoft/onnx-script)

#### [C++ Operators](#id25) 

If a model uses a custom operator implemented in C++ as described in [Extending TorchScript with Custom C++ Operators](https://localhost:8000/tutorials/advanced/torch_script_custom_ops.html)  ,
you can export it by following this example: 

```
from torch.onnx import symbolic_helper

# Define custom symbolic function
@symbolic_helper.parse_args("v", "v", "f", "i")
def symbolic_foo_forward(g, input1, input2, attr1, attr2):
    return g.op("custom_domain::Foo", input1, input2, attr1_f=attr1, attr2_i=attr2)

# Register custom symbolic function
torch.onnx.register_custom_op_symbolic("custom_ops::foo_forward", symbolic_foo_forward, 9)

class FooModel(torch.nn.Module):
    def __init__(self, attr1, attr2):
        super().__init__()
        self.attr1 = attr1
        self.attr2 = attr2

    def forward(self, input1, input2):
        # Calling custom op
        return torch.ops.custom_ops.foo_forward(input1, input2, self.attr1, self.attr2)

model = FooModel(attr1, attr2)
torch.onnx.export(
    model,
    (example_input1, example_input1),
    "model.onnx",
    # only needed if you want to specify an opset version > 1.
    custom_opsets={"custom_domain": 2}
)

```

The example above exports it as a custom operator in the “custom_domain” opset.
When exporting a custom operator, you can specify the custom domain version using the `custom_opsets`  dictionary at export. If not specified, the custom opset version defaults to 1. 

The runtime that consumes the model needs to support the custom op. See [Caffe2 custom ops](https://caffe2.ai/docs/custom-operators.html)  , [ONNX Runtime custom ops](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)  ,
or your runtime of choice’s documentation.

### [Discovering all unconvertible ATen ops at once](#id26) 

When export fails due to an unconvertible ATen op, there may in fact be more
than one such op but the error message only mentions the first. To discover
all of the unconvertible ops in one go you can: 

```
# prepare model, args, opset_version
...

torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
    model, args, opset_version=opset_version
)

print(set(unconvertible_ops))

```

The set is approximated because some ops may be removed during the conversion
process and don’t need to be converted. Some other ops may have partial support
that will fail conversion with particular inputs, but this should give you a
general idea of what ops are not supported. Please feel free to open GitHub Issues
for op support requests.

[Frequently Asked Questions](#id27) 
-------------------------------------------------------------------------------------------------

Q: I have exported my LSTM model, but its input size seems to be fixed? 

> The tracer records the shapes of the example inputs. If the model should accept
> inputs of dynamic shapes, set `dynamic_axes`  when calling [`torch.onnx.export()`](onnx_dynamo.html#torch.onnx.export "torch.onnx.export")  .

Q: How to export models containing loops? 

> See [Tracing vs Scripting](#id1)  .

Q: How to export models with primitive type inputs (e.g. int, float)? 

> Support for primitive numeric type inputs was added in PyTorch 1.9.
> However, the exporter does not support models with str inputs.

Q: Does ONNX support implicit scalar datatype casting? 

> The ONNX standard does not, but the exporter will try to handle that part.
> Scalars are exported as constant tensors.
> The exporter will figure out the right data type for scalars. In rare cases when it is unable
> to do so, you will need to manually specify the datatype with e.g. *dtype=torch.float32* .
> If you see any errors, please [create a GitHub issue]( [pytorch/pytorch#issues](https://github.com/pytorch/pytorch/issues)  ).

Q: Are lists of Tensors exportable to ONNX? 

> Yes, for `opset_version`  >= 11, since ONNX introduced the Sequence type in opset 11.

[Python API](#id28) 
------------------------------------------------------------------------

### [Functions](#id29) 

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
:   [`torch.onnx.ONNXProgram`](onnx_dynamo.html#torch.onnx.ONNXProgram "torch.onnx.ONNXProgram")  if dynamo is True, otherwise None.

Return type
:   [ONNXProgram](onnx_dynamo.html#torch.onnx.ONNXProgram "torch.onnx.ONNXProgram")  | None

Changed in version 2.6: *training*  is now deprecated. Instead, set the training mode of the model before exporting. *operator_export_type*  is now deprecated. Only ONNX is supported. *do_constant_folding*  is now deprecated. It is always enabled. *export_modules_as_functions*  is now deprecated. *autograd_inlining*  is now deprecated.

Changed in version 2.7: *optimize*  is now True by default.

torch.onnx. register_custom_op_symbolic ( *symbolic_name*  , *symbolic_fn*  , *opset_version* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/utils.py#L1785) 
:   Registers a symbolic function for a custom operator. 

When the user registers symbolic for custom/contrib ops,
it is highly recommended to add shape inference for that operator via setType API,
otherwise the exported graph may have incorrect shape inference in some extreme cases.
An example of setType is *test_aten_embedding_2* in *test_operators.py* . 

See “Custom Operators” in the module documentation for an example usage. 

Parameters
:   * **symbolic_name** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – The name of the custom operator in “<domain>::<op>”
format.
* **symbolic_fn** ( *Callable*  ) – A function that takes in the ONNX graph and
the input arguments to the current operator, and returns new
operator nodes to add to the graph.
* **opset_version** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The ONNX opset version in which to register.

torch.onnx. unregister_custom_op_symbolic ( *symbolic_name*  , *opset_version* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/utils.py#L1815) 
:   Unregisters `symbolic_name`  . 

See “Custom Operators” in the module documentation for an example usage. 

Parameters
:   * **symbolic_name** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – The name of the custom operator in “<domain>::<op>”
format.
* **opset_version** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The ONNX opset version in which to unregister.

torch.onnx. select_model_mode_for_export ( *model*  , *mode* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/utils.py#L53) 
:   A context manager to temporarily set the training mode of `model`  to `mode`  , resetting it when we exit the with-block. 

Deprecated since version 2.7:  Please set training mode before exporting the model.

Parameters
:   * **model** – Same type and meaning as `model`  arg to [`export()`](onnx_dynamo.html#torch.onnx.export "torch.onnx.export")  .
* **mode** ( *TrainingMode*  ) – Same type and meaning as `training`  arg to [`export()`](onnx_dynamo.html#torch.onnx.export "torch.onnx.export")  .

torch.onnx. is_in_onnx_export ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/__init__.py#L548)
:   Returns whether it is in the middle of ONNX export. 

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

### [Classes](#id30) 

| [`JitScalarType`](generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType "torch.onnx.JitScalarType") | Scalar types defined in torch. |
| --- | --- |

