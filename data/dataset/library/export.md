torch.export 
============================================================

Warning 

This feature is a prototype under active development and there WILL BE
BREAKING CHANGES in the future.

Overview 
----------------------------------------------------

[`torch.export.export()`](#torch.export.export "torch.export.export")  takes a [`torch.nn.Module`](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  and produces a traced graph
representing only the Tensor computation of the function in an Ahead-of-Time
(AOT) fashion, which can subsequently be executed with different outputs or
serialized. 

```
import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: torch.export.ExportedProgram = export(
    Mod(), args=example_args
)
print(exported_program)

```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[10, 10]", y: "f32[10, 10]"):
            # code: a = torch.sin(x)
            sin: "f32[10, 10]" = torch.ops.aten.sin.default(x)

            # code: b = torch.cos(y)
            cos: "f32[10, 10]" = torch.ops.aten.cos.default(y)

            # code: return a + b
            add: f32[10, 10] = torch.ops.aten.add.Tensor(sin, cos)
            return (add,)

    Graph signature:
        ExportGraphSignature(
            input_specs=[
                InputSpec(
                    kind=<InputKind.USER_INPUT: 1>,
                    arg=TensorArgument(name='x'),
                    target=None,
                    persistent=None
                ),
                InputSpec(
                    kind=<InputKind.USER_INPUT: 1>,
                    arg=TensorArgument(name='y'),
                    target=None,
                    persistent=None
                )
            ],
            output_specs=[
                OutputSpec(
                    kind=<OutputKind.USER_OUTPUT: 1>,
                    arg=TensorArgument(name='add'),
                    target=None
                )
            ]
        )
    Range constraints: {}

```

`torch.export`  produces a clean intermediate representation (IR) with the
following invariants. More specifications about the IR can be found [here](export.ir_spec.html#export-ir-spec)  . 

* **Soundness** : It is guaranteed to be a sound representation of the original
program, and maintains the same calling conventions of the original program.
* **Normalized** : There are no Python semantics within the graph. Submodules
from the original programs are inlined to form one fully flattened
computational graph.
* **Graph properties** : The graph is purely functional, meaning it does not
contain operations with side effects such as mutations or aliasing. It does
not mutate any intermediate values, parameters, or buffers.
* **Metadata** : The graph contains metadata captured during tracing, such as a
stacktrace from user’s code.

Under the hood, `torch.export`  leverages the following latest technologies: 

* **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython feature
called the Frame Evaluation API to safely trace PyTorch graphs. This
provides a massively improved graph capturing experience, with much fewer
rewrites needed in order to fully trace the PyTorch code.
* **AOT Autograd** provides a functionalized PyTorch graph and ensures the graph
is decomposed/lowered to the ATen operator set.
* **Torch FX (torch.fx)** is the underlying representation of the graph,
allowing flexible Python-based transformations.

### Existing frameworks 

[`torch.compile()`](generated/torch.compile.html#torch.compile "torch.compile")  also utilizes the same PT2 stack as `torch.export`  , but
is slightly different: 

* **JIT vs. AOT** : [`torch.compile()`](generated/torch.compile.html#torch.compile "torch.compile")  is a JIT compiler whereas
which is not intended to be used to produce compiled artifacts outside of
deployment.
* **Partial vs. Full Graph Capture** : When [`torch.compile()`](generated/torch.compile.html#torch.compile "torch.compile")  runs into an
untraceable part of a model, it will “graph break” and fall back to running
the program in the eager Python runtime. In comparison, `torch.export`  aims
to get a full graph representation of a PyTorch model, so it will error out
when something untraceable is reached. Since `torch.export`  produces a full
graph disjoint from any Python features or runtime, this graph can then be
saved, loaded, and run in different environments and languages.
* **Usability tradeoff** : Since [`torch.compile()`](generated/torch.compile.html#torch.compile "torch.compile")  is able to fallback to the
Python runtime whenever it reaches something untraceable, it is a lot more
flexible. `torch.export`  will instead require users to provide more
information or rewrite their code to make it traceable.

Compared to [`torch.fx.symbolic_trace()`](fx.html#torch.fx.symbolic_trace "torch.fx.symbolic_trace")  , `torch.export`  traces using
TorchDynamo which operates at the Python bytecode level, giving it the ability
to trace arbitrary Python constructs not limited by what Python operator
overloading supports. Additionally, `torch.export`  keeps fine-grained track of
tensor metadata, so that conditionals on things like tensor shapes do not
fail tracing. In general, `torch.export`  is expected to work on more user
programs, and produce lower-level graphs (at the `torch.ops.aten`  operator
level). Note that users can still use [`torch.fx.symbolic_trace()`](fx.html#torch.fx.symbolic_trace "torch.fx.symbolic_trace")  as a
preprocessing step before `torch.export`  . 

Compared to [`torch.jit.script()`](generated/torch.jit.script.html#torch.jit.script "torch.jit.script")  , `torch.export`  does not capture Python
control flow or data structures, but it supports more Python language features
than TorchScript (as it is easier to have comprehensive coverage over Python
bytecodes). The resulting graphs are simpler and only have straight line control
flow (except for explicit control flow operators). 

Compared to [`torch.jit.trace()`](generated/torch.jit.trace.html#torch.jit.trace "torch.jit.trace")  , `torch.export`  is sound: it is able to
trace code that performs integer computation on sizes and records all of the
side-conditions necessary to show that a particular trace is valid for other
inputs.

Exporting a PyTorch Model 
--------------------------------------------------------------------------------------

### An Example 

The main entrypoint is through [`torch.export.export()`](#torch.export.export "torch.export.export")  , which takes a
callable ( [`torch.nn.Module`](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  , function, or method) and sample inputs, and
captures the computation graph into an [`torch.export.ExportedProgram`](#torch.export.ExportedProgram "torch.export.ExportedProgram")  . An
example: 

```
import torch
from torch.export import export

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

example_args = (torch.randn(1, 3, 256, 256),)
example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)

```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
    def forward(self, p_conv_weight: "f32[16, 3, 3, 3]", p_conv_bias: "f32[16]", x: "f32[1, 3, 256, 256]", constant: "f32[1, 16, 256, 256]"):
            # code: a = self.conv(x)
            conv2d: "f32[1, 16, 256, 256]" = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias, [1, 1], [1, 1])

            # code: a.add_(constant)
            add_: "f32[1, 16, 256, 256]" = torch.ops.aten.add_.Tensor(conv2d, constant)

            # code: return self.maxpool(self.relu(a))
            relu: "f32[1, 16, 256, 256]" = torch.ops.aten.relu.default(add_)
            max_pool2d: "f32[1, 16, 85, 85]" = torch.ops.aten.max_pool2d.default(relu, [3, 3], [3, 3])
            return (max_pool2d,)

Graph signature:
    ExportGraphSignature(
        input_specs=[
            InputSpec(
                kind=<InputKind.PARAMETER: 2>,
                arg=TensorArgument(name='p_conv_weight'),
                target='conv.weight',
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.PARAMETER: 2>,
                arg=TensorArgument(name='p_conv_bias'),
                target='conv.bias',
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.USER_INPUT: 1>,
                arg=TensorArgument(name='x'),
                target=None,
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.USER_INPUT: 1>,
                arg=TensorArgument(name='constant'),
                target=None,
                persistent=None
            )
        ],
        output_specs=[
            OutputSpec(
                kind=<OutputKind.USER_OUTPUT: 1>,
                arg=TensorArgument(name='max_pool2d'),
                target=None
            )
        ]
    )
Range constraints: {}

```

Inspecting the `ExportedProgram`  , we can note the following: 

* The [`torch.fx.Graph`](fx.html#torch.fx.Graph "torch.fx.Graph")  contains the computation graph of the original
program, along with records of the original code for easy debugging.
* The graph contains only `torch.ops.aten`  operators found [here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)  and custom operators, and is fully functional, without any inplace operators
such as `torch.add_`  .
* The parameters (weight and bias to conv) are lifted as inputs to the graph,
resulting in no `get_attr`  nodes in the graph, which previously existed in
the result of [`torch.fx.symbolic_trace()`](fx.html#torch.fx.symbolic_trace "torch.fx.symbolic_trace")  .
* The [`torch.export.ExportGraphSignature`](#torch.export.ExportGraphSignature "torch.export.ExportGraphSignature")  models the input and output
signature, along with specifying which inputs are parameters.
* The resulting shape and dtype of tensors produced by each node in the graph is
noted. For example, the `convolution`  node will result in a tensor of dtype `torch.float32`  and shape (1, 16, 256, 256).

### Non-Strict Export 

In PyTorch 2.3, we introduced a new mode of tracing called **non-strict mode** .
It’s still going through hardening, so if you run into any issues, please file
them to Github with the “oncall: export” tag. 

In *non-strict mode*  , we trace through the program using the Python interpreter.
Your code will execute exactly as it would in eager mode; the only difference is
that all Tensor objects will be replaced by ProxyTensors, which will record all
their operations into a graph. 

In *strict*  mode, which is currently the default, we first trace through the
program using TorchDynamo, a bytecode analysis engine. TorchDynamo does not
actually execute your Python code. Instead, it symbolically analyzes it and
builds a graph based on the results. This analysis allows torch.export to
provide stronger guarantees about safety, but not all Python code is supported. 

An example of a case where one might want to use non-strict mode is if you run
into a unsupported TorchDynamo feature that might not be easily solved, and you
know the python code is not exactly needed for computation. For example: 

```
import contextlib
import torch

class ContextManager():
    def __init__(self):
        self.count = 0
    def __enter__(self):
        self.count += 1
    def __exit__(self, exc_type, exc_value, traceback):
        self.count -= 1

class M(torch.nn.Module):
    def forward(self, x):
        with ContextManager():
            return x.sin() + x.cos()

export(M(), (torch.ones(3, 3),), strict=False)  # Non-strict traces successfully
export(M(), (torch.ones(3, 3),))  # Strict mode fails with torch._dynamo.exc.Unsupported: ContextManager

```

In this example, the first call using non-strict mode (through the `strict=False`  flag) traces successfully whereas the second call using strict
mode (default) results with a failure, where TorchDynamo is unable to support
context managers. One option is to rewrite the code (see [Limitations of torch.export](#limitations-of-torch-export)  ),
but seeing as the context manager does not affect the tensor
computations in the model, we can go with the non-strict mode’s result.

### Export for Training and Inference 

In PyTorch 2.5, we introduced a new API called `export_for_training()`  .
It’s still going through hardening, so if you run into any issues, please file
them to Github with the “oncall: export” tag. 

In this API, we produce the most generic IR that contains all ATen operators
(including both functional and non-functional) which can be used to train in
eager PyTorch Autograd. This API is intended for eager training use cases such as PT2 Quantization
and will soon be the default IR of torch.export.export. To read further about
the motivation behind this change, please refer to [https://dev-discuss.localhost:8000/t/why-pytorch-does-not-need-a-new-standardized-operator-set/2206](https://dev-discuss.localhost:8000/t/why-pytorch-does-not-need-a-new-standardized-operator-set/2206) 

When this API is combined with `run_decompositions()`  , you should be able to get inference IR with
any desired decomposition behavior. 

To show some examples: 

```
class ConvBatchnorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

mod = ConvBatchnorm()
inp = torch.randn(1, 1, 3, 3)

ep_for_training = torch.export.export_for_training(mod, (inp,))
print(ep_for_training)

```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_conv_weight: "f32[3, 1, 1, 1]", p_conv_bias: "f32[3]", p_bn_weight: "f32[3]", p_bn_bias: "f32[3]", b_bn_running_mean: "f32[3]", b_bn_running_var: "f32[3]", b_bn_num_batches_tracked: "i64[]", x: "f32[1, 1, 3, 3]"):
            conv2d: "f32[1, 3, 3, 3]" = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias)
            add_: "i64[]" = torch.ops.aten.add_.Tensor(b_bn_num_batches_tracked, 1)
            batch_norm: "f32[1, 3, 3, 3]" = torch.ops.aten.batch_norm.default(conv2d, p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, True, 0.1, 1e-05, True)
            return (batch_norm,)

```

From the above output, you can see that `export_for_training()`  produces pretty much the same ExportedProgram
as `export()`  except for the operators in the graph. You can see that we captured batch_norm in the most general
form. This op is non-functional and will be lowered to different ops when running inference. 

You can also go from this IR to an inference IR via `run_decompositions()`  with arbitrary customizations. 

```
# Lower to core aten inference IR, but keep conv2d
decomp_table = torch.export.default_decompositions()
del decomp_table[torch.ops.aten.conv2d.default]
ep_for_inference = ep_for_training.run_decompositions(decomp_table)

print(ep_for_inference)

```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_conv_weight: "f32[3, 1, 1, 1]", p_conv_bias: "f32[3]", p_bn_weight: "f32[3]", p_bn_bias: "f32[3]", b_bn_running_mean: "f32[3]", b_bn_running_var: "f32[3]", b_bn_num_batches_tracked: "i64[]", x: "f32[1, 1, 3, 3]"):
            conv2d: "f32[1, 3, 3, 3]" = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias)
            add: "i64[]" = torch.ops.aten.add.Tensor(b_bn_num_batches_tracked, 1)
            _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(conv2d, p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, True, 0.1, 1e-05)
            getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
            getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
            getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4]
            return (getitem_3, getitem_4, add, getitem)

```

Here you can see that we kept `conv2d`  op in the IR while decomposing the rest. Now the IR is a functional IR
containing core aten operators except for `conv2d`  . 

You can do even more customization by directly registering your chosen decomposition behaviors. 

You can do even more customizations by directly registering custom decomp behaviour 

```
# Lower to core aten inference IR, but customize conv2d
decomp_table = torch.export.default_decompositions()

def my_awesome_custom_conv2d_function(x, weight, bias, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1):
    return 2 * torch.ops.aten.convolution(x, weight, bias, stride, padding, dilation, False, [0, 0], groups)

decomp_table[torch.ops.aten.conv2d.default] = my_awesome_conv2d_function
ep_for_inference = ep_for_training.run_decompositions(decomp_table)

print(ep_for_inference)

```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_conv_weight: "f32[3, 1, 1, 1]", p_conv_bias: "f32[3]", p_bn_weight: "f32[3]", p_bn_bias: "f32[3]", b_bn_running_mean: "f32[3]", b_bn_running_var: "f32[3]", b_bn_num_batches_tracked: "i64[]", x: "f32[1, 1, 3, 3]"):
            convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(x, p_conv_weight, p_conv_bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            mul: "f32[1, 3, 3, 3]" = torch.ops.aten.mul.Tensor(convolution, 2)
            add: "i64[]" = torch.ops.aten.add.Tensor(b_bn_num_batches_tracked, 1)
            _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(mul, p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, True, 0.1, 1e-05)
            getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
            getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
            getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];
            return (getitem_3, getitem_4, add, getitem)

```

### Expressing Dynamism 

By default `torch.export`  will trace the program assuming all input shapes are **static** , and specializing the exported program to those dimensions. However,
some dimensions, such as a batch dimension, can be dynamic and vary from run to
run. Such dimensions must be specified by using the `torch.export.Dim()`  API to create them and by passing them into [`torch.export.export()`](#torch.export.export "torch.export.export")  through the `dynamic_shapes`  argument. An example: 

```
import torch
from torch.export import Dim, export

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.ReLU()
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU()
        )
        self.buffer = torch.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)

example_args = (torch.randn(32, 64), torch.randn(32, 128))

# Create a dynamic batch size
batch = Dim("batch")
# Specify that the first dimension of each input is that batch size
dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, dynamic_shapes=dynamic_shapes
)
print(exported_program)

```

```
ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, p_branch1_0_weight: "f32[32, 64]", p_branch1_0_bias: "f32[32]", p_branch2_0_weight: "f32[64, 128]", p_branch2_0_bias: "f32[64]", c_buffer: "f32[32]", x1: "f32[s0, 64]", x2: "f32[s0, 128]"):

         # code: out1 = self.branch1(x1)
        linear: "f32[s0, 32]" = torch.ops.aten.linear.default(x1, p_branch1_0_weight, p_branch1_0_bias)
        relu: "f32[s0, 32]" = torch.ops.aten.relu.default(linear)

         # code: out2 = self.branch2(x2)
        linear_1: "f32[s0, 64]" = torch.ops.aten.linear.default(x2, p_branch2_0_weight, p_branch2_0_bias)
        relu_1: "f32[s0, 64]" = torch.ops.aten.relu.default(linear_1)

         # code: return (out1 + self.buffer, out2)
        add: "f32[s0, 32]" = torch.ops.aten.add.Tensor(relu, c_buffer)
        return (add, relu_1)

Range constraints: {s0: VR[0, int_oo]}

```

Some additional things to note: 

* Through the `torch.export.Dim()`  API and the `dynamic_shapes`  argument, we specified the first
dimension of each input to be dynamic. Looking at the inputs `x1`  and `x2`  , they have a symbolic shape of (s0, 64) and (s0, 128), instead of
the (32, 64) and (32, 128) shaped tensors that we passed in as example inputs. `s0`  is a symbol representing that this dimension can be a range
of values.
* `exported_program.range_constraints`  describes the ranges of each symbol
appearing in the graph. In this case, we see that `s0`  has the range
[0, int_oo]. For technical reasons that are difficult to explain here, they are
assumed to be not 0 or 1. This is not a bug, and does not necessarily mean
that the exported program will not work for dimensions 0 or 1. See [The 0/1 Specialization Problem](https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk)  for an in-depth discussion of this topic.

We can also specify more expressive relationships between input shapes, such as
where a pair of shapes might differ by one, a shape might be double of
another, or a shape is even. An example: 

```
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y[1:]

x, y = torch.randn(5), torch.randn(6)
dimx = torch.export.Dim("dimx", min=3, max=6)
dimy = dimx + 1

exported_program = torch.export.export(
    M(), (x, y), dynamic_shapes=({0: dimx}, {0: dimy}),
)
print(exported_program)

```

```
ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s0]", y: "f32[s0 + 1]"):
        # code: return x + y[1:]
        slice_1: "f32[s0]" = torch.ops.aten.slice.Tensor(y, 0, 1, 9223372036854775807)
        add: "f32[s0]" = torch.ops.aten.add.Tensor(x, slice_1)
        return (add,)

Range constraints: {s0: VR[3, 6], s0 + 1: VR[4, 7]}

```

Some things to note: 

* By specifying `{0: dimx}`  for the first input, we see that the resulting
shape of the first input is now dynamic, being `[s0]`  . And now by specifying `{0: dimy}`  for the second input, we see that the resulting shape of the
second input is also dynamic. However, because we expressed `dimy = dimx + 1`  ,
instead of `y`  ’s shape containing a new symbol, we see that it is
now being represented with the same symbol used in `x`  , `s0`  . We can
see that relationship of `dimy = dimx + 1`  is being shown through `s0 + 1`  .
* Looking at the range constraints, we see that `s0`  has the range [3, 6],
which is specified initially, and we can see that `s0 + 1`  has the solved
range of [4, 7].

### Serialization 

To save the `ExportedProgram`  , users can use the [`torch.export.save()`](#torch.export.save "torch.export.save")  and [`torch.export.load()`](#torch.export.load "torch.export.load")  APIs. A convention is to save the `ExportedProgram`  using a `.pt2`  file extension. 

An example: 

```
import torch
import io

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10

exported_program = torch.export.export(MyModule(), torch.randn(5))

torch.export.save(exported_program, 'exported_program.pt2')
saved_exported_program = torch.export.load('exported_program.pt2')

```

### Specializations 

A key concept in understanding the behavior of `torch.export`  is the
difference between *static*  and *dynamic*  values. 

A *dynamic*  value is one that can change from run to run. These behave like
normal arguments to a Python function—you can pass different values for an
argument and expect your function to do the right thing. Tensor *data*  is
treated as dynamic. 

A *static*  value is a value that is fixed at export time and cannot change
between executions of the exported program. When the value is encountered during
tracing, the exporter will treat it as a constant and hard-code it into the
graph. 

When an operation is performed (e.g. `x + y`  ) and all inputs are static, then
the output of the operation will be directly hard-coded into the graph, and the
operation won’t show up (i.e. it will get constant-folded). 

When a value has been hard-coded into the graph, we say that the graph has been *specialized*  to that value. 

The following values are static: 

#### Input Tensor Shapes 

By default, `torch.export`  will trace the program specializing on the input
tensors’ shapes, unless a dimension is specified as dynamic via the `dynamic_shapes`  argument to `torch.export`  . This means that if there exists
shape-dependent control flow, `torch.export`  will specialize on the branch
that is being taken with the given sample inputs. For example: 

```
import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x):
        if x.shape[0] > 5:
            return x + 1
        else:
            return x - 1

example_inputs = (torch.rand(10, 2),)
exported_program = export(Mod(), example_inputs)
print(exported_program)

```

```
ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[10, 2]"):
        # code: return x + 1
        add: "f32[10, 2]" = torch.ops.aten.add.Tensor(x, 1)
        return (add,)

```

The conditional of ( `x.shape[0] > 5`  ) does not appear in the `ExportedProgram`  because the example inputs have the static
shape of (10, 2). Since `torch.export`  specializes on the inputs’ static
shapes, the else branch ( `x - 1`  ) will never be reached. To preserve the dynamic
branching behavior based on the shape of a tensor in the traced graph, `torch.export.Dim()`  will need to be used to specify the dimension
of the input tensor ( `x.shape[0]`  ) to be dynamic, and the source code will
need to be [rewritten](#data-shape-dependent-control-flow)  . 

Note that tensors that are part of the module state (e.g. parameters and
buffers) always have static shapes.

#### Python Primitives 

`torch.export`  also specializes on Python primitives,
such as `int`  , `float`  , `bool`  , and `str`  . However they do have dynamic
variants such as `SymInt`  , `SymFloat`  , and `SymBool`  . 

For example: 

```
import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, const: int, times: int):
        for i in range(times):
            x = x + const
        return x

example_inputs = (torch.rand(2, 2), 1, 3)
exported_program = export(Mod(), example_inputs)
print(exported_program)

```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[2, 2]", const, times):
            # code: x = x + const
            add: "f32[2, 2]" = torch.ops.aten.add.Tensor(x, 1)
            add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 1)
            add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 1)
            return (add_2,)

```

Because integers are specialized, the `torch.ops.aten.add.Tensor`  operations
are all computed with the hard-coded constant `1`  , rather than `const`  . If
a user passes a different value for `const`  at runtime, like 2, than the one used
during export time, 1, this will result in an error.
Additionally, the `times`  iterator used in the `for`  loop is also “inlined”
in the graph through the 3 repeated `torch.ops.aten.add.Tensor`  calls, and the
input `times`  is never used.

#### Python Containers 

Python containers ( `List`  , `Dict`  , `NamedTuple`  , etc.) are considered to
have static structure.

Limitations of torch.export 
------------------------------------------------------------------------------------------

### Graph Breaks 

As `torch.export`  is a one-shot process for capturing a computation graph from
a PyTorch program, it might ultimately run into untraceable parts of programs as
it is nearly impossible to support tracing all PyTorch and Python features. In
the case of `torch.compile`  , an unsupported operation will cause a “graph
break” and the unsupported operation will be run with default Python evaluation.
In contrast, `torch.export`  will require users to provide additional
information or rewrite parts of their code to make it traceable. As the
tracing is based on TorchDynamo, which evaluates at the Python
bytecode level, there will be significantly fewer rewrites required compared to
previous tracing frameworks. 

When a graph break is encountered, [ExportDB](generated/exportdb/index.html#torch-export-db)  is a great
resource for learning about the kinds of programs that are supported and
unsupported, along with ways to rewrite programs to make them traceable. 

An option to get past dealing with this graph breaks is by using [non-strict export](#non-strict-export)

### Data/Shape-Dependent Control Flow 

Graph breaks can also be encountered on data-dependent control flow ( `if x.shape[0] > 2`  ) when shapes are not being specialized, as a tracing compiler cannot
possibly deal with without generating code for a combinatorially exploding
number of paths. In such cases, users will need to rewrite their code using
special control flow operators. Currently, we support [torch.cond](cond.html#cond)  to express if-else like control flow (more coming soon!).

### Missing Fake/Meta/Abstract Kernels for Operators 

When tracing, a FakeTensor kernel (aka meta kernel, abstract impl) is
required for all operators. This is used to reason about the input/output shapes
for this operator. 

Please see [`torch.library.register_fake()`](library.html#torch.library.register_fake "torch.library.register_fake")  for more details. 

In the unfortunate case where your model uses an ATen operator that is does not
have a FakeTensor kernel implementation yet, please file an issue.

Read More 
------------------------------------------------------

Additional Links for Export Users 

* [torch.export Programming Model](export.programming_model.html)
* [torch.export IR Specification](export.ir_spec.html)
* [Draft Export](draft_export.html)
* [Writing Graph Transformations on ATen IR](torch.compiler_transformations.html)
* [IRs](torch.compiler_ir.html)
* [ExportDB](generated/exportdb/index.html)
* [Control Flow - Cond](cond.html)

Deep Dive for PyTorch Developers 

* [Dynamo Overview](torch.compiler_dynamo_overview.html)
* [Dynamo Deep-Dive](torch.compiler_dynamo_deepdive.html)
* [Dynamic Shapes](torch.compiler_dynamic_shapes.html)
* [Fake tensor](torch.compiler_fake_tensor.html)

API Reference 
--------------------------------------------------------------------

torch.export. export ( *mod*  , *args*  , *kwargs = None*  , *** , *dynamic_shapes = None*  , *strict = False*  , *preserve_module_call_signature = ()* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/__init__.py#L175) 
:   [`export()`](#torch.export.export "torch.export.export")  takes any nn.Module along with example inputs, and produces a traced graph representing
only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion,
which can subsequently be executed with different inputs or serialized. The
traced graph (1) produces normalized operators in the functional ATen operator set
(as well as any user-specified custom operators), (2) has eliminated all Python control
flow and data structures (with certain exceptions), and (3) records the set of
shape constraints needed to show that this normalization and control-flow elimination
is sound for future inputs. 

**Soundness Guarantee** 

While tracing, [`export()`](#torch.export.export "torch.export.export")  takes note of shape-related assumptions
made by the user program and the underlying PyTorch operator kernels.
The output [`ExportedProgram`](#torch.export.ExportedProgram "torch.export.ExportedProgram")  is considered valid only when these
assumptions hold true. 

Tracing makes assumptions on the shapes (not values) of input tensors.
Such assumptions must be validated at graph capture time for [`export()`](#torch.export.export "torch.export.export")  to succeed. Specifically: 

* Assumptions on static shapes of input tensors are automatically validated without additional effort.
* Assumptions on dynamic shape of input tensors require explicit specification
by using the `Dim()`  API to construct dynamic dimensions and by associating
them with example inputs through the `dynamic_shapes`  argument.

If any assumption can not be validated, a fatal error will be raised. When that happens,
the error message will include suggested fixes to the specification that are needed
to validate the assumptions. For example [`export()`](#torch.export.export "torch.export.export")  might suggest the
following fix to the definition of a dynamic dimension `dim0_x`  , say appearing in the
shape associated with input `x`  , that was previously defined as `Dim("dim0_x")`  : 

```
dim = Dim("dim0_x", max=5)

```

This example means the generated code requires dimension 0 of input `x`  to be less
than or equal to 5 to be valid. You can inspect the suggested fixes to dynamic dimension
definitions and then copy them verbatim into your code without needing to change the `dynamic_shapes`  argument to your [`export()`](#torch.export.export "torch.export.export")  call. 

Parameters
:   * **mod** ( [*Module*](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")  ) – We will trace the forward method of this module.
* **args** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *,* *...* *]*  ) – Example positional inputs.
* **kwargs** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]* *]*  ) – Optional example keyword inputs.
* **dynamic_shapes** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]* *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]* *,* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)") *[* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]* *]* *]*  ) –

    An optional argument where the type should either be:
        1) a dict from argument names of `f`  to their dynamic shape specifications,
        2) a tuple that specifies dynamic shape specifications for each input in original order.
        If you are specifying dynamism on keyword args, you will need to pass them in the order that
        is defined in the original function signature.

    The dynamic shape of a tensor argument can be specified as either
        (1) a dict from dynamic dimension indices to `Dim()`  types, where it is
        not required to include static dimension indices in this dict, but when they are,
        they should be mapped to None; or (2) a tuple / list of `Dim()`  types or None,
        where the `Dim()`  types correspond to dynamic dimensions, and static dimensions
        are denoted by None. Arguments that are dicts or tuples / lists of tensors are
        recursively specified by using mappings or sequences of contained specifications.

* **strict** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – When disabled (default), the export function will trace the program through
Python runtime, which by itself will not validate some of the implicit assumptions
baked into the graph. It will still validate most critical assumptions like shape
safety. When enabled (by setting `strict=True`  ), the export function will trace
the program through TorchDynamo which will ensure the soundness of the resulting
graph. TorchDynamo has limited Python feature coverage, thus you may experience more
errors. Note that toggling this argument does not affect the resulting IR spec to be
different and the model will be serialized in the same way regardless of what value
is passed here.
* **preserve_module_call_signature** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *...* *]*  ) – A list of submodule paths for which the original
calling conventions are preserved as metadata. The metadata will be used when calling
torch.export.unflatten to preserve the original calling conventions of modules.

Returns
:   An [`ExportedProgram`](#torch.export.ExportedProgram "torch.export.ExportedProgram")  containing the traced callable.

Return type
:   [*ExportedProgram*](#torch.export.ExportedProgram "torch.export.exported_program.ExportedProgram")

**Acceptable input/output types** 

Acceptable types of inputs (for `args`  and `kwargs`  ) and outputs include: 

* Primitive types, i.e. `torch.Tensor`  , `int`  , `float`  , `bool`  and `str`  .
* Dataclasses, but they must be registered by calling [`register_dataclass()`](#torch.export.register_dataclass "torch.export.register_dataclass")  first.
* (Nested) Data structures comprising of `dict`  , `list`  , `tuple`  , `namedtuple`  and `OrderedDict`  containing all above types.

torch.export. save ( *ep*  , *f*  , *** , *extra_files = None*  , *opset_version = None*  , *pickle_protocol = 2* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/__init__.py#L325) 
:   Warning 

Under active development, saved files may not be usable in newer versions
of PyTorch.

Saves an [`ExportedProgram`](#torch.export.ExportedProgram "torch.export.ExportedProgram")  to a file-like object. It can then be
loaded using the Python API [`torch.export.load`](#torch.export.load "torch.export.load")  . 

Parameters
:   * **ep** ( [*ExportedProgram*](#torch.export.ExportedProgram "torch.export.ExportedProgram")  ) – The exported program to save.
* **f** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *|* [*os.PathLike*](https://docs.python.org/3/library/os.html#os.PathLike "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]* *|* *IO* *[* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.13)") *]*  ) – implement write and flush) or a string containing a file name.
* **extra_files** ( *Optional* *[* *Dict* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *Any* *]* *]*  ) – Map from filename to contents
which will be stored as part of f.
* **opset_version** ( *Optional* *[* *Dict* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – A map of opset names
to the version of this opset
* **pickle_protocol** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – can be specified to override the default protocol

Example: 

```
import torch
import io

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10

ep = torch.export.export(MyModule(), (torch.randn(5),))

# Save to file
torch.export.save(ep, "exported_program.pt2")

# Save to io.BytesIO buffer
buffer = io.BytesIO()
torch.export.save(ep, buffer)

# Save with extra files
extra_files = {"foo.txt": b"bar".decode("utf-8")}
torch.export.save(ep, "exported_program.pt2", extra_files=extra_files)

```

torch.export. load ( *f*  , *** , *extra_files = None*  , *expected_opset_version = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/__init__.py#L397) 
:   Warning 

Under active development, saved files may not be usable in newer versions
of PyTorch.

Loads an [`ExportedProgram`](#torch.export.ExportedProgram "torch.export.ExportedProgram")  previously saved with [`torch.export.save`](#torch.export.save "torch.export.save")  . 

Parameters
:   * **f** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *|* [*os.PathLike*](https://docs.python.org/3/library/os.html#os.PathLike "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]* *|* *IO* *[* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.13)") *]*  ) – A file-like object (has to
implement write and flush) or a string containing a file name.
* **extra_files** ( *Optional* *[* *Dict* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *Any* *]* *]*  ) – The extra filenames given in
this map would be loaded and their content would be stored in the
provided map.
* **expected_opset_version** ( *Optional* *[* *Dict* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – A map of opset names
to expected opset versions

Returns
:   An [`ExportedProgram`](#torch.export.ExportedProgram "torch.export.ExportedProgram")  object

Return type
:   [*ExportedProgram*](#torch.export.ExportedProgram "torch.export.exported_program.ExportedProgram")

Example: 

```
import torch
import io

# Load ExportedProgram from file
ep = torch.export.load("exported_program.pt2")

# Load ExportedProgram from io.BytesIO object
with open("exported_program.pt2", "rb") as f:
    buffer = io.BytesIO(f.read())
buffer.seek(0)
ep = torch.export.load(buffer)

# Load with extra files.
extra_files = {"foo.txt": ""}  # values will be replaced with data
ep = torch.export.load("exported_program.pt2", extra_files=extra_files)
print(extra_files["foo.txt"])
print(ep(torch.randn(5)))

```

torch.export. draft_export ( *mod*  , *args*  , *kwargs = None*  , *** , *dynamic_shapes = None*  , *preserve_module_call_signature = ()*  , *strict = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/__init__.py#L534) 
:   A version of torch.export.export which is designed to consistently produce
an ExportedProgram, even if there are potential soundness issues, and to
generate a report listing the issues found. 

Return type
:   [*ExportedProgram*](#torch.export.ExportedProgram "torch.export.exported_program.ExportedProgram")

torch.export. register_dataclass ( *cls*  , *** , *serialized_type_name = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/__init__.py#L560) 
:   Registers a dataclass as a valid input/output type for [`torch.export.export()`](#torch.export.export "torch.export.export")  . 

Parameters
:   * **cls** ( [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.13)") *[* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]*  ) – the dataclass type to register
* **serialized_type_name** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]*  ) – The serialized name for the dataclass. This is
* **this** ( *required if you want to serialize the pytree TreeSpec containing*  ) –
* **dataclass.** –

Example: 

```
import torch
from dataclasses import dataclass

@dataclass
class InputDataClass:
    feature: torch.Tensor
    bias: int

@dataclass
class OutputDataClass:
    res: torch.Tensor

torch.export.register_dataclass(InputDataClass)
torch.export.register_dataclass(OutputDataClass)

class Mod(torch.nn.Module):
    def forward(self, x: InputDataClass) -> OutputDataClass:
        res = x.feature + x.bias
        return OutputDataClass(res=res)

ep = torch.export.export(Mod(), (InputDataClass(torch.ones(2, 2), 1),))
print(ep)

```

*class* torch.export.dynamic_shapes. Dim ( *name*  , *** , *min = None*  , *max = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/dynamic_shapes.py#L86) 
:   The *Dim* class allows users to specify dynamism in their exported programs. By marking a dimension with a *Dim* ,
the compiler associates the dimension with a symbolic integer containing a dynamic range. 

The API can be used in 2 ways: Dim hints (i.e. automatic dynamic shapes: *Dim.AUTO* , *Dim.DYNAMIC* , *Dim.STATIC* ),
or named Dims (i.e. *Dim(“name”, min=1, max=2)* ). 

Dim hints provide the lowest barrier to exportability, with the user only needing to specify if a dimension
if dynamic, static, or left for the compiler to decide ( *Dim.AUTO* ). The export process will automatically
infer the remaining constraints on min/max ranges and relationships between dimensions. 

Example: 

```
class Foo(nn.Module):
    def forward(self, x, y):
        assert x.shape[0] == 4
        assert y.shape[0] >= 16
        return x @ y

x = torch.randn(4, 8)
y = torch.randn(8, 16)
dynamic_shapes = {
    "x": {0: Dim.AUTO, 1: Dim.AUTO},
    "y": {0: Dim.AUTO, 1: Dim.AUTO},
}
ep = torch.export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)

```

Here, export would raise an exception if we replaced all uses of *Dim.AUTO* with *Dim.DYNAMIC* ,
as x.shape[0] is constrained to be static by the model. 

More complex relations between dimensions may also be codegened as runtime assertion nodes by the compiler,
e.g. (x.shape[0] + y.shape[1]) % 4 == 0, to be raised if runtime inputs do not satisfy such constraints. 

You may also specify min-max bounds for Dim hints, e.g. *Dim.AUTO(min=16, max=32)* , *Dim.DYNAMIC(max=64)* ,
with the compiler inferring the remaining constraints within the ranges. An exception will be raised if
the valid range is entirely outside the user-specified range. 

Named Dims provide a stricter way of specifying dynamism, where exceptions are raised if the compiler
infers constraints that do not match the user specification. For example, exporting the previous
model, the user would need the following *dynamic_shapes* argument: 

```
s0 = Dim("s0")
s1 = Dim("s1", min=16)
dynamic_shapes = {
    "x": {0: 4, 1: s0},
    "y": {0: s0, 1: s1},
}
ep = torch.export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)

```

Named Dims also allow specification of relationships between dimensions, up to univariate linear relations.
For example, the following indicates one dimension is a multiple of another plus 4: 

```
s0 = Dim("s0")
s1 = 3 * s0 + 4

```

*class* torch.export.dynamic_shapes. ShapesCollection [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/dynamic_shapes.py#L701) 
:   Builder for dynamic_shapes.
Used to assign dynamic shape specifications to tensors that appear in inputs. 

This is useful particularly when `args()`  is a nested input structure, and it’s
easier to index the input tensors, than to replicate the structure of `args()`  in
the [`dynamic_shapes()`](#torch.export.dynamic_shapes.ShapesCollection.dynamic_shapes "torch.export.dynamic_shapes.ShapesCollection.dynamic_shapes")  specification. 

Example: 

```
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}

dim = torch.export.Dim(...)
dynamic_shapes = torch.export.ShapesCollection()
dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
dynamic_shapes[tensor_y] = {0: dim * 2}
# This is equivalent to the following (now auto-generated):
# dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [{0: dim * 2}, None]}

torch.export(..., args, dynamic_shapes=dynamic_shapes)

```

To specify dynamism for integers, we need to first wrap the integers using
_IntWrapper so that we have a “unique identification tag” for each integer. 

Example: 

```
args = {"x": tensor_x, "others": [int_x, int_y]}
# Wrap all ints with _IntWrapper
mapped_args = pytree.tree_map_only(int, lambda a: _IntWrapper(a), args)

dynamic_shapes = torch.export.ShapesCollection()
dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
dynamic_shapes[mapped_args["others"][0]] = Dim.DYNAMIC

# This is equivalent to the following (now auto-generated):
# dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [Dim.DYNAMIC, None]}

torch.export(..., args, dynamic_shapes=dynamic_shapes)

```

dynamic_shapes ( *m*  , *args*  , *kwargs = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/dynamic_shapes.py#L770) 
:   Generates the [`dynamic_shapes()`](#torch.export.dynamic_shapes.ShapesCollection.dynamic_shapes "torch.export.dynamic_shapes.ShapesCollection.dynamic_shapes")  pytree structure according to `args()`  and `kwargs()`  .

*class* torch.export.dynamic_shapes. AdditionalInputs [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/dynamic_shapes.py#L796) 
:   Infers dynamic_shapes based on additional inputs. 

This is useful particularly for deployment engineers who, on the one hand, may
have access to ample testing or profiling data that can provide a fair sense of
representative inputs for a model, but on the other hand, may not know enough
about the model to guess which input shapes should be dynamic. 

Input shapes that are different than the original are considered dynamic; conversely,
those that are the same as the original are considered static. Moreover, we verify
that the additional inputs are valid for the exported program. This guarantees that
tracing with them instead of the original would have generated the same graph. 

Example: 

```
args0, kwargs0 = ...  # example inputs for export

# other representative inputs that the exported program will run on
dynamic_shapes = torch.export.AdditionalInputs()
dynamic_shapes.add(args1, kwargs1)
...
dynamic_shapes.add(argsN, kwargsN)

torch.export(..., args0, kwargs0, dynamic_shapes=dynamic_shapes)

```

add ( *args*  , *kwargs = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/dynamic_shapes.py#L826) 
:   Additional input `args()`  and `kwargs()`  .

dynamic_shapes ( *m*  , *args*  , *kwargs = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/dynamic_shapes.py#L837) 
:   Infers a [`dynamic_shapes()`](#torch.export.dynamic_shapes.AdditionalInputs.dynamic_shapes "torch.export.dynamic_shapes.AdditionalInputs.dynamic_shapes")  pytree structure by merging shapes of the
original input `args()`  and `kwargs()`  and of each additional input
args and kwargs.

verify ( *ep* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/dynamic_shapes.py#L879) 
:   Verifies that an exported program is valid for each additional input.

torch.export.dynamic_shapes. refine_dynamic_shapes_from_suggested_fixes ( *msg*  , *dynamic_shapes* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/dynamic_shapes.py#L1230) 
:   When exporting with `dynamic_shapes()`  , export may fail with a ConstraintViolation error if the specification
doesn’t match the constraints inferred from tracing the model. The error message may provide suggested fixes -
changes that can be made to `dynamic_shapes()`  to export successfully. 

Example ConstraintViolation error message: 

```
Suggested fixes:

    dim = Dim('dim', min=3, max=6)  # this just refines the dim's range
    dim = 4  # this specializes to a constant
    dy = dx + 1  # dy was specified as an independent dim, but is actually tied to dx with this relation

```

This is a helper function that takes the ConstraintViolation error message and the original `dynamic_shapes()`  spec,
and returns a new `dynamic_shapes()`  spec that incorporates the suggested fixes. 

Example usage: 

```
try:
    ep = export(mod, args, dynamic_shapes=dynamic_shapes)
except torch._dynamo.exc.UserError as exc:
    new_shapes = refine_dynamic_shapes_from_suggested_fixes(
        exc.msg, dynamic_shapes
    )
    ep = export(mod, args, dynamic_shapes=new_shapes)

```

Return type
:   [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ], [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ], [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]]

*class* torch.export. ExportedProgram ( *root*  , *graph*  , *graph_signature*  , *state_dict*  , *range_constraints*  , *module_call_graph*  , *example_inputs = None*  , *constants = None*  , *** , *verifiers = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/exported_program.py#L998) 
:   Package of a program from [`export()`](#torch.export.export "torch.export.export")  . It contains
an [`torch.fx.Graph`](fx.html#torch.fx.Graph "torch.fx.Graph")  that represents Tensor computation, a state_dict containing
tensor values of all lifted parameters and buffers, and various metadata. 

You can call an ExportedProgram like the original callable traced by [`export()`](#torch.export.export "torch.export.export")  with the same calling convention. 

To perform transformations on the graph, use `.module`  property to access
an [`torch.fx.GraphModule`](fx.html#torch.fx.GraphModule "torch.fx.GraphModule")  . You can then use [FX transformation](https://localhost:8000/docs/stable/fx.html#writing-transformations)  to rewrite the graph. Afterwards, you can simply use [`export()`](#torch.export.export "torch.export.export")  again to construct a correct ExportedProgram. 

graph 
:

graph_signature 
:

state_dict 
:

constants 
:

range_constraints 
:

module_call_graph 
:

example_inputs 
:

module ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/exported_program.py#L1399) 
:   Returns a self contained GraphModule with all the parameters/buffers inlined. 

Return type
:   [*Module*](generated/torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")

run_decompositions ( *decomp_table = None*  , *decompose_custom_triton_ops = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/exported_program.py#L1427) 
:   Run a set of decompositions on the exported program and returns a new
exported program. By default we will run the Core ATen decompositions to
get operators in the [Core ATen Operator Set](https://localhost:8000/docs/stable/torch.compiler_ir.html)  . 

For now, we do not decompose joint graphs. 

Parameters
: **decomp_table** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") *[* *torch._ops.OperatorBase* *,* [*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.13)") *]* *]*  ) – An optional argument that specifies decomp behaviour for Aten ops
(1) If None, we decompose to core aten decompositions
(2) If empty, we don’t decompose any operator

Return type
:   [*ExportedProgram*](#torch.export.ExportedProgram "torch.export.exported_program.ExportedProgram")

Some examples: 

If you don’t want to decompose anything 

```
ep = torch.export.export(model, ...)
ep = ep.run_decompositions(decomp_table={})

```

If you want to get a core aten operator set except for certain operator, you can do following: 

```
ep = torch.export.export(model, ...)
decomp_table = torch.export.default_decompositions()
decomp_table[your_op] = your_custom_decomp
ep = ep.run_decompositions(decomp_table=decomp_table)

```

*class* torch.export. ExportGraphSignature ( *input_specs*  , *output_specs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L162) 
:   [`ExportGraphSignature`](#torch.export.ExportGraphSignature "torch.export.ExportGraphSignature")  models the input/output signature of Export Graph,
which is a fx.Graph with stronger invariants gurantees. 

Export Graph is functional and does not access “states” like parameters
or buffers within the graph via `getattr`  nodes. Instead, [`export()`](#torch.export.export "torch.export.export")  gurantees that parameters, buffers, and constant tensors are lifted out of
the graph as inputs. Similarly, any mutations to buffers are not included
in the graph either, instead the updated values of mutated buffers are
modeled as additional outputs of Export Graph. 

The ordering of all inputs and outputs are: 

```
Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
Outputs = [*mutated_inputs, *flattened_user_outputs]

```

e.g. If following module is exported: 

```
class CustomModule(nn.Module):
    def __init__(self) -> None:
        super(CustomModule, self).__init__()

        # Define a parameter
        self.my_parameter = nn.Parameter(torch.tensor(2.0))

        # Define two buffers
        self.register_buffer("my_buffer1", torch.tensor(3.0))
        self.register_buffer("my_buffer2", torch.tensor(4.0))

    def forward(self, x1, x2):
        # Use the parameter, buffers, and both inputs in the forward method
        output = (
            x1 + self.my_parameter
        ) * self.my_buffer1 + x2 * self.my_buffer2

        # Mutate one of the buffers (e.g., increment it by 1)
        self.my_buffer2.add_(1.0)  # In-place addition

        return output

mod = CustomModule()
ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))

```

Resulting Graph is non-functional: 

```
graph():
    %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
    %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
    %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
    %x1 : [num_users=1] = placeholder[target=x1]
    %x2 : [num_users=1] = placeholder[target=x2]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
    %add_ : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
    return (add_1,)

```

Resulting ExportGraphSignature of the non-functional Graph would be: 

```
# inputs
p_my_parameter: PARAMETER target='my_parameter'
b_my_buffer1: BUFFER target='my_buffer1' persistent=True
b_my_buffer2: BUFFER target='my_buffer2' persistent=True
x1: USER_INPUT
x2: USER_INPUT

# outputs
add_1: USER_OUTPUT

```

To get a functional Graph, you can use `run_decompositions()`  : 

```
mod = CustomModule()
ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))
ep = ep.run_decompositions()

```

Resulting Graph is functional: 

```
graph():
    %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
    %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
    %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
    %x1 : [num_users=1] = placeholder[target=x1]
    %x2 : [num_users=1] = placeholder[target=x2]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
    return (add_2, add_1)

```

Resulting ExportGraphSignature of the functional Graph would be: 

```
# inputs
p_my_parameter: PARAMETER target='my_parameter'
b_my_buffer1: BUFFER target='my_buffer1' persistent=True
b_my_buffer2: BUFFER target='my_buffer2' persistent=True
x1: USER_INPUT
x2: USER_INPUT

# outputs
add_2: BUFFER_MUTATION target='my_buffer2'
add_1: USER_OUTPUT

```

*class* torch.export. ModuleCallSignature ( *inputs : [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)") [ [Union](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") [ torch.export.graph_signature.TensorArgument , [torch.export.graph_signature.SymIntArgument](#torch.export.graph_signature.SymIntArgument "torch.export.graph_signature.SymIntArgument") , [torch.export.graph_signature.SymFloatArgument](#torch.export.graph_signature.SymFloatArgument "torch.export.graph_signature.SymFloatArgument") , [torch.export.graph_signature.SymBoolArgument](#torch.export.graph_signature.SymBoolArgument "torch.export.graph_signature.SymBoolArgument") , torch.export.graph_signature.ConstantArgument , [torch.export.graph_signature.CustomObjArgument](#torch.export.graph_signature.CustomObjArgument "torch.export.graph_signature.CustomObjArgument") , torch.export.graph_signature.TokenArgument ] ]*  , *outputs : [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)") [ [Union](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") [ torch.export.graph_signature.TensorArgument , [torch.export.graph_signature.SymIntArgument](#torch.export.graph_signature.SymIntArgument "torch.export.graph_signature.SymIntArgument") , [torch.export.graph_signature.SymFloatArgument](#torch.export.graph_signature.SymFloatArgument "torch.export.graph_signature.SymFloatArgument") , [torch.export.graph_signature.SymBoolArgument](#torch.export.graph_signature.SymBoolArgument "torch.export.graph_signature.SymBoolArgument") , torch.export.graph_signature.ConstantArgument , [torch.export.graph_signature.CustomObjArgument](#torch.export.graph_signature.CustomObjArgument "torch.export.graph_signature.CustomObjArgument") , torch.export.graph_signature.TokenArgument ] ]*  , *in_spec : torch.utils._pytree.TreeSpec*  , *out_spec : torch.utils._pytree.TreeSpec*  , *forward_arg_names : Optional [ [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)") [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") ] ] = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/exported_program.py#L96) 
:

*class* torch.export. ModuleCallEntry ( *fqn : [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")*  , *signature : Optional [ [torch.export.exported_program.ModuleCallSignature](#torch.export.ModuleCallSignature "torch.export.exported_program.ModuleCallSignature") ] = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/exported_program.py#L113) 
:

*class* torch.export.decomp_utils. CustomDecompTable [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/decomp_utils.py#L27) 
:   This is a custom dictionary that is specifically used for handling decomp_table in export.
The reason we need this is because in the new world, you can only *delete*  an op from decomp
table to preserve it. This is problematic for custom ops because we don’t know when the custom
op will actually be loaded to the dispatcher. As a result, we need to record the custom ops operations
until we really need to materialize it (which is when we run decomposition pass.) 

Invariants we hold are:
:   1. All aten decomp is loaded at the init time
2. We materialize ALL ops when user ever reads from the table to make it more likely
that dispatcher picks up the custom op.
3. If it is write operation, we don’t necessarily materialize
4. We load the final time during export, right before calling run_decompositions()

copy ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/decomp_utils.py#L100) 
:   Return type
:   [*CustomDecompTable*](#torch.export.decomp_utils.CustomDecompTable "torch.export.decomp_utils.CustomDecompTable")

items ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/decomp_utils.py#L137) 
:

keys ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/decomp_utils.py#L70) 
:

materialize ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/decomp_utils.py#L141) 
:   Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [torch._ops.OperatorBase, [*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.13)")  ]

pop ( ** args* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/decomp_utils.py#L107) 
:

update ( *other_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/decomp_utils.py#L77) 
:

torch.export.exported_program. default_decompositions ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/exported_program.py#L318) 
:   This is the default decomposition table which contains decomposition of
all ATEN operators to core aten opset. Use this API together with `run_decompositions()` 

Return type
:   [*CustomDecompTable*](#torch.export.decomp_utils.CustomDecompTable "torch.export.decomp_utils.CustomDecompTable")

*class* torch.export.graph_signature. ExportGraphSignature ( *input_specs*  , *output_specs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L162) 
:   [`ExportGraphSignature`](#torch.export.graph_signature.ExportGraphSignature "torch.export.graph_signature.ExportGraphSignature")  models the input/output signature of Export Graph,
which is a fx.Graph with stronger invariants gurantees. 

Export Graph is functional and does not access “states” like parameters
or buffers within the graph via `getattr`  nodes. Instead, `export()`  gurantees that parameters, buffers, and constant tensors are lifted out of
the graph as inputs. Similarly, any mutations to buffers are not included
in the graph either, instead the updated values of mutated buffers are
modeled as additional outputs of Export Graph. 

The ordering of all inputs and outputs are: 

```
Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
Outputs = [*mutated_inputs, *flattened_user_outputs]

```

e.g. If following module is exported: 

```
class CustomModule(nn.Module):
    def __init__(self) -> None:
        super(CustomModule, self).__init__()

        # Define a parameter
        self.my_parameter = nn.Parameter(torch.tensor(2.0))

        # Define two buffers
        self.register_buffer("my_buffer1", torch.tensor(3.0))
        self.register_buffer("my_buffer2", torch.tensor(4.0))

    def forward(self, x1, x2):
        # Use the parameter, buffers, and both inputs in the forward method
        output = (
            x1 + self.my_parameter
        ) * self.my_buffer1 + x2 * self.my_buffer2

        # Mutate one of the buffers (e.g., increment it by 1)
        self.my_buffer2.add_(1.0)  # In-place addition

        return output

mod = CustomModule()
ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))

```

Resulting Graph is non-functional: 

```
graph():
    %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
    %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
    %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
    %x1 : [num_users=1] = placeholder[target=x1]
    %x2 : [num_users=1] = placeholder[target=x2]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
    %add_ : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
    return (add_1,)

```

Resulting ExportGraphSignature of the non-functional Graph would be: 

```
# inputs
p_my_parameter: PARAMETER target='my_parameter'
b_my_buffer1: BUFFER target='my_buffer1' persistent=True
b_my_buffer2: BUFFER target='my_buffer2' persistent=True
x1: USER_INPUT
x2: USER_INPUT

# outputs
add_1: USER_OUTPUT

```

To get a functional Graph, you can use `run_decompositions()`  : 

```
mod = CustomModule()
ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))
ep = ep.run_decompositions()

```

Resulting Graph is functional: 

```
graph():
    %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
    %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
    %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
    %x1 : [num_users=1] = placeholder[target=x1]
    %x2 : [num_users=1] = placeholder[target=x2]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
    return (add_2, add_1)

```

Resulting ExportGraphSignature of the functional Graph would be: 

```
# inputs
p_my_parameter: PARAMETER target='my_parameter'
b_my_buffer1: BUFFER target='my_buffer1' persistent=True
b_my_buffer2: BUFFER target='my_buffer2' persistent=True
x1: USER_INPUT
x2: USER_INPUT

# outputs
add_2: BUFFER_MUTATION target='my_buffer2'
add_1: USER_OUTPUT

```

replace_all_uses ( *old*  , *new* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L504) 
:   Replace all uses of the old name with new name in the signature.

get_replace_hook ( *replace_inputs = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L527) 
:

*class* torch.export.graph_signature. ExportBackwardSignature ( *gradients_to_parameters : [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") , [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") ]*  , *gradients_to_user_inputs : [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") , [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") ]*  , *loss_output : [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L155) 
:

*class* torch.export.graph_signature. InputKind ( *value*  , *names=<not given>*  , **values*  , *module=None*  , *qualname=None*  , *type=None*  , *start=1*  , *boundary=None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L80) 
:

*class* torch.export.graph_signature. InputSpec ( *kind : [torch.export.graph_signature.InputKind](#torch.export.graph_signature.InputKind "torch.export.graph_signature.InputKind")*  , *arg : Union [ torch.export.graph_signature.TensorArgument , [torch.export.graph_signature.SymIntArgument](#torch.export.graph_signature.SymIntArgument "torch.export.graph_signature.SymIntArgument") , [torch.export.graph_signature.SymFloatArgument](#torch.export.graph_signature.SymFloatArgument "torch.export.graph_signature.SymFloatArgument") , [torch.export.graph_signature.SymBoolArgument](#torch.export.graph_signature.SymBoolArgument "torch.export.graph_signature.SymBoolArgument") , torch.export.graph_signature.ConstantArgument , [torch.export.graph_signature.CustomObjArgument](#torch.export.graph_signature.CustomObjArgument "torch.export.graph_signature.CustomObjArgument") , torch.export.graph_signature.TokenArgument ]*  , *target : Optional [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") ]*  , *persistent : Optional [ [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") ] = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L89) 
:

*class* torch.export.graph_signature. OutputKind ( *value*  , *names=<not given>*  , **values*  , *module=None*  , *qualname=None*  , *type=None*  , *start=1*  , *boundary=None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L120) 
:

*class* torch.export.graph_signature. OutputSpec ( *kind : [torch.export.graph_signature.OutputKind](#torch.export.graph_signature.OutputKind "torch.export.graph_signature.OutputKind")*  , *arg : Union [ torch.export.graph_signature.TensorArgument , [torch.export.graph_signature.SymIntArgument](#torch.export.graph_signature.SymIntArgument "torch.export.graph_signature.SymIntArgument") , [torch.export.graph_signature.SymFloatArgument](#torch.export.graph_signature.SymFloatArgument "torch.export.graph_signature.SymFloatArgument") , [torch.export.graph_signature.SymBoolArgument](#torch.export.graph_signature.SymBoolArgument "torch.export.graph_signature.SymBoolArgument") , torch.export.graph_signature.ConstantArgument , [torch.export.graph_signature.CustomObjArgument](#torch.export.graph_signature.CustomObjArgument "torch.export.graph_signature.CustomObjArgument") , torch.export.graph_signature.TokenArgument ]*  , *target : Optional [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") ]* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L130) 
:

*class* torch.export.graph_signature. SymIntArgument ( *name : [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L41) 
:

*class* torch.export.graph_signature. SymBoolArgument ( *name : [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L51) 
:

*class* torch.export.graph_signature. SymFloatArgument ( *name : [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L46) 
:

*class* torch.export.graph_signature. CustomObjArgument ( *name : [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")*  , *class_fqn : [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")*  , *fake_val : Optional [ torch._library.fake_class_registry.FakeScriptObject ] = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/graph_signature.py#L56) 
:

*class* torch.export.unflatten. FlatArgsAdapter [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/unflatten.py#L266) 
:   Adapts input arguments with `input_spec`  to align `target_spec`  . 

*abstract* adapt ( *target_spec*  , *input_spec*  , *input_args*  , *metadata = None*  , *obj = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/unflatten.py#L271) 
:   NOTE: This adapter may mutate given `input_args_with_path`  . 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

*class* torch.export.unflatten. InterpreterModule ( *graph*  , *ty = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/unflatten.py#L130) 
:   A module that uses torch.fx.Interpreter to execute instead of the usual
codegen that GraphModule uses. This provides better stack trace information
and makes it easier to debug execution.

*class* torch.export.unflatten. InterpreterModuleDispatcher ( *attrs*  , *call_modules* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/unflatten.py#L218) 
:   A module that carries a sequence of InterpreterModules corresponding to
a sequence of calls of that module. Each call to the module dispatches
to the next InterpreterModule, and wraps back around after the last.

torch.export.unflatten. unflatten ( *module*  , *flat_args_adapter = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/unflatten.py#L698) 
:   Unflatten an ExportedProgram, producing a module with the same module
hierarchy as the original eager module. This can be useful if you are trying
to use [`torch.export`](#module-torch.export "torch.export")  with another system that expects a module
hierachy instead of the flat graph that [`torch.export`](#module-torch.export "torch.export")  usually produces. 

Note 

The args/kwargs of unflattened modules will not necessarily match
the eager module, so doing a module swap (e.g. `self.submod = new_mod`  ) will not necessarily work. If you need to swap a module out, you
need to set the `preserve_module_call_signature`  parameter of [`torch.export.export()`](#torch.export.export "torch.export.export")  .

Parameters
:   * **module** ( [*ExportedProgram*](#torch.export.ExportedProgram "torch.export.ExportedProgram")  ) – The ExportedProgram to unflatten.
* **flat_args_adapter** ( *Optional* *[* [*FlatArgsAdapter*](#torch.export.unflatten.FlatArgsAdapter "torch.export.unflatten.FlatArgsAdapter") *]*  ) – Adapt flat args if input TreeSpec does not match with exported module’s.

Returns
:   An instance of `UnflattenedModule`  , which has the same module
hierarchy as the original eager module pre-export.

Return type
:   *UnflattenedModule*

torch.export.passes. move_to_device_pass ( *ep*  , *location* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/export/passes/__init__.py#L11) 
:   Move the exported program to the given device. 

Parameters
:   * **ep** ( [*ExportedProgram*](#torch.export.ExportedProgram "torch.export.ExportedProgram")  ) – The exported program to move.
* **location** ( *Union* *[* [*torch.device*](tensor_attributes.html#torch.device "torch.device") *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *Dict* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *]* *]*  ) – The device to move the exported program to.
If a string, it is interpreted as a device name.
If a dict, it is interpreted as a mapping from
the existing device to the intended one

Returns
:   The moved exported program.

Return type
:   [ExportedProgram](#torch.export.ExportedProgram "torch.export.ExportedProgram")

