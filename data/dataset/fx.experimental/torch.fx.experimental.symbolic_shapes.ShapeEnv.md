ShapeEnv 
====================================================

*class* torch.fx.experimental.symbolic_shapes. ShapeEnv ( *** , *should_record_events = None*  , *tracked_fakes = None*  , *** kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L3555) 
:   add_var_to_val ( *expr*  , *val* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L5163) 
:   Adds a new symbol to the symbolic environment.

bind_symbols ( *placeholders*  , *args* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6069) 
:   Given a paired list of placeholders (fake tensors with
symbolic sizes) and concrete arguments (regular tensors
with real sizes), returns a dictionary mapping each
symbol to its real value. So for example, if you
have a placeholder with size (s0, s1), binding
(2, 4) to it will give you {s0: 2, s1: 4}. This is
not guaranteed to bind ALL symbols in the ShapeEnv;
we can’t bind a symbol if it doesn’t occur in any placeholder,
and symbols that already have replacements won’t get bindings. 

This is a little duplicative with evaluate_guards but
it’s different enough that it seemed cleanest to make
another copy. This assumes the guards are already checked,
though if it’s cheap we’ll check for shenanigans 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [sympy.Symbol, [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ]

bound_sympy ( *expr*  , *size_oblivious = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6140) 
:   Given a sympy expression, computes a ValueRanges bound for what values it can be 

Return type
:   *ValueRanges*

check_equal ( *other* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L3946) 
:   Compare another ShapeEnv for equivalence

cleanup ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L7660) 
:   Break reference cycles. 

This destroys the stacks. If you really want to keep them, we
just need some way to break references on code objects.

create_symbol ( *val*  , *source*  , *dynamic_dim = DimDynamic.DUCK*  , *constraint_dim = None*  , *positive = True*  , *do_not_specialize_zero_one = False*  , *symbolic_context = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4906) 
:   Create a new symbol which is tracked by this ShapeEnv 

Return type
:   *Expr*

create_symbolic_sizes_strides_storage_offset ( *ex*  , *source*  , *** , *symbolic_context = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4400) 
:   Returns a list of symbolic sizes and strides for the given tensor.
We try our best to express stride in terms of the sizes, so as to not
introduce new symbolic variables. 

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [IntLikeType, …], [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [IntLikeType, …], IntLikeType]

create_symboolnode ( *sym* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4754) 
:   Create a SymBool object from a sympy boolean expression 

Return type
:   [*SymBool*](../torch.html#torch.SymBool "torch.SymBool")

create_symfloatnode ( *sym*  , *** , *hint*  , *source = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4702) 
:   Create a SymFloat value from a symbolic expression 

Return type
:   FloatLikeType

create_symintnode ( *sym*  , *** , *hint*  , *source = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4658) 
:   Create a SymInt value from a symbolic expression 

If you know what the current hint value of the SymInt to be created
is, pass it into hint. Otherwise, pass None and we will make our best
guess 

Return type
:   IntLikeType

create_unbacked_symbool ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4851) 
:   Create a symbolic boolean without a hint value 

Return type
:   [*SymBool*](../torch.html#torch.SymBool "torch.SymBool")

create_unbacked_symfloat ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4797) 
:   Create a symbolic float without a hint value 

Return type
:   [*SymFloat*](../torch.html#torch.SymFloat "torch.SymFloat")

create_unbacked_symint ( *source = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4822) 
:   Create a symbolic integer without a hint value 

Return type
:   [*SymInt*](../torch.html#torch.SymInt "torch.SymInt")

create_unspecified_symbol ( *val*  , *source*  , *dynamic_dim = DimDynamic.DUCK*  , *constraint_dim = None*  , *symbolic_context = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4876) 
:   Create a symbol with an unspecified value 

Compared to standard symbols we do not assume the value is positive,
nor do we specialze on zero or one values. 

Return type
:   *Expr*

create_unspecified_symint_and_symbol ( *value*  , *source*  , *dynamic_dim* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4739) 
:   Create a SymInt wrapping a new unspecified symbol 

Return type
:   IntLikeType

deserialize_symexpr ( *code* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6025) 
:   To be used by compile_fx to deserialize symexprs 

Return type
:   [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [*SymInt*](../torch.html#torch.SymInt "torch.SymInt")  , [*SymFloat*](../torch.html#torch.SymFloat "torch.SymFloat")  , [*SymBool*](../torch.html#torch.SymBool "torch.SymBool")  ]

evaluate_expr ( *orig_expr*  , *hint = None*  , *fx_node = None*  , *size_oblivious = False*  , *fallback_value = None*  , *** , *forcing_spec = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L7306) 
:   Given an expression, evaluates it, adding guards if necessary
When fallback_value is not None the function return fallback_value instead of failing with data dependent error. 

Return type
:   *Basic*

evaluate_guards_expression ( *code*  , *args* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6035) 
:   Expected to be used with produce_guards_expression(). Evaluates an expression
generated by produce_guards_expression for the given concrete args. 

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

evaluate_guards_for_args ( *placeholders*  , *args*  , *** , *ignore_static = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6043) 
:   Generate guards for a graph’s placeholder values and evaluate the guards with args 

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

evaluate_sym_node ( *sym_node*  , *size_oblivious = False*  , *fallback_value = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L7212) 
:   Given a a SymNode, evaluates sym_node.expr, adding guards if necessary. 

Return type
:   *Basic*

evaluate_symexpr ( *code* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6018) 
:   To be used by compile_fx to evaluate symexprs 

Return type
:   [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  , [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  , [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ]

format_guards ( *verbose = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6132) 
:   Format this shape env’s guard expressions with optional traceback info if verbose 

Return type
:   [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")

freeze ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4174) 
:   Freeze this ShapeEnv to stop accumulating guards 

A frozen ShapeEnv will ignore any further guards generated on it and
only emit a warning which may lead to accuracy problems.

freeze_runtime_asserts ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4183) 
:   Freeze this ShapeEnv to stop adding deferred runtime asserts. 

We will error if you try to install a new runtime assert when it is
frozen. This would indicate a lowering violation, or perhaps something
we know statically is already True but we are checking it again in a way
that is not clearly dischargeable.

get_axioms ( *symbols = None*  , *compute_hint = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6158) 
:   Given the symbols in an expression, it returns all the runtime asserts that have those symbols
concatenated with all the guards.
If symbols is None, it returns all the runtime asserts (and all the guards) 

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [sympy.logic.boolalg.Boolean, …]

get_implications ( *e* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6188) 
:   Given a expression, it returns a list of predicates that follow from it 

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [sympy.logic.boolalg.Boolean, sympy.logic.boolalg.BooleanAtom], …]

get_nontrivial_guards ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6121) 
:   Returns a list of guard expressions that aren’t statically known (i.e. not trivial) 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [sympy.logic.boolalg.Boolean]

get_pruned_guards ( *symints* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6056) 
:   Get a list of guards, but pruned so it only provides guards that
reference symints from the passed in input 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [torch._guards.ShapeGuard]

guard_or_defer_runtime_assert ( *orig_expr*  , *msg*  , *fx_node = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L7673) 
:   Adds a guard that orig_expr is True if we can or fall back to adding an assert
that is checked at runtime. 

Parameters
:   * **orig_expr** ( *sympy.Expr*  ) – Boolean expression to assert is true
* **msg** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – Message to display on assertion failure
* **fx_node** ( *Optional* *,* [*torch.fx.Node*](../fx.html#torch.fx.Node "torch.fx.Node")  ) – node in `self.graph`  corresponding
to the expression, if applicable

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

ignore_fresh_unbacked_symbols ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4162) 
:   Indicates that the newly allocated unbacked SymInts are being
discarded 

Return type
:   [*Iterator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator "(in Python v3.13)")  [None]

is_unbacked_symint ( *symbol* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4847) 
:   Check if a sympy symbol matches the naming convention for unbacked symbols 

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

patch_source_specialization ( *source*  , *check_fn* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L3900) 
:   Temporarily add symbol-level axioms to the ShapeEnv. This is useful when you want to “fork”
and have parallel universes of ShapeEnvs. For example, we use this when doing multi-graph
compile so we can support various graphs with varying levels of specializations. 

This context manager allows for temporarily adding constraints to the shape environment
based on a specialization function applied to a symbol associated with a source. 

Parameters
:   * **source** ( *Source*  ) – The source of the symbol to specialize
* **check_fn** ( [*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.13)") *[* *[* *Symbol* *]* *,* *Expr* *]*  ) – A function that takes a sympy Symbol and returns a sympy expression
representing a constraint/specialization to be applied

Return type
:   [*Iterator*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator "(in Python v3.13)")  [None]

produce_guards ( ** args*  , *** kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L5195) 
:   Like produce_guards_verbose, but only returns the non-verbose python guard expressions
(no verbose guards produced.) 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ]

produce_guards_expression ( *placeholders*  , *** , *guards = None*  , *ignore_static = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L5993) 
:   Expected to be used with evaluate_guards_expression(). Produces the guards
for the given placeholders and returns a string expression to be evaluated
by evaluate_guards_expression given concrete values for the placeholders. 

Return type
:   Optional[ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ]

produce_guards_verbose ( *placeholders*  , *sources*  , *source_ref=<function ShapeEnv.<lambda>>*  , *** , *guards=None*  , *input_contexts=None*  , *equalities_inputs=None*  , *_simplified=False*  , *ignore_static=True*  , *langs=('python'*  , *'verbose_python')* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L5202) 
:   Generates a list of guards strings which, when evaluated in a context that
defines tensors for all the sources, returns True or False depending
on if the guards in the list evaluated to True or not. Primarily used by Dynamo,
but this is also helpful for manual testing of guards (see
evaluate_guards_for_args) 

For convenience in testing, a source is allowed to be a str,
in which case we will assume it is a LocalSource 

simplified lets you omit duck sizing, equality and 0/1 guards.
This is useful for testing when you don’t care about the boilerplate
guards, and it may be helpful for user output too (be careful though;
some equality guards are nontrivial! It would be nice to get simplified
output to print them too). It’s private because it’s not
intended for normal use 

Returns guards in python and python with verbose comments (verbose) by
default. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [_ShapeGuardsHelper]

replace ( *expr* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6317) 
:   Apply symbol replacements to any symbols in the given expression. 

Return type
:   *_SympyT*

set_unbacked_var_to_val ( *k*  , *v* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4048) 
:   Used only when propagate_real_tensors; registers a value for an
unbacked symbol, which can be used last resort to resolve hints.

simplify ( *expr*  , *size_oblivious = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6347) 
:   Use known constraints and replacements to simplify the given expr 

Return type
:   *_SympyT*

size_hint ( *expr*  , *** , *allow_none = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L6423) 
:   Gets a size hint for a given expression from the underlying shapes we had.
Does not introduce a guard, so only use this when you can guarantee that
your code is still valid for arbitrary shapes (such as optimization decisions) 

Return type
:   [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)")  [ *Basic*  ]

suppress_guards ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L4311) 
:   Context manager to ignore all guards generated inside 

Return type
:   *_GeneratorContextManager*  [None]

