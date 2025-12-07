torch.nn.attention.flex_attention 
==============================================================================================================

torch.nn.attention.flex_attention. flex_attention ( *query*  , *key*  , *value*  , *score_mod = None*  , *block_mask = None*  , *scale = None*  , *enable_gqa = False*  , *return_lse = False*  , *kernel_options = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L1234) 
:   This function implements scaled dot product attention with an arbitrary attention score modification function. 

This function computes the scaled dot product attention between query, key, and value tensors with a user-defined
attention score modification function. The attention score modification function will be applied after the attention
scores have been calculated between the query and key tensors. The attention scores are calculated as follows: 

The `score_mod`  function should have the following signature: 

```
def score_mod(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    q_idx: Tensor,
    k_idx: Tensor
) -> Tensor:

```

Where:
:   * `score`  : A scalar tensor representing the attention score,
with the same data type and device as the query, key, and value tensors.
* `batch`  , `head`  , `q_idx`  , `k_idx`  : Scalar tensors indicating
the batch index, query head index, query index, and key/value index, respectively.
These should have the `torch.int`  data type and be located on the same device as the score tensor.

Parameters
:   * **query** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Query tensor; shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                B
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                H
               </mi>
<mi>
                q
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                L
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                E
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (B, Hq, L, E)
              </annotation>
</semantics>
</math> -->( B , H q , L , E ) (B, Hq, L, E)( B , H q , L , E )  . For FP8 dtypes, should be in row-major memory layout for optimal performance.

* **key** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Key tensor; shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                B
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                H
               </mi>
<mi>
                k
               </mi>
<mi>
                v
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                S
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                E
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (B, Hkv, S, E)
              </annotation>
</semantics>
</math> -->( B , H k v , S , E ) (B, Hkv, S, E)( B , H k v , S , E )  . For FP8 dtypes, should be in row-major memory layout for optimal performance.

* **value** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Value tensor; shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                B
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                H
               </mi>
<mi>
                k
               </mi>
<mi>
                v
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                S
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                E
               </mi>
<mi>
                v
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (B, Hkv, S, Ev)
              </annotation>
</semantics>
</math> -->( B , H k v , S , E v ) (B, Hkv, S, Ev)( B , H k v , S , E v )  . For FP8 dtypes, should be in column-major memory layout for optimal performance.

* **score_mod** ( *Optional* *[* *Callable* *]*  ) – Function to modify attention scores. By default no score_mod is applied.
* **block_mask** ( *Optional* *[* [*BlockMask*](#torch.nn.attention.flex_attention.BlockMask "torch.nn.attention.flex_attention.BlockMask") *]*  ) – BlockMask object that controls the blocksparsity pattern of the attention.
* **scale** ( *Optional* *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]*  ) – Scaling factor applied prior to softmax. If none, the default value is set to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mn>
                 1
                </mn>
<msqrt>
<mi>
                  E
                 </mi>
</msqrt>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
               frac{1}{sqrt{E}}
              </annotation>
</semantics>
</math> -->1 E frac{1}{sqrt{E}}E ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 1 ​  .

* **enable_gqa** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to True, enables Grouped Query Attention (GQA) and broadcasts key/value heads to query heads.
* **return_lse** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to return the logsumexp of the attention scores. Default is False.
* **kernel_options** ( *Optional* *[* *Dict* *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *Any* *]* *]*  ) – Options to pass into the Triton kernels.

Returns
:   Attention output; shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              B
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              H
             </mi>
<mi>
              q
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              L
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              E
             </mi>
<mi>
              v
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (B, Hq, L, Ev)
            </annotation>
</semantics>
</math> -->( B , H q , L , E v ) (B, Hq, L, Ev)( B , H q , L , E v )  .

Return type
:   output ( [Tensor](tensors.html#torch.Tensor "torch.Tensor")  )

Shape legend:
:   * <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                N
               </mi>
<mo>
                :
               </mo>
<mtext>
                Batch size
               </mtext>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mo>
                :
               </mo>
<mtext>
                Any number of other batch dimensions (optional)
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               N: text{Batch size} ... : text{Any number of other batch dimensions (optional)}
              </annotation>
</semantics>
</math> -->N : Batch size . . . : Any number of other batch dimensions (optional) N: text{Batch size} ... : text{Any number of other batch dimensions (optional)}N : Batch size ... : Any number of other batch dimensions (optional)

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                S
               </mi>
<mo>
                :
               </mo>
<mtext>
                Source sequence length
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               S: text{Source sequence length}
              </annotation>
</semantics>
</math> -->S : Source sequence length S: text{Source sequence length}S : Source sequence length

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                L
               </mi>
<mo>
                :
               </mo>
<mtext>
                Target sequence length
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               L: text{Target sequence length}
              </annotation>
</semantics>
</math> -->L : Target sequence length L: text{Target sequence length}L : Target sequence length

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                E
               </mi>
<mo>
                :
               </mo>
<mtext>
                Embedding dimension of the query and key
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               E: text{Embedding dimension of the query and key}
              </annotation>
</semantics>
</math> -->E : Embedding dimension of the query and key E: text{Embedding dimension of the query and key}E : Embedding dimension of the query and key

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                E
               </mi>
<mi>
                v
               </mi>
<mo>
                :
               </mo>
<mtext>
                Embedding dimension of the value
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               Ev: text{Embedding dimension of the value}
              </annotation>
</semantics>
</math> -->E v : Embedding dimension of the value Ev: text{Embedding dimension of the value}E v : Embedding dimension of the value

Warning 

*torch.nn.attention.flex_attention* is a prototype feature in PyTorch.
Please look forward to a more stable implementation in a future version of PyTorch.
Read more about feature classification at: [https://localhost:8000/blog/pytorch-feature-classification-changes/#prototype](https://localhost:8000/blog/pytorch-feature-classification-changes/#prototype)

BlockMask Utilities 
--------------------------------------------------------------------------

torch.nn.attention.flex_attention. create_block_mask ( *mask_mod*  , *B*  , *H*  , *Q_LEN*  , *KV_LEN*  , *device = 'cuda'*  , *BLOCK_SIZE = 128*  , *_compile = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L832) 
:   This function creates a block mask tuple from a mask_mod function. 

Parameters
:   * **mask_mod** ( *Callable*  ) – mask_mod function. This is a callable that defines the
masking pattern for the attention mechanism. It takes four arguments:
b (batch size), h (number of heads), q_idx (query index), and kv_idx (key/value index).
It should return a boolean tensor indicating which attention connections are allowed (True)
or masked out (False).
* **B** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Batch size.
* **H** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of query heads.
* **Q_LEN** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Sequence length of query.
* **KV_LEN** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Sequence length of key/value.
* **device** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – Device to run the mask creation on.
* **BLOCK_SIZE** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – Block size for the block mask. If a single int is provided it is used for both query and key/value.

Returns
:   A BlockMask object that contains the block mask information.

Return type
:   [BlockMask](#torch.nn.attention.flex_attention.BlockMask "torch.nn.attention.flex_attention.BlockMask")

Example Usage:
:   ```
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, 1, 1, 8192, 8192, device="cuda")
query = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
key = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
value = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
output = flex_attention(query, key, value, block_mask=block_mask)

```

torch.nn.attention.flex_attention. create_mask ( *mod_fn*  , *B*  , *H*  , *Q_LEN*  , *KV_LEN*  , *device = 'cuda'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L783) 
:   This function creates a mask tensor from a mod_fn function. 

Parameters
:   * **mod_fn** ( *Union* *[* *_score_mod_signature* *,* *_mask_mod_signature* *]*  ) – Function to modify attention scores.
* **B** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Batch size.
* **H** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of query heads.
* **Q_LEN** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Sequence length of query.
* **KV_LEN** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Sequence length of key/value.
* **device** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – Device to run the mask creation on.

Returns
:   A mask tensor with shape (B, H, M, N).

Return type
:   mask ( [Tensor](tensors.html#torch.Tensor "torch.Tensor")  )

torch.nn.attention.flex_attention. create_nested_block_mask ( *mask_mod*  , *B*  , *H*  , *q_nt*  , *kv_nt = None*  , *BLOCK_SIZE = 128*  , *_compile = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L1010) 
:   This function creates a nested tensor compatible block mask tuple from a mask_mod
function. The returned BlockMask will be on the device specified by the input nested tensor. 

Parameters
:   * **mask_mod** ( *Callable*  ) – mask_mod function. This is a callable that defines the
masking pattern for the attention mechanism. It takes four arguments:
b (batch size), h (number of heads), q_idx (query index), and kv_idx (key/value index).
It should return a boolean tensor indicating which attention connections are allowed
(True) or masked out (False).
* **B** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Batch size.
* **H** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of query heads.
* **q_nt** ( [*torch.Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Jagged layout nested tensor (NJT) that defines the sequence length
structure for query. The block mask will be constructed to operate on a “stacked
sequence” of length `sum(S)`  for sequence length `S`  from the NJT.
* **kv_nt** ( [*torch.Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Jagged layout nested tensor (NJT) that defines the sequence length
structure for key / value, allowing for cross attention. The block mask will be
constructed to operate on a “stacked sequence” of length `sum(S)`  for sequence
length `S`  from the NJT. If this is None, `q_nt`  is used to define the structure
for key / value as well. Default: None
* **BLOCK_SIZE** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – Block size for the block mask. If a single int is
provided it is used for both query and key/value.

Returns
:   A BlockMask object that contains the block mask information.

Return type
:   [BlockMask](#torch.nn.attention.flex_attention.BlockMask "torch.nn.attention.flex_attention.BlockMask")

Example Usage:
:   ```
# shape (B, num_heads, seq_len*, D) where seq_len* varies across the batch
query = torch.nested.nested_tensor(..., layout=torch.jagged)
key = torch.nested.nested_tensor(..., layout=torch.jagged)
value = torch.nested.nested_tensor(..., layout=torch.jagged)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_nested_block_mask(
    causal_mask, 1, 1, query, _compile=True
)
output = flex_attention(query, key, value, block_mask=block_mask)

```

```
# shape (B, num_heads, seq_len*, D) where seq_len* varies across the batch
query = torch.nested.nested_tensor(..., layout=torch.jagged)
key = torch.nested.nested_tensor(..., layout=torch.jagged)
value = torch.nested.nested_tensor(..., layout=torch.jagged)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# cross attention case: pass both query and key/value NJTs
block_mask = create_nested_block_mask(
    causal_mask, 1, 1, query, key, _compile=True
)
output = flex_attention(query, key, value, block_mask=block_mask)

```

torch.nn.attention.flex_attention. and_masks ( ** mask_mods* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L727) 
:   Returns a mask_mod that’s the intersection of provided mask_mods 

Return type
:   [*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.13)")  [[ [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  , [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  , [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  , [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ], [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ]

torch.nn.attention.flex_attention. or_masks ( ** mask_mods* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L713) 
:   Returns a mask_mod that’s the union of provided mask_mods 

Return type
:   [*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.13)")  [[ [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  , [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  , [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  , [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ], [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ]

torch.nn.attention.flex_attention. noop_mask ( *batch*  , *head*  , *token_q*  , *token_kv* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L132) 
:   Returns a noop mask_mod 

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

BlockMask 
------------------------------------------------------

*class* torch.nn.attention.flex_attention. BlockMask ( *seq_lengths*  , *kv_num_blocks*  , *kv_indices*  , *full_kv_num_blocks*  , *full_kv_indices*  , *q_num_blocks*  , *q_indices*  , *full_q_num_blocks*  , *full_q_indices*  , *BLOCK_SIZE*  , *mask_mod* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L204) 
:   BlockMask is our format for representing a block-sparse attention mask.
It is somewhat of a cross in-between BCSR and a non-sparse format. 

**Basics** 

A block-sparse mask means that instead of representing the sparsity of
individual elements in the mask, a KV_BLOCK_SIZE x Q_BLOCK_SIZE block is
considered sparse only if every element within that block is sparse.
This aligns well with hardware, which generally expects to perform
contiguous loads and computation. 

This format is primarily optimized for 1. simplicity, and 2. kernel
efficiency. Notably, it is *not*  optimized for size, as this mask is always
reduced by a factor of KV_BLOCK_SIZE * Q_BLOCK_SIZE. If the size is a
concern, the tensors can be reduced in size by increasing the block size. 

The essentials of our format are: 

num_blocks_in_row: Tensor[ROWS]:
Describes the number of blocks present in each row. 

col_indices: Tensor[ROWS, MAX_BLOCKS_IN_COL]: *col_indices[i]* is the sequence of block positions for row i. The values of
this row after *col_indices[i][num_blocks_in_row[i]]* are undefined. 

For example, to reconstruct the original tensor from this format: 

```
dense_mask = torch.zeros(ROWS, COLS)
for row in range(ROWS):
    for block_idx in range(num_blocks_in_row[row]):
        dense_mask[row, col_indices[row, block_idx]] = 1

```

Notably, this format makes it easier to implement a reduction along the *rows*  of the mask. 

**Details** 

The basics of our format require only kv_num_blocks and kv_indices. But, we
have up to 8 tensors on this object. This represents 4 pairs: 

1. (kv_num_blocks, kv_indices): Used for the forwards pass of attention, as
we reduce along the KV dimension. 

2. [OPTIONAL] (full_kv_num_blocks, full_kv_indices): This is optional and
purely an optimization. As it turns out, applying masking to every block
is quite expensive! If we specifically know which blocks are “full” and
don’t require masking at all, then we can skip applying mask_mod to these
blocks. This requires the user to split out a separate mask_mod from the
score_mod. For causal masks, this is about a 15% speedup. 

3. [GENERATED] (q_num_blocks, q_indices): Required for the backwards pass,
as computing dKV requires iterating along the mask along the Q dimension. These are autogenerated from 1. 

4. [GENERATED] (full_q_num_blocks, full_q_indices): Same as above, but for
the backwards pass. These are autogenerated from 2. 

BLOCK_SIZE *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") [ [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") ]* 
:

as_tuple ( *flatten = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L385) 
:   Returns a tuple of the attributes of the BlockMask. 

Parameters
: **flatten** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If True, it will flatten the tuple of (KV_BLOCK_SIZE, Q_BLOCK_SIZE)

*classmethod* from_kv_blocks ( *kv_num_blocks*  , *kv_indices*  , *full_kv_num_blocks = None*  , *full_kv_indices = None*  , *BLOCK_SIZE = 128*  , *mask_mod = None*  , *seq_lengths = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L316) 
:   Creates a BlockMask instance from key-value block information. 

Parameters
:   * **kv_num_blocks** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Number of kv_blocks in each Q_BLOCK_SIZE row tile.
* **kv_indices** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")  ) – Indices of key-value blocks in each Q_BLOCK_SIZE row tile.
* **full_kv_num_blocks** ( *Optional* *[* [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *]*  ) – Number of full kv_blocks in each Q_BLOCK_SIZE row tile.
* **full_kv_indices** ( *Optional* *[* [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") *]*  ) – Indices of full key-value blocks in each Q_BLOCK_SIZE row tile.
* **BLOCK_SIZE** ( *Union* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – Size of KV_BLOCK_SIZE x Q_BLOCK_SIZE tiles.
* **mask_mod** ( *Optional* *[* *Callable* *]*  ) – Function to modify the mask.

Returns
:   Instance with full Q information generated via _transposed_ordered

Return type
:   [BlockMask](#torch.nn.attention.flex_attention.BlockMask "torch.nn.attention.flex_attention.BlockMask")

Raises
:   * [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.13)")  – If kv_indices has < 2 dimensions.
* [**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError "(in Python v3.13)")  – If only one of full_kv_* args is provided.

full_kv_indices *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](tensors.html#torch.Tensor "torch.Tensor") ]* 
:

full_kv_num_blocks *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](tensors.html#torch.Tensor "torch.Tensor") ]* 
:

full_q_indices *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](tensors.html#torch.Tensor "torch.Tensor") ]* 
:

full_q_num_blocks *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](tensors.html#torch.Tensor "torch.Tensor") ]* 
:

kv_indices *: [Tensor](tensors.html#torch.Tensor "torch.Tensor")* 
:

kv_num_blocks *: [Tensor](tensors.html#torch.Tensor "torch.Tensor")* 
:

mask_mod *: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.13)") [ [ [Tensor](tensors.html#torch.Tensor "torch.Tensor") , [Tensor](tensors.html#torch.Tensor "torch.Tensor") , [Tensor](tensors.html#torch.Tensor "torch.Tensor") , [Tensor](tensors.html#torch.Tensor "torch.Tensor") ] , [Tensor](tensors.html#torch.Tensor "torch.Tensor") ]* 
:

numel ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L536) 
:   Returns the number of elements (not accounting for sparsity) in the mask.

q_indices *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](tensors.html#torch.Tensor "torch.Tensor") ]* 
:

q_num_blocks *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](tensors.html#torch.Tensor "torch.Tensor") ]* 
:

seq_lengths *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") [ [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") ]* 
:

*property* shape 
:

sparsity ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L545) 
:   Computes the percentage of blocks that are sparse (i.e. not computed) 

Return type
:   [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")

to ( *device* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L632) 
:   Moves the BlockMask to the specified device. 

Parameters
: **device** ( [*torch.device*](tensor_attributes.html#torch.device "torch.device") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – The target device to move the BlockMask to.
Can be a torch.device object or a string (e.g., ‘cpu’, ‘cuda:0’).

Returns
:   A new BlockMask instance with all tensor components moved
to the specified device.

Return type
:   [BlockMask](#torch.nn.attention.flex_attention.BlockMask "torch.nn.attention.flex_attention.BlockMask")

Note 

This method does not modify the original BlockMask in-place.
Instead, it returns a new BlockMask instance where invidual tensor attributes
may or may not be moved to the specified device, depending on their
current device placement.

to_dense ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L556) 
:   Returns a dense block that is equivalent to the block mask. 

Return type
:   [*Tensor*](tensors.html#torch.Tensor "torch.Tensor")

to_string ( *grid_size = (20, 20)*  , *limit = 4* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py#L566) 
:   Returns a string representation of the block mask. Quite nifty. 

If grid_size is -1, prints out an uncompressed version. Warning, it can be quite big!

