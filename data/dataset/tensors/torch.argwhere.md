torch.argwhere 
================================================================

torch. argwhere ( *input* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a tensor containing the indices of all non-zero elements of `input`  . Each row in the result contains the indices of a non-zero
element in `input`  . The result is sorted lexicographically, with
the last index changing the fastest (C-style). 

If `input`  has <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           n
          </annotation>
</semantics>
</math> -->n nn  dimensions, then the resulting indices tensor `out`  is of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            z
           </mi>
<mo>
            ×
           </mo>
<mi>
            n
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (z times n)
          </annotation>
</semantics>
</math> -->( z × n ) (z times n)( z × n )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            z
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           z
          </annotation>
</semantics>
</math> -->z zz  is the total number of
non-zero elements in the `input`  tensor. 

Note 

This function is similar to NumPy’s *argwhere* . 

When `input`  is on CUDA, this function causes host-device synchronization.

Parameters
: **{input}** –

Example: 

```
>>> t = torch.tensor([1, 0, 1])
>>> torch.argwhere(t)
tensor([[0],
        [2]])
>>> t = torch.tensor([[1, 0, 1], [0, 1, 1]])
>>> torch.argwhere(t)
tensor([[0, 0],
        [0, 2],
        [1, 1],
        [1, 2]])

```

