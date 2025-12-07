torch.signal.windows.bartlett 
==============================================================================================

torch.signal.windows. bartlett ( *M*  , *** , *sym = True*  , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/signal/windows/windows.py#L596) 
:   Computes the Bartlett window. 

The Bartlett window is defined as follows: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             w
            </mi>
<mi>
             n
            </mi>
</msub>
<mo>
            =
           </mo>
<mn>
            1
           </mn>
<mo>
            −
           </mo>
<mrow>
<mo fence="true">
             ∣
            </mo>
<mfrac>
<mrow>
<mn>
               2
              </mn>
<mi>
               n
              </mi>
</mrow>
<mrow>
<mi>
               M
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</mfrac>
<mo>
             −
            </mo>
<mn>
             1
            </mn>
<mo fence="true">
             ∣
            </mo>
</mrow>
<mo>
            =
           </mo>
<mrow>
<mo fence="true">
             {
            </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mfrac>
<mrow>
<mn>
                   2
                  </mn>
<mi>
                   n
                  </mi>
</mrow>
<mrow>
<mi>
                   M
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</mfrac>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                  if
                 </mtext>
<mn>
                  0
                 </mn>
<mo>
                  ≤
                 </mo>
<mi>
                  n
                 </mi>
<mo>
                  ≤
                 </mo>
<mfrac>
<mrow>
<mi>
                    M
                   </mi>
<mo>
                    −
                   </mo>
<mn>
                    1
                   </mn>
</mrow>
<mn>
                   2
                  </mn>
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mn>
                  2
                 </mn>
<mo>
                  −
                 </mo>
<mfrac>
<mrow>
<mn>
                    2
                   </mn>
<mi>
                    n
                   </mi>
</mrow>
<mrow>
<mi>
                    M
                   </mi>
<mo>
                    −
                   </mo>
<mn>
                    1
                   </mn>
</mrow>
</mfrac>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                  if
                 </mtext>
<mfrac>
<mrow>
<mi>
                    M
                   </mi>
<mo>
                    −
                   </mo>
<mn>
                    1
                   </mn>
</mrow>
<mn>
                   2
                  </mn>
</mfrac>
<mo>
                  &lt;
                 </mo>
<mi>
                  n
                 </mi>
<mo>
                  &lt;
                 </mo>
<mi>
                  M
                 </mi>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           w_n = 1 - left| frac{2n}{M - 1} - 1 right| = begin{cases}
    frac{2n}{M - 1} &amp; text{if } 0 leq n leq frac{M - 1}{2} 
    2 - frac{2n}{M - 1} &amp; text{if } frac{M - 1}{2} &lt; n &lt; M  end{cases}
          </annotation>
</semantics>
</math> -->
w n = 1 − ∣ 2 n M − 1 − 1 ∣ = { 2 n M − 1 if 0 ≤ n ≤ M − 1 2 2 − 2 n M − 1 if M − 1 2 < n < M w_n = 1 - left| frac{2n}{M - 1} - 1 right| = begin{cases}
 frac{2n}{M - 1} & text{if } 0 leq n leq frac{M - 1}{2} 
 2 - frac{2n}{M - 1} & text{if } frac{M - 1}{2} < n < M  end{cases}

w n ​ = 1 − ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjIuNDAwZW0iIHZpZXdib3g9IjAgMCAzMzMgMjQwMCIgd2lkdGg9IjAuMzMzZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xNDUgMTUgdjU4NSB2MTIwMCB2NTg1IGMyLjY2NywxMCw5LjY2NywxNSwyMSwxNQpjMTAsMCwxNi42NjcsLTUsMjAsLTE1IHYtNTg1IHYtMTIwMCB2LTU4NSBjLTIuNjY3LC0xMCwtOS42NjcsLTE1LC0yMSwtMTUKYy0xMCwwLC0xNi42NjcsNSwtMjAsMTV6IE0xODggMTUgSDE0NSB2NTg1IHYxMjAwIHY1ODUgaDQzeiI+CjwvcGF0aD4KPC9zdmc+)​ M − 1 2 n ​ − 1 ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjIuNDAwZW0iIHZpZXdib3g9IjAgMCAzMzMgMjQwMCIgd2lkdGg9IjAuMzMzZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xNDUgMTUgdjU4NSB2MTIwMCB2NTg1IGMyLjY2NywxMCw5LjY2NywxNSwyMSwxNQpjMTAsMCwxNi42NjcsLTUsMjAsLTE1IHYtNTg1IHYtMTIwMCB2LTU4NSBjLTIuNjY3LC0xMCwtOS42NjcsLTE1LC0yMSwtMTUKYy0xMCwwLC0xNi42NjcsNSwtMjAsMTV6IE0xODggMTUgSDE0NSB2NTg1IHYxMjAwIHY1ODUgaDQzeiI+CjwvcGF0aD4KPC9zdmc+)​ = { M − 1 2 n ​ 2 − M − 1 2 n ​ ​ if 0 ≤ n ≤ 2 M − 1 ​ if 2 M − 1 ​ < n < M ​

The window is normalized to 1 (maximum value is 1). However, the 1 doesn’t appear if `M`  is even and `sym`  is *True* . 

Parameters
: **M** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the length of the window.
In other words, the number of points of the returned window.

Keyword Arguments
:   * **sym** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If *False* , returns a periodic window suitable for use in spectral analysis.
If *True* , returns a symmetric window suitable for use in filter design. Default: *True* .
* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned Tensor.
Default: `torch.strided`  .
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , uses the current device for the default tensor type
(see [`torch.set_default_device()`](torch.set_default_device.html#torch.set_default_device "torch.set_default_device")  ). `device`  will be the CPU
for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Examples: 

```
>>> # Generates a symmetric Bartlett window.
>>> torch.signal.windows.bartlett(10)
tensor([0.0000, 0.2222, 0.4444, 0.6667, 0.8889, 0.8889, 0.6667, 0.4444, 0.2222, 0.0000])

>>> # Generates a periodic Bartlett window.
>>> torch.signal.windows.bartlett(10, sym=False)
tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 0.8000, 0.6000, 0.4000, 0.2000])

```

