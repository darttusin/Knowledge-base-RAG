torch.istft 
==========================================================

torch. istft ( *input*  , *n_fft*  , *hop_length = None*  , *win_length = None*  , *window = None*  , *center = True*  , *normalized = False*  , *onesided = None*  , *length = None*  , *return_complex = False* ) → Tensor: 
:   Inverse short time Fourier Transform. This is expected to be the inverse of [`stft()`](torch.stft.html#torch.stft "torch.stft")  . 

Warning 

From version 2.1, a warning will be provided if a `window`  is
not specified. In a future release, this attribute will be required.
Please provide the same window used in the stft call.

It has the same parameters (+ additional optional parameter of `length`  ) and it should return the
least squares estimation of the original signal. The algorithm will check using the NOLA condition (
nonzero overlap). 

Important consideration in the parameters `window`  and `center`  so that the envelope
created by the summation of all the windows is never zero at certain point in time. Specifically, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mo>
             ∑
            </mo>
<mrow>
<mi>
              t
             </mi>
<mo>
              =
             </mo>
<mo>
              −
             </mo>
<mi mathvariant="normal">
              ∞
             </mi>
</mrow>
<mi mathvariant="normal">
             ∞
            </mi>
</msubsup>
<mi mathvariant="normal">
            ∣
           </mi>
<mi>
            w
           </mi>
<msup>
<mi mathvariant="normal">
             ∣
            </mi>
<mn>
             2
            </mn>
</msup>
<mo stretchy="false">
            [
           </mo>
<mi>
            n
           </mi>
<mo>
            −
           </mo>
<mi>
            t
           </mi>
<mo>
            ×
           </mo>
<mi>
            h
           </mi>
<mi>
            o
           </mi>
<mi>
            p
           </mi>
<mi mathvariant="normal">
            _
           </mi>
<mi>
            l
           </mi>
<mi>
            e
           </mi>
<mi>
            n
           </mi>
<mi>
            g
           </mi>
<mi>
            t
           </mi>
<mi>
            h
           </mi>
<mo stretchy="false">
            ]
           </mo>
<menclose notation="updiagonalstrike">
<mo lspace="0em" rspace="0em">
             =
            </mo>
</menclose>
<mn>
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           sum_{t=-infty}^{infty} |w|^2[n-ttimes hop_length] cancel{=} 0
          </annotation>
</semantics>
</math> -->∑ t = − ∞ ∞ ∣ w ∣ 2 [ n − t × h o p _ l e n g t h ] = 0 sum_{t=-infty}^{infty} |w|^2[n-ttimes hop_length] cancel{=} 0∑ t = − ∞ ∞ ​ ∣ w ∣ 2 [ n − t × h o p _ l e n g t h ] = ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuNzY2OWVtIiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGxpbmUgc3Ryb2tlLXdpZHRoPSIwLjA0NmVtIiB4MT0iMCIgeDI9IjEwMCUiIHkxPSIxMDAlIiB5Mj0iMCI+CjwvbGluZT4KPC9zdmc+)​ 0  . 

Since [`stft()`](torch.stft.html#torch.stft "torch.stft")  discards elements at the end of the signal if they do not fit in a frame, `istft`  may return a shorter signal than the original signal (can occur if `center`  is False
since the signal isn’t padded). If *length* is given in the arguments and is longer than expected, `istft`  will pad zeros to the end of the returned signal. 

If `center`  is `True`  , then there will be padding e.g. `'constant'`  , `'reflect'`  , etc.
Left padding can be trimmed off exactly because they can be calculated but right padding cannot be
calculated without additional information. 

Example: Suppose the last window is: `[17, 18, 0, 0, 0]`  vs `[18, 0, 0, 0, 0]` 

The `n_fft`  , `hop_length`  , `win_length`  are all the same which prevents the calculation
of right padding. These additional values could be zeros or a reflection of the signal so providing `length`  could be useful. If `length`  is `None`  then padding will be aggressively removed
(some loss of signal). 

[1] D. W. Griffin and J. S. Lim, “Signal estimation from modified short-time Fourier transform,”
IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) –

    The input tensor. Expected to be in the format of [`stft()`](torch.stft.html#torch.stft "torch.stft")  ,
        output. That is a complex tensor of shape *(B?, N, T)* where

    + *B?* is an optional batch dimension
        + *N* is the number of frequency samples, *(n_fft // 2) + 1* for onesided input, or otherwise *n_fft* .
        + *T* is the number of frames, *1 + length // hop_length* for centered stft,
        or *1 + (length - n_fft) // hop_length* otherwise.
    Changed in version 2.0:  Real datatype inputs are no longer supported. Input must now have a
        complex datatype, as returned by `stft(..., return_complex=True)`  .

* **n_fft** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Size of Fourier transform
* **hop_length** ( *Optional* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – The distance between neighboring sliding window frames.
(Default: `n_fft // 4`  )
* **win_length** ( *Optional* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – The size of window frame and STFT filter. (Default: `n_fft`  )
* **window** ( *Optional* *[* [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – The optional window function.
Shape must be 1d and *<= n_fft* (Default: `torch.ones(win_length)`  )
* **center** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether `input`  was padded on both sides so that the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                t
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               t
              </annotation>
</semantics>
</math> -->t tt  -th frame is
centered at time <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                t
               </mi>
<mo>
                ×
               </mo>
<mtext>
                hop_length
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               t times text{hop_length}
              </annotation>
</semantics>
</math> -->t × hop_length t times text{hop_length}t × hop_length  .
(Default: `True`  )

* **normalized** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether the STFT was normalized. (Default: `False`  )
* **onesided** ( *Optional* *[* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *]*  ) – Whether the STFT was onesided.
(Default: `True`  if *n_fft != fft_size* in the input size)
* **length** ( *Optional* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – The amount to trim the signal by (i.e. the
original signal length). Defaults to *(T - 1) * hop_length* for
centered stft, or *n_fft + (T - 1) * hop_length* otherwise, where *T* is the number of input frames.
* **return_complex** ( *Optional* *[* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *]*  ) – Whether the output should be complex, or if the input should be
assumed to derive from a real signal and window.
Note that this is incompatible with `onesided=True`  .
(Default: `False`  )

Returns
:   Least squares estimation of the original signal of shape *(B?, length)* where
:   *B?* is an optional batch dimension from the input tensor.

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

