torch.stft 
========================================================

torch. stft ( *input*  , *n_fft*  , *hop_length = None*  , *win_length = None*  , *window = None*  , *center = True*  , *pad_mode = 'reflect'*  , *normalized = False*  , *onesided = None*  , *return_complex = None*  , *align_to_window = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/functional.py#L557) 
:   Short-time Fourier transform (STFT). 

Warning 

From version 1.8.0, `return_complex`  must always be given
explicitly for real inputs and *return_complex=False* has been
deprecated. Strongly prefer *return_complex=True* as in a future
pytorch release, this function will only return complex tensors. 

Note that [`torch.view_as_real()`](torch.view_as_real.html#torch.view_as_real "torch.view_as_real")  can be used to recover a real
tensor with an extra last dimension for real and imaginary components.

Warning 

From version 2.1, a warning will be provided if a `window`  is
not specified. In a future release, this attribute will be required.
Not providing a window currently defaults to using a rectangular window,
which may result in undesirable artifacts. Consider using tapered windows,
such as [`torch.hann_window()`](torch.hann_window.html#torch.hann_window "torch.hann_window")  .

The STFT computes the Fourier transform of short overlapping windows of the
input. This giving frequency components of the signal as they change over
time. The interface of this function is modeled after (but *not*  a drop-in
replacement for) [librosa](https://librosa.org/doc/latest/generated/librosa.stft.html)  stft function. 

Ignoring the optional batch dimension, this method computes the following
expression: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
           </mi>
<mo stretchy="false">
            [
           </mo>
<mi>
            ω
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            m
           </mi>
<mo stretchy="false">
            ]
           </mo>
<mo>
            =
           </mo>
<munderover>
<mo>
             ∑
            </mo>
<mrow>
<mi>
              k
             </mi>
<mo>
              =
             </mo>
<mn>
              0
             </mn>
</mrow>
<mtext>
             win_length-1
            </mtext>
</munderover>
<mtext>
            window
           </mtext>
<mo stretchy="false">
            [
           </mo>
<mi>
            k
           </mi>
<mo stretchy="false">
            ]
           </mo>
<mtext>
            input
           </mtext>
<mo stretchy="false">
            [
           </mo>
<mi>
            m
           </mi>
<mo>
            ×
           </mo>
<mtext>
            hop_length
           </mtext>
<mo>
            +
           </mo>
<mi>
            k
           </mi>
<mo stretchy="false">
            ]
           </mo>
<mtext>
</mtext>
<mi>
            exp
           </mi>
<mo>
            ⁡
           </mo>
<mrow>
<mo fence="true">
             (
            </mo>
<mo>
             −
            </mo>
<mi>
             j
            </mi>
<mfrac>
<mrow>
<mn>
               2
              </mn>
<mi>
               π
              </mi>
<mo>
               ⋅
              </mo>
<mi>
               ω
              </mi>
<mi>
               k
              </mi>
</mrow>
<mtext>
              n_fft
             </mtext>
</mfrac>
<mo fence="true">
             )
            </mo>
</mrow>
<mo separator="true">
            ,
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           X[omega, m] = sum_{k = 0}^{text{win_length-1}}%
                    text{window}[k] text{input}[m times text{hop_length} + k] %
                    expleft(- j frac{2 pi cdot omega k}{text{n_fft}}right),
          </annotation>
</semantics>
</math> -->
X [ ω , m ] = ∑ k = 0 win_length-1 window [ k ] input [ m × hop_length + k ] exp ⁡ ( − j 2 π ⋅ ω k n_fft ) , X[omega, m] = sum_{k = 0}^{text{win_length-1}}%
 text{window}[k] text{input}[m times text{hop_length} + k] %
 expleft(- j frac{2 pi cdot omega k}{text{n_fft}}right),

X [ ω , m ] = k = 0 ∑ win_length-1 ​ window [ k ] input [ m × hop_length + k ] exp ( − j n_fft 2 π ⋅ ωk ​ ) ,

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            m
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           m
          </annotation>
</semantics>
</math> -->m mm  is the index of the sliding window, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            ω
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           omega
          </annotation>
</semantics>
</math> -->ω omegaω  is
the frequency <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            0
           </mn>
<mo>
            ≤
           </mo>
<mi>
            ω
           </mi>
<mo>
            &lt;
           </mo>
<mtext>
            n_fft
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           0 leq omega &lt; text{n_fft}
          </annotation>
</semantics>
</math> -->0 ≤ ω < n_fft 0 leq omega < text{n_fft}0 ≤ ω < n_fft  for `onesided=False`  ,
or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            0
           </mn>
<mo>
            ≤
           </mo>
<mi>
            ω
           </mi>
<mo>
            &lt;
           </mo>
<mo stretchy="false">
            ⌊
           </mo>
<mtext>
            n_fft
           </mtext>
<mi mathvariant="normal">
            /
           </mi>
<mn>
            2
           </mn>
<mo stretchy="false">
            ⌋
           </mo>
<mo>
            +
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           0 leq omega &lt; lfloor text{n_fft} / 2 rfloor + 1
          </annotation>
</semantics>
</math> -->0 ≤ ω < ⌊ n_fft / 2 ⌋ + 1 0 leq omega < lfloor text{n_fft} / 2 rfloor + 10 ≤ ω < ⌊ n_fft /2 ⌋ + 1  for `onesided=True`  . 

* `input`  must be either a 1-D time sequence or a 2-D batch of time
sequences.
* If `hop_length`  is `None`  (default), it is treated as equal to `floor(n_fft / 4)`  .
* If `win_length`  is `None`  (default), it is treated as equal to `n_fft`  .
* `window`  can be a 1-D tensor of size `win_length`  , e.g., from [`torch.hann_window()`](torch.hann_window.html#torch.hann_window "torch.hann_window")  . If `window`  is `None`  (default), it is
treated as if having <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
              1
             </mn>
</mrow>
<annotation encoding="application/x-tex">
             1
            </annotation>
</semantics>
</math> -->1 11  everywhere in the window. If <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              win_length
             </mtext>
<mo>
              &lt;
             </mo>
<mtext>
              n_fft
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             text{win_length} &lt; text{n_fft}
            </annotation>
</semantics>
</math> -->win_length < n_fft text{win_length} < text{n_fft}win_length < n_fft  , `window`  will be padded on
both sides to length `n_fft`  before being applied.

* If `center`  is `True`  (default), `input`  will be padded on
both sides so that the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->t tt  -th frame is centered at time <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->t × hop_length t times text{hop_length}t × hop_length  . Otherwise, the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->t tt  -th frame
begins at time <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

* `pad_mode`  determines the padding method used on `input`  when `center`  is `True`  . See [`torch.nn.functional.pad()`](torch.nn.functional.pad.html#torch.nn.functional.pad "torch.nn.functional.pad")  for
all available options. Default is `"reflect"`  .
* If `onesided`  is `True`  (default for real input), only values for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              ω
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             omega
            </annotation>
</semantics>
</math> -->ω omegaω  in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo fence="true">
              [
             </mo>
<mn>
              0
             </mn>
<mo separator="true">
              ,
             </mo>
<mn>
              1
             </mn>
<mo separator="true">
              ,
             </mo>
<mn>
              2
             </mn>
<mo separator="true">
              ,
             </mo>
<mo>
              …
             </mo>
<mo separator="true">
              ,
             </mo>
<mrow>
<mo fence="true">
               ⌊
              </mo>
<mfrac>
<mtext>
                n_fft
               </mtext>
<mn>
                2
               </mn>
</mfrac>
<mo fence="true">
               ⌋
              </mo>
</mrow>
<mo>
              +
             </mo>
<mn>
              1
             </mn>
<mo fence="true">
              ]
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             left[0, 1, 2, dots, leftlfloor
frac{text{n_fft}}{2} rightrfloor + 1right]
            </annotation>
</semantics>
</math> -->[ 0 , 1 , 2 , … , ⌊ n_fft 2 ⌋ + 1 ] left[0, 1, 2, dots, leftlfloor
frac{text{n_fft}}{2} rightrfloor + 1right][ 0 , 1 , 2 , … , ⌊ 2 n_fft ​ ⌋ + 1 ]  are returned because
the real-to-complex Fourier transform satisfies the conjugate symmetry,
i.e., <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              X
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              m
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              ω
             </mi>
<mo stretchy="false">
              ]
             </mo>
<mo>
              =
             </mo>
<mi>
              X
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              m
             </mi>
<mo separator="true">
              ,
             </mo>
<mtext>
              n_fft
             </mtext>
<mo>
              −
             </mo>
<mi>
              ω
             </mi>
<msup>
<mo stretchy="false">
               ]
              </mo>
<mo>
               ∗
              </mo>
</msup>
</mrow>
<annotation encoding="application/x-tex">
             X[m, omega] = X[m, text{n_fft} - omega]^*
            </annotation>
</semantics>
</math> -->X [ m , ω ] = X [ m , n_fft − ω ] ∗ X[m, omega] = X[m, text{n_fft} - omega]^*X [ m , ω ] = X [ m , n_fft − ω ] ∗  .
Note if the input or window tensors are complex, then `onesided`  output is not possible.

* If `normalized`  is `True`  (default is `False`  ), the function
returns the normalized STFT results, i.e., multiplied by <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mtext>
              frame_length
             </mtext>
<msup>
<mo stretchy="false">
               )
              </mo>
<mrow>
<mo>
                −
               </mo>
<mn>
                0.5
               </mn>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
             (text{frame_length})^{-0.5}
            </annotation>
</semantics>
</math> -->( frame_length ) − 0.5 (text{frame_length})^{-0.5}( frame_length ) − 0.5  .

* If `return_complex`  is `True`  (default if input is complex), the
return is a `input.dim() + 1`  dimensional complex tensor. If `False`  ,
the output is a `input.dim() + 2`  dimensional real tensor where the last
dimension represents the real and imaginary components.

Returns either a complex tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mo>
            ∗
           </mo>
<mo>
            ×
           </mo>
<mi>
            N
           </mi>
<mo>
            ×
           </mo>
<mi>
            T
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (* times N times T)
          </annotation>
</semantics>
</math> -->( ∗ × N × T ) (* times N times T)( ∗ × N × T )  if `return_complex`  is true, or a real tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mo>
            ∗
           </mo>
<mo>
            ×
           </mo>
<mi>
            N
           </mi>
<mo>
            ×
           </mo>
<mi>
            T
           </mi>
<mo>
            ×
           </mo>
<mn>
            2
           </mn>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (* times N
times T times 2)
          </annotation>
</semantics>
</math> -->( ∗ × N × T × 2 ) (* times N
times T times 2)( ∗ × N × T × 2 )  . Where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ∗
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           *
          </annotation>
</semantics>
</math> -->∗ *∗  is the optional batch size of `input`  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            N
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           N
          </annotation>
</semantics>
</math> -->N NN  is the number of frequencies where STFT is applied
and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            T
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           T
          </annotation>
</semantics>
</math> -->T TT  is the total number of frames used. 

Warning 

This function changed signature at version 0.4.1. Calling with the
previous signature may cause error or return incorrect result.

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor of shape *(B?, L)* where *B?* is an optional
batch dimension
* **n_fft** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – size of Fourier transform
* **hop_length** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the distance between neighboring sliding window
frames. Default: `None`  (treated as equal to `floor(n_fft / 4)`  )
* **win_length** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the size of window frame and STFT filter.
Default: `None`  (treated as equal to `n_fft`  )
* **window** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the optional window function.
Shape must be 1d and *<= n_fft* Default: `None`  (treated as window of all <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               1
              </annotation>
</semantics>
</math> -->1 11  s)

* **center** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to pad `input`  on both sides so
that the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->t tt  -th frame is centered at time <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
Default: `True`

* **pad_mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – controls the padding method used when `center`  is `True`  . Default: `"reflect"`
* **normalized** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – controls whether to return the normalized STFT results
Default: `False`
* **onesided** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – controls whether to return half of results to
avoid redundancy for real inputs.
Default: `True`  for real `input`  and `window`  , `False`  otherwise.
* **return_complex** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) –

    whether to return a complex tensor, or
        a real tensor with an extra last dimension for the real and
        imaginary components.

    Changed in version 2.0: `return_complex`  is now a required argument for real inputs,
        as the default is being transitioned to `True`  .

    Deprecated since version 2.0: `return_complex=False`  is deprecated, instead use `return_complex=True`  Note that calling [`torch.view_as_real()`](torch.view_as_real.html#torch.view_as_real "torch.view_as_real")  on the output will
            recover the deprecated output format.

Returns
:   A tensor containing the STFT result with shape *(B?, N, T, C?)* where
:   * *B?* is an optional batch dimension from the input.
* *N* is the number of frequency samples, *(n_fft // 2) + 1* for *onesided=True* , or otherwise *n_fft* .
* *T* is the number of frames, *1 + L // hop_length* for *center=True* , or *1 + (L - n_fft) // hop_length* otherwise.
* *C?* is an optional length-2 dimension of real and imaginary
components, present when *return_complex=False* .

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

