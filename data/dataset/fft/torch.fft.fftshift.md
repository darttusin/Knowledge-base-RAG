torch.fft.fftshift 
========================================================================

torch.fft. fftshift ( *input*  , *dim = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Reorders n-dimensional FFT data, as provided by [`fftn()`](torch.fft.fftn.html#torch.fft.fftn "torch.fft.fftn")  , to have
negative frequency terms first. 

This performs a periodic shift of n-dimensional data such that the origin `(0, ..., 0)`  is moved to the center of the tensor. Specifically, to `input.shape[dim] // 2`  in each selected dimension. 

Note 

By convention, the FFT returns positive frequency terms first, followed by
the negative frequencies in reverse order, so that `f[-i]`  for all <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
             0
            </mn>
<mo>
             &lt;
            </mo>
<mi>
             i
            </mi>
<mo>
             ≤
            </mo>
<mi>
             n
            </mi>
<mi mathvariant="normal">
             /
            </mi>
<mn>
             2
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            0 &lt; i leq n/2
           </annotation>
</semantics>
</math> -->0 < i ≤ n / 2 0 < i leq n/20 < i ≤ n /2  in Python gives the negative frequency terms. [`fftshift()`](#torch.fft.fftshift "torch.fft.fftshift")  rearranges all frequencies into ascending order
from negative to positive with the zero-frequency term in the center.

Note 

For even lengths, the Nyquist frequency at `f[n/2]`  can be thought of as
either negative or positive. [`fftshift()`](#torch.fft.fftshift "torch.fft.fftshift")  always puts the
Nyquist term at the 0-index. This is the same convention used by [`fftfreq()`](torch.fft.fftfreq.html#torch.fft.fftfreq "torch.fft.fftfreq")  .

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor in FFT order
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *Tuple* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *,* *optional*  ) – The dimensions to rearrange.
Only dimensions specified here will be rearranged, any other dimensions
will be left in their original order.
Default: All dimensions of `input`  .

Example 

```
>>> f = torch.fft.fftfreq(4)
>>> f
tensor([ 0.0000,  0.2500, -0.5000, -0.2500])

```

```
>>> torch.fft.fftshift(f)
tensor([-0.5000, -0.2500,  0.0000,  0.2500])

```

Also notice that the Nyquist frequency term at `f[2]`  was moved to the
beginning of the tensor. 

This also works for multi-dimensional transforms: 

```
>>> x = torch.fft.fftfreq(5, d=1/5) + 0.1 * torch.fft.fftfreq(5, d=1/5).unsqueeze(1)
>>> x
tensor([[ 0.0000,  1.0000,  2.0000, -2.0000, -1.0000],
        [ 0.1000,  1.1000,  2.1000, -1.9000, -0.9000],
        [ 0.2000,  1.2000,  2.2000, -1.8000, -0.8000],
        [-0.2000,  0.8000,  1.8000, -2.2000, -1.2000],
        [-0.1000,  0.9000,  1.9000, -2.1000, -1.1000]])

```

```
>>> torch.fft.fftshift(x)
tensor([[-2.2000, -1.2000, -0.2000,  0.8000,  1.8000],
        [-2.1000, -1.1000, -0.1000,  0.9000,  1.9000],
        [-2.0000, -1.0000,  0.0000,  1.0000,  2.0000],
        [-1.9000, -0.9000,  0.1000,  1.1000,  2.1000],
        [-1.8000, -0.8000,  0.2000,  1.2000,  2.2000]])

```

[`fftshift()`](#torch.fft.fftshift "torch.fft.fftshift")  can also be useful for spatial data. If our
data is defined on a centered grid ( `[-(N//2), (N-1)//2]`  ) then we can
use the standard FFT defined on an uncentered grid ( `[0, N)`  ) by first
applying an [`ifftshift()`](torch.fft.ifftshift.html#torch.fft.ifftshift "torch.fft.ifftshift")  . 

```
>>> x_centered = torch.arange(-5, 5)
>>> x_uncentered = torch.fft.ifftshift(x_centered)
>>> fft_uncentered = torch.fft.fft(x_uncentered)

```

Similarly, we can convert the frequency domain components to centered
convention by applying [`fftshift()`](#torch.fft.fftshift "torch.fft.fftshift")  . 

```
>>> fft_centered = torch.fft.fftshift(fft_uncentered)

```

The inverse transform, from centered Fourier space back to centered spatial
data, can be performed by applying the inverse shifts in reverse order: 

```
>>> x_centered_2 = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fft_centered)))
>>> torch.testing.assert_close(x_centered.to(torch.complex64), x_centered_2, check_stride=False)

```

