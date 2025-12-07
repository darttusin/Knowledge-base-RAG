LocalResponseNorm 
======================================================================

*class* torch.nn. LocalResponseNorm ( *size*  , *alpha = 0.0001*  , *beta = 0.75*  , *k = 1.0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L17) 
:   Applies local response normalization over an input signal. 

The input signal is composed of several input planes, where channels occupy the second dimension.
Applies normalization across channels. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             b
            </mi>
<mi>
             c
            </mi>
</msub>
<mo>
            =
           </mo>
<msub>
<mi>
             a
            </mi>
<mi>
             c
            </mi>
</msub>
<msup>
<mrow>
<mo fence="true">
              (
             </mo>
<mi>
              k
             </mi>
<mo>
              +
             </mo>
<mfrac>
<mi>
               α
              </mi>
<mi>
               n
              </mi>
</mfrac>
<munderover>
<mo>
               ∑
              </mo>
<mrow>
<msup>
<mi>
                 c
                </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msup>
<mo>
                =
               </mo>
<mi>
                max
               </mi>
<mo>
                ⁡
               </mo>
<mo stretchy="false">
                (
               </mo>
<mn>
                0
               </mn>
<mo separator="true">
                ,
               </mo>
<mi>
                c
               </mi>
<mo>
                −
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
<mo stretchy="false">
                )
               </mo>
</mrow>
<mrow>
<mi>
                min
               </mi>
<mo>
                ⁡
               </mo>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
<mo separator="true">
                ,
               </mo>
<mi>
                c
               </mi>
<mo>
                +
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
<mo stretchy="false">
                )
               </mo>
</mrow>
</munderover>
<msubsup>
<mi>
               a
              </mi>
<msup>
<mi>
                c
               </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                ′
               </mo>
</msup>
<mn>
               2
              </mn>
</msubsup>
<mo fence="true">
              )
             </mo>
</mrow>
<mrow>
<mo>
              −
             </mo>
<mi>
              β
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           b_{c} = a_{c}left(k + frac{alpha}{n}
sum_{c'=max(0, c-n/2)}^{min(N-1,c+n/2)}a_{c'}^2right)^{-beta}
          </annotation>
</semantics>
</math> -->
b c = a c ( k + α n ∑ c ′ = max ⁡ ( 0 , c − n / 2 ) min ⁡ ( N − 1 , c + n / 2 ) a c ′ 2 ) − β b_{c} = a_{c}left(k + frac{alpha}{n}
sum_{c'=max(0, c-n/2)}^{min(N-1,c+n/2)}a_{c'}^2right)^{-beta}

b c ​ = a c ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuNjAwZW0iIHZpZXdib3g9IjAgMCA4NzUgMzYwMCIgd2lkdGg9IjAuODc1ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik04NjMsOWMwLC0yLC0yLC01LC02LC05YzAsMCwtMTcsMCwtMTcsMGMtMTIuNywwLC0xOS4zLDAuMywtMjAsMQpjLTUuMyw1LjMsLTEwLjMsMTEsLTE1LDE3Yy0yNDIuNywyOTQuNywtMzk1LjMsNjgyLC00NTgsMTE2MmMtMjEuMywxNjMuMywtMzMuMywzNDksCi0zNiw1NTcgbDAsODRjMC4yLDYsMCwyNiwwLDYwYzIsMTU5LjMsMTAsMzEwLjcsMjQsNDU0YzUzLjMsNTI4LDIxMCwKOTQ5LjcsNDcwLDEyNjVjNC43LDYsOS43LDExLjcsMTUsMTdjMC43LDAuNyw3LDEsMTksMWMwLDAsMTgsMCwxOCwwYzQsLTQsNiwtNyw2LC05CmMwLC0yLjcsLTMuMywtOC43LC0xMCwtMThjLTEzNS4zLC0xOTIuNywtMjM1LjUsLTQxNC4zLC0zMDAuNSwtNjY1Yy02NSwtMjUwLjcsLTEwMi41LAotNTQ0LjcsLTExMi41LC04ODJjLTIsLTEwNCwtMywtMTY3LC0zLC0xODkKbDAsLTkyYzAsLTE2Mi43LDUuNywtMzE0LDE3LC00NTRjMjAuNywtMjcyLDYzLjcsLTUxMywxMjksLTcyM2M2NS4zLAotMjEwLDE1NS4zLC0zOTYuMywyNzAsLTU1OWM2LjcsLTkuMywxMCwtMTUuMywxMCwtMTh6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ k + n α ​ c ′ = m a x ( 0 , c − n /2 ) ∑ m i n ( N − 1 , c + n /2 ) ​ a c ′ 2 ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuNjAwZW0iIHZpZXdib3g9IjAgMCA4NzUgMzYwMCIgd2lkdGg9IjAuODc1ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik03NiwwYy0xNi43LDAsLTI1LDMsLTI1LDljMCwyLDIsNi4zLDYsMTNjMjEuMywyOC43LDQyLjMsNjAuMywKNjMsOTVjOTYuNywxNTYuNywxNzIuOCwzMzIuNSwyMjguNSw1MjcuNWM1NS43LDE5NSw5Mi44LDQxNi41LDExMS41LDY2NC41CmMxMS4zLDEzOS4zLDE3LDI5MC43LDE3LDQ1NGMwLDI4LDEuNyw0MywzLjMsNDVsMCw5CmMtMyw0LC0zLjMsMTYuNywtMy4zLDM4YzAsMTYyLC01LjcsMzEzLjcsLTE3LDQ1NWMtMTguNywyNDgsLTU1LjgsNDY5LjMsLTExMS41LDY2NApjLTU1LjcsMTk0LjcsLTEzMS44LDM3MC4zLC0yMjguNSw1MjdjLTIwLjcsMzQuNywtNDEuNyw2Ni4zLC02Myw5NWMtMiwzLjMsLTQsNywtNiwxMQpjMCw3LjMsNS43LDExLDE3LDExYzAsMCwxMSwwLDExLDBjOS4zLDAsMTQuMywtMC4zLDE1LC0xYzUuMywtNS4zLDEwLjMsLTExLDE1LC0xNwpjMjQyLjcsLTI5NC43LDM5NS4zLC02ODEuNyw0NTgsLTExNjFjMjEuMywtMTY0LjcsMzMuMywtMzUwLjcsMzYsLTU1OApsMCwtMTQ0Yy0yLC0xNTkuMywtMTAsLTMxMC43LC0yNCwtNDU0Yy01My4zLC01MjgsLTIxMCwtOTQ5LjcsCi00NzAsLTEyNjVjLTQuNywtNiwtOS43LC0xMS43LC0xNSwtMTdjLTAuNywtMC43LC02LjcsLTEsLTE4LC0xeiI+CjwvcGF0aD4KPC9zdmc+)​ − β

Parameters
:   * **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – amount of neighbouring channels used for normalization
* **alpha** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – multiplicative factor. Default: 0.0001
* **beta** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – exponent. Default: 0.75
* **k** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – additive factor. Default: 1

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, *)
              </annotation>
</semantics>
</math> -->( N , C , ∗ ) (N, C, *)( N , C , ∗ )

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, *)
              </annotation>
</semantics>
</math> -->( N , C , ∗ ) (N, C, *)( N , C , ∗ )  (same shape as input)

Examples: 

```
>>> lrn = nn.LocalResponseNorm(2)
>>> signal_2d = torch.randn(32, 5, 24, 24)
>>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
>>> output_2d = lrn(signal_2d)
>>> output_4d = lrn(signal_4d)

```

