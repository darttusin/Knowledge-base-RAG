Dropout2d 
======================================================

*class* torch.nn. Dropout2d ( *p = 0.5*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/dropout.py#L118) 
:   Randomly zero out entire channels. 

A channel is a 2D feature map,
e.g., the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            j
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           j
          </annotation>
</semantics>
</math> -->j jj  -th channel of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           i
          </annotation>
</semantics>
</math> -->i ii  -th sample in the
batched input is a 2D tensor <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            input
           </mtext>
<mo stretchy="false">
            [
           </mo>
<mi>
            i
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            j
           </mi>
<mo stretchy="false">
            ]
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{input}[i, j]
          </annotation>
</semantics>
</math> -->input [ i , j ] text{input}[i, j]input [ i , j ]  . 

Each channel will be zeroed out independently on every forward call with
probability `p`  using samples from a Bernoulli distribution. 

Usually the input comes from `nn.Conv2d`  modules. 

As described in the paper [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)  ,
if adjacent pixels within feature maps are strongly correlated
(as is normally the case in early convolution layers) then i.i.d. dropout
will not regularize the activations and will otherwise just result
in an effective learning rate decrease. 

In this case, `nn.Dropout2d()`  will help promote independence between
feature maps and should be used instead. 

Parameters
:   * **p** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – probability of an element to be zero-ed.
* **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If set to `True`  , will do this operation
in-place

Warning 

Due to historical reasons, this class will perform 1D channel-wise dropout
for 3D inputs (as done by `nn.Dropout1d`  ). Thus, it currently does NOT
support inputs without a batch dimension of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             (
            </mo>
<mi>
             C
            </mi>
<mo separator="true">
             ,
            </mo>
<mi>
             H
            </mi>
<mo separator="true">
             ,
            </mo>
<mi>
             W
            </mi>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            (C, H, W)
           </annotation>
</semantics>
</math> -->( C , H , W ) (C, H, W)( C , H , W )  . This
behavior will change in a future release to interpret 3D inputs as no-batch-dim
inputs. To maintain the old behavior, switch to `nn.Dropout1d`  .

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
<mi>
                H
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                W
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, H, W)
              </annotation>
</semantics>
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi>
                L
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, L)
              </annotation>
</semantics>
</math> -->( N , C , L ) (N, C, L)( N , C , L )  .

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
<mi>
                H
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                W
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, H, W)
              </annotation>
</semantics>
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi>
                L
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, L)
              </annotation>
</semantics>
</math> -->( N , C , L ) (N, C, L)( N , C , L )  (same shape as input).

Examples: 

```
>>> m = nn.Dropout2d(p=0.2)
>>> input = torch.randn(20, 16, 32, 32)
>>> output = m(input)

```

