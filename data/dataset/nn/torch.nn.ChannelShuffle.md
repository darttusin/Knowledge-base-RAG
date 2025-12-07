ChannelShuffle 
================================================================

*class* torch.nn. ChannelShuffle ( *groups* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/channelshuffle.py#L10) 
:   Divides and rearranges the channels in a tensor. 

This operation divides the channels in a tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C , ∗ ) (N, C, *)( N , C , ∗ )  into g groups as <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mfrac>
<mi>
             C
            </mi>
<mi>
             g
            </mi>
</mfrac>
<mo separator="true">
            ,
           </mo>
<mi>
            g
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
           (N, frac{C}{g}, g, *)
          </annotation>
</semantics>
</math> -->( N , C g , g , ∗ ) (N, frac{C}{g}, g, *)( N , g C ​ , g , ∗ )  and shuffles them,
while retaining the original tensor shape in the final output. 

Parameters
: **groups** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – number of groups to divide channels in.

Examples: 

```
>>> channel_shuffle = nn.ChannelShuffle(2)
>>> input = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 2, 2)
>>> input
tensor([[[[ 1.,  2.],
          [ 3.,  4.]],
         [[ 5.,  6.],
          [ 7.,  8.]],
         [[ 9., 10.],
          [11., 12.]],
         [[13., 14.],
          [15., 16.]]]])
>>> output = channel_shuffle(input)
>>> output
tensor([[[[ 1.,  2.],
          [ 3.,  4.]],
         [[ 9., 10.],
          [11., 12.]],
         [[ 5.,  6.],
          [ 7.,  8.]],
         [[13., 14.],
          [15., 16.]]]])

```

