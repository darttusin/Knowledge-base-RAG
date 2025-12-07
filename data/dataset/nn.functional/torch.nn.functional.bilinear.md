torch.nn.functional.bilinear 
============================================================================================

torch.nn.functional. bilinear ( *input1*  , *input2*  , *weight*  , *bias = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies a bilinear transformation to the incoming data: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            =
           </mo>
<msubsup>
<mi>
             x
            </mi>
<mn>
             1
            </mn>
<mi>
             T
            </mi>
</msubsup>
<mi>
            A
           </mi>
<msub>
<mi>
             x
            </mi>
<mn>
             2
            </mn>
</msub>
<mo>
            +
           </mo>
<mi>
            b
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y = x_1^T A x_2 + b
          </annotation>
</semantics>
</math> -->y = x 1 T A x 2 + b y = x_1^T A x_2 + by = x 1 T ​ A x 2 ​ + b 

Shape: 

> * input1: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> N
> </mi>
> <mo separator="true">
> ,
> </mo>
> <mo>
> ∗
> </mo>
> <mo separator="true">
> ,
> </mo>
> <msub>
> <mi>
> H
> </mi>
> <mrow>
> <mi>
> i
> </mi>
> <mi>
> n
> </mi>
> <mn>
> 1
> </mn>
> </mrow>
> </msub>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (N, *, H_{in1})
> </annotation>
> </semantics>
> </math> -->( N , ∗ , H i n 1 ) (N, *, H_{in1})( N , ∗ , H in 1 ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <msub>
> <mi>
> H
> </mi>
> <mrow>
> <mi>
> i
> </mi>
> <mi>
> n
> </mi>
> <mn>
> 1
> </mn>
> </mrow>
> </msub>
> <mo>
> =
> </mo>
> <mtext>
> in1_features
> </mtext>
> </mrow>
> <annotation encoding="application/x-tex">
> H_{in1}=text{in1_features}
> </annotation>
> </semantics>
> </math> -->H i n 1 = in1_features H_{in1}=text{in1_features}H in 1 ​ = in1_features  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo>
> ∗
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> *
> </annotation>
> </semantics>
> </math> -->∗ *∗  means any number of additional dimensions.
> All but the last dimension of the inputs should be the same.
> 
> * input2: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> N
> </mi>
> <mo separator="true">
> ,
> </mo>
> <mo>
> ∗
> </mo>
> <mo separator="true">
> ,
> </mo>
> <msub>
> <mi>
> H
> </mi>
> <mrow>
> <mi>
> i
> </mi>
> <mi>
> n
> </mi>
> <mn>
> 2
> </mn>
> </mrow>
> </msub>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (N, *, H_{in2})
> </annotation>
> </semantics>
> </math> -->( N , ∗ , H i n 2 ) (N, *, H_{in2})( N , ∗ , H in 2 ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <msub>
> <mi>
> H
> </mi>
> <mrow>
> <mi>
> i
> </mi>
> <mi>
> n
> </mi>
> <mn>
> 2
> </mn>
> </mrow>
> </msub>
> <mo>
> =
> </mo>
> <mtext>
> in2_features
> </mtext>
> </mrow>
> <annotation encoding="application/x-tex">
> H_{in2}=text{in2_features}
> </annotation>
> </semantics>
> </math> -->H i n 2 = in2_features H_{in2}=text{in2_features}H in 2 ​ = in2_features
> 
> * weight: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mtext>
> out_features
> </mtext>
> <mo separator="true">
> ,
> </mo>
> <mtext>
> in1_features
> </mtext>
> <mo separator="true">
> ,
> </mo>
> <mtext>
> in2_features
> </mtext>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (text{out_features}, text{in1_features},
> text{in2_features})
> </annotation>
> </semantics>
> </math> -->( out_features , in1_features , in2_features ) (text{out_features}, text{in1_features},
> text{in2_features})( out_features , in1_features , in2_features )
> 
> * bias: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mtext>
> out_features
> </mtext>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (text{out_features})
> </annotation>
> </semantics>
> </math> -->( out_features ) (text{out_features})( out_features )
> 
> * output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> N
> </mi>
> <mo separator="true">
> ,
> </mo>
> <mo>
> ∗
> </mo>
> <mo separator="true">
> ,
> </mo>
> <msub>
> <mi>
> H
> </mi>
> <mrow>
> <mi>
> o
> </mi>
> <mi>
> u
> </mi>
> <mi>
> t
> </mi>
> </mrow>
> </msub>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (N, *, H_{out})
> </annotation>
> </semantics>
> </math> -->( N , ∗ , H o u t ) (N, *, H_{out})( N , ∗ , H o u t ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <msub>
> <mi>
> H
> </mi>
> <mrow>
> <mi>
> o
> </mi>
> <mi>
> u
> </mi>
> <mi>
> t
> </mi>
> </mrow>
> </msub>
> <mo>
> =
> </mo>
> <mtext>
> out_features
> </mtext>
> </mrow>
> <annotation encoding="application/x-tex">
> H_{out}=text{out_features}
> </annotation>
> </semantics>
> </math> -->H o u t = out_features H_{out}=text{out_features}H o u t ​ = out_features  and all but the last dimension are the same shape as the input.

