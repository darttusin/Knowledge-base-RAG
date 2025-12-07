torch.nn.functional.linear 
========================================================================================

torch.nn.functional. linear ( *input*  , *weight*  , *bias = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies a linear transformation to the incoming data: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            =
           </mo>
<mi>
            x
           </mi>
<msup>
<mi>
             A
            </mi>
<mi>
             T
            </mi>
</msup>
<mo>
            +
           </mo>
<mi>
            b
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y = xA^T + b
          </annotation>
</semantics>
</math> -->y = x A T + b y = xA^T + by = x A T + b  . 

This operation supports 2-D `weight`  with [sparse layout](../sparse.html#sparse-docs) 

Warning 

Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
or may not have autograd support. If you notice missing functionality please
open a feature request.

This operator supports [TensorFloat32](../notes/cuda.html#tf32-on-ampere)  . 

Shape: 

> * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mo>
> ∗
> </mo>
> <mo separator="true">
> ,
> </mo>
> <mi>
> i
> </mi>
> <mi>
> n
> </mi>
> <mi mathvariant="normal">
> _
> </mi>
> <mi>
> f
> </mi>
> <mi>
> e
> </mi>
> <mi>
> a
> </mi>
> <mi>
> t
> </mi>
> <mi>
> u
> </mi>
> <mi>
> r
> </mi>
> <mi>
> e
> </mi>
> <mi>
> s
> </mi>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (*, in_features)
> </annotation>
> </semantics>
> </math> -->( ∗ , i n _ f e a t u r e s ) (*, in_features)( ∗ , in _ f e a t u res )  where *** means any number of
> additional dimensions, including none
> 
> * Weight: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> o
> </mi>
> <mi>
> u
> </mi>
> <mi>
> t
> </mi>
> <mi mathvariant="normal">
> _
> </mi>
> <mi>
> f
> </mi>
> <mi>
> e
> </mi>
> <mi>
> a
> </mi>
> <mi>
> t
> </mi>
> <mi>
> u
> </mi>
> <mi>
> r
> </mi>
> <mi>
> e
> </mi>
> <mi>
> s
> </mi>
> <mo separator="true">
> ,
> </mo>
> <mi>
> i
> </mi>
> <mi>
> n
> </mi>
> <mi mathvariant="normal">
> _
> </mi>
> <mi>
> f
> </mi>
> <mi>
> e
> </mi>
> <mi>
> a
> </mi>
> <mi>
> t
> </mi>
> <mi>
> u
> </mi>
> <mi>
> r
> </mi>
> <mi>
> e
> </mi>
> <mi>
> s
> </mi>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (out_features, in_features)
> </annotation>
> </semantics>
> </math> -->( o u t _ f e a t u r e s , i n _ f e a t u r e s ) (out_features, in_features)( o u t _ f e a t u res , in _ f e a t u res )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> i
> </mi>
> <mi>
> n
> </mi>
> <mi mathvariant="normal">
> _
> </mi>
> <mi>
> f
> </mi>
> <mi>
> e
> </mi>
> <mi>
> a
> </mi>
> <mi>
> t
> </mi>
> <mi>
> u
> </mi>
> <mi>
> r
> </mi>
> <mi>
> e
> </mi>
> <mi>
> s
> </mi>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (in_features)
> </annotation>
> </semantics>
> </math> -->( i n _ f e a t u r e s ) (in_features)( in _ f e a t u res )
> 
> * Bias: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> o
> </mi>
> <mi>
> u
> </mi>
> <mi>
> t
> </mi>
> <mi mathvariant="normal">
> _
> </mi>
> <mi>
> f
> </mi>
> <mi>
> e
> </mi>
> <mi>
> a
> </mi>
> <mi>
> t
> </mi>
> <mi>
> u
> </mi>
> <mi>
> r
> </mi>
> <mi>
> e
> </mi>
> <mi>
> s
> </mi>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (out_features)
> </annotation>
> </semantics>
> </math> -->( o u t _ f e a t u r e s ) (out_features)( o u t _ f e a t u res )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> ()
> </annotation>
> </semantics>
> </math> -->( ) ()( )
> 
> * Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mo>
> ∗
> </mo>
> <mo separator="true">
> ,
> </mo>
> <mi>
> o
> </mi>
> <mi>
> u
> </mi>
> <mi>
> t
> </mi>
> <mi mathvariant="normal">
> _
> </mi>
> <mi>
> f
> </mi>
> <mi>
> e
> </mi>
> <mi>
> a
> </mi>
> <mi>
> t
> </mi>
> <mi>
> u
> </mi>
> <mi>
> r
> </mi>
> <mi>
> e
> </mi>
> <mi>
> s
> </mi>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (*, out_features)
> </annotation>
> </semantics>
> </math> -->( ∗ , o u t _ f e a t u r e s ) (*, out_features)( ∗ , o u t _ f e a t u res )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mo>
> ∗
> </mo>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (*)
> </annotation>
> </semantics>
> </math> -->( ∗ ) (*)( ∗ )  , based on the shape of the weight

