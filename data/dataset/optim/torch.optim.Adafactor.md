Adafactor 
======================================================

*class* torch.optim. Adafactor ( *params*  , *lr = 0.01*  , *beta2_decay = -0.8*  , *eps = (None, 0.001)*  , *d = 1.0*  , *weight_decay = 0.0*  , *** , *foreach = None*  , *maximize = False* ) 
:   Implements Adafactor algorithm. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mpadded height="0em" voffset="0em">
<mspace height="0.04em" mathbackground="black" width="31.298em">
</mspace>
</mpadded>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mtext mathvariant="bold">
                input
               </mtext>
<mo>
                :
               </mo>
<mi>
                γ
               </mi>
<mtext>
                (lr)
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
</mtext>
<mi>
                τ
               </mi>
<mtext>
                (
               </mtext>
<msub>
<mi>
                 β
                </mi>
<mn>
                 2
                </mn>
</msub>
<mtext>
                decay)
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
</mtext>
<msub>
<mi>
                 θ
                </mi>
<mn>
                 0
                </mn>
</msub>
<mtext>
                (params)
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
</mtext>
<mi>
                f
               </mi>
<mo stretchy="false">
                (
               </mo>
<mi>
                θ
               </mi>
<mo stretchy="false">
                )
               </mo>
<mtext>
                (objective)
               </mtext>
<mo separator="true">
                ,
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="4.2679em">
</mspace>
<mtext>
</mtext>
<msub>
<mi>
                 ϵ
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 ϵ
                </mi>
<mn>
                 2
                </mn>
</msub>
<mtext>
                (epsilons)
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
</mtext>
<mi>
                d
               </mi>
<mtext>
                (clipping threshold)
               </mtext>
<mo separator="true">
                ,
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="4.2679em">
</mspace>
<mtext>
</mtext>
<mi>
                λ
               </mi>
<mtext>
                (weight decay)
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
</mtext>
<mtext mathvariant="italic">
                maximize
               </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mtext mathvariant="bold">
                initialize
               </mtext>
<mo>
                :
               </mo>
<mtext>
</mtext>
<msub>
<mi>
                 R
                </mi>
<mn>
                 0
                </mn>
</msub>
<mo>
                ←
               </mo>
<mn>
                0
               </mn>
<mtext>
                (second moment row factor)
               </mtext>
<mo separator="true">
                ,
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="6.5441em">
</mspace>
<mtext>
</mtext>
<msub>
<mi>
                 C
                </mi>
<mn>
                 0
                </mn>
</msub>
<mo>
                ←
               </mo>
<mn>
                0
               </mn>
<mtext>
                (second moment col factor)
               </mtext>
<mo separator="true">
                ,
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="6.5441em">
</mspace>
<mtext>
</mtext>
<msub>
<mover accent="true">
<mi>
                  V
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<mn>
                 0
                </mn>
</msub>
<mo>
                ←
               </mo>
<mn>
                0
               </mn>
<mtext>
                (second moment for vectors)
               </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mpadded height="0em" voffset="0em">
<mspace height="0.04em" mathbackground="black" width="31.298em">
</mspace>
</mpadded>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mtext mathvariant="bold">
                for
               </mtext>
<mtext>
</mtext>
<mi>
                t
               </mi>
<mo>
                =
               </mo>
<mn>
                1
               </mn>
<mtext>
</mtext>
<mtext mathvariant="bold">
                to
               </mtext>
<mtext>
</mtext>
<mo>
                …
               </mo>
<mtext>
</mtext>
<mtext mathvariant="bold">
                do
               </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<mtext mathvariant="bold">
                if
               </mtext>
<mtext>
</mtext>
<mtext mathvariant="italic">
                maximize
               </mtext>
<mo>
                :
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="2.8453em">
</mspace>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<mo>
                −
               </mo>
<msub>
<mi mathvariant="normal">
                 ∇
                </mi>
<mi>
                 θ
                </mi>
</msub>
<msub>
<mi>
                 f
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 θ
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<mtext mathvariant="bold">
                else
               </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="2.8453em">
</mspace>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<msub>
<mi mathvariant="normal">
                 ∇
                </mi>
<mi>
                 θ
                </mi>
</msub>
<msub>
<mi>
                 f
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 θ
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mover accent="true">
<mi>
                  β
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<msub>
<mn>
                  2
                 </mn>
<mi>
                  t
                 </mi>
</msub>
</msub>
<mo>
                ←
               </mo>
<mn>
                1
               </mn>
<mo>
                −
               </mo>
<msup>
<mi>
                 t
                </mi>
<mi>
                 τ
                </mi>
</msup>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                 ρ
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<mi>
                m
               </mi>
<mi>
                i
               </mi>
<mi>
                n
               </mi>
<mo stretchy="false">
                (
               </mo>
<mi>
                l
               </mi>
<mi>
                r
               </mi>
<mo separator="true">
                ,
               </mo>
<mfrac>
<mn>
                 1
                </mn>
<msqrt>
<mi>
                  t
                 </mi>
</msqrt>
</mfrac>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                 α
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<mi>
                m
               </mi>
<mi>
                a
               </mi>
<mi>
                x
               </mi>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 ϵ
                </mi>
<mn>
                 2
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<mtext>
                RMS
               </mtext>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 θ
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo stretchy="false">
                )
               </mo>
<msub>
<mi>
                 ρ
                </mi>
<mi>
                 t
                </mi>
</msub>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                 θ
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<msub>
<mi>
                 θ
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
<mo>
                −
               </mo>
<mi>
                γ
               </mi>
<mi>
                λ
               </mi>
<msub>
<mi>
                 θ
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<mtext mathvariant="bold">
                if
               </mtext>
<mtext>
</mtext>
<mtext>
                dim
               </mtext>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo>
                &gt;
               </mo>
<mn>
                1
               </mn>
<mo>
                :
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="2.8453em">
</mspace>
<msub>
<mi>
                 R
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<msub>
<mover accent="true">
<mi>
                  β
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<msub>
<mn>
                  2
                 </mn>
<mi>
                  t
                 </mi>
</msub>
</msub>
<msub>
<mi>
                 R
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
<mo>
                +
               </mo>
<mo stretchy="false">
                (
               </mo>
<mn>
                1
               </mn>
<mo>
                −
               </mo>
<msub>
<mover accent="true">
<mi>
                  β
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<msub>
<mn>
                  2
                 </mn>
<mi>
                  t
                 </mi>
</msub>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ⊙
               </mo>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo>
                ⋅
               </mo>
<msub>
<mn>
                 1
                </mn>
<mi>
                 m
                </mi>
</msub>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="2.8453em">
</mspace>
<msub>
<mi>
                 C
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<msub>
<mover accent="true">
<mi>
                  β
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<msub>
<mn>
                  2
                 </mn>
<mi>
                  t
                 </mi>
</msub>
</msub>
<msub>
<mi>
                 C
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
<mo>
                +
               </mo>
<mo stretchy="false">
                (
               </mo>
<mn>
                1
               </mn>
<mo>
                −
               </mo>
<msub>
<mover accent="true">
<mi>
                  β
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<msub>
<mn>
                  2
                 </mn>
<mi>
                  t
                 </mi>
</msub>
</msub>
<mo stretchy="false">
                )
               </mo>
<msubsup>
<mn>
                 1
                </mn>
<mi>
                 n
                </mi>
<mi mathvariant="normal">
                 ⊤
                </mi>
</msubsup>
<mo>
                ⋅
               </mo>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ⊙
               </mo>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="2.8453em">
</mspace>
<msub>
<mover accent="true">
<mi>
                  V
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<mfrac>
<mrow>
<msub>
<mi>
                   R
                  </mi>
<mi>
                   t
                  </mi>
</msub>
<mo>
                  ⋅
                 </mo>
<msub>
<mi>
                   C
                  </mi>
<mi>
                   t
                  </mi>
</msub>
</mrow>
<mrow>
<mi>
                  m
                 </mi>
<mi>
                  a
                 </mi>
<mi>
                  x
                 </mi>
<mo stretchy="false">
                  (
                 </mo>
<msubsup>
<mn>
                   1
                  </mn>
<mi>
                   n
                  </mi>
<mi mathvariant="normal">
                   ⊤
                  </mi>
</msubsup>
<mo>
                  ⋅
                 </mo>
<msub>
<mi>
                   R
                  </mi>
<mi>
                   t
                  </mi>
</msub>
<mo separator="true">
                  ,
                 </mo>
<msub>
<mi>
                   ϵ
                  </mi>
<mn>
                   1
                  </mn>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<mtext mathvariant="bold">
                else
               </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="2.8453em">
</mspace>
<msub>
<mover accent="true">
<mi>
                  V
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<msub>
<mover accent="true">
<mi>
                  β
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<msub>
<mn>
                  2
                 </mn>
<mi>
                  t
                 </mi>
</msub>
</msub>
<msub>
<mover accent="true">
<mi>
                  V
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
<mo>
                +
               </mo>
<mo stretchy="false">
                (
               </mo>
<mn>
                1
               </mn>
<mo>
                −
               </mo>
<msub>
<mover accent="true">
<mi>
                  β
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<msub>
<mn>
                  2
                 </mn>
<mi>
                  t
                 </mi>
</msub>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo>
                ⋅
               </mo>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ⊙
               </mo>
<msub>
<mi>
                 G
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                 U
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<mfrac>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mrow>
<mi>
                  m
                 </mi>
<mi>
                  a
                 </mi>
<mi>
                  x
                 </mi>
<mo stretchy="false">
                  (
                 </mo>
<msqrt>
<msub>
<mover accent="true">
<mi>
                     V
                    </mi>
<mo stretchy="true">
                     ^
                    </mo>
</mover>
<mi>
                    t
                   </mi>
</msub>
</msqrt>
<mo separator="true">
                  ,
                 </mo>
<msub>
<mi>
                   ϵ
                  </mi>
<mn>
                   1
                  </mn>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mover accent="true">
<mi>
                  U
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<mfrac>
<msub>
<mi>
                  U
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mrow>
<mi>
                  m
                 </mi>
<mi>
                  a
                 </mi>
<mi>
                  x
                 </mi>
<mo stretchy="false">
                  (
                 </mo>
<mn>
                  1
                 </mn>
<mo separator="true">
                  ,
                 </mo>
<mfrac>
<mrow>
<mtext>
                    RMS
                   </mtext>
<mo stretchy="false">
                    (
                   </mo>
<msub>
<mi>
                     U
                    </mi>
<mi>
                     t
                    </mi>
</msub>
<mo stretchy="false">
                    )
                   </mo>
</mrow>
<mi>
                   d
                  </mi>
</mfrac>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                 θ
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<msub>
<mi>
                 θ
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
<mo>
                −
               </mo>
<msub>
<mi>
                 α
                </mi>
<mi>
                 t
                </mi>
</msub>
<msub>
<mover accent="true">
<mi>
                  U
                 </mi>
<mo stretchy="true">
                  ^
                 </mo>
</mover>
<mi>
                 t
                </mi>
</msub>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mpadded height="0em" voffset="0em">
<mspace height="0.04em" mathbackground="black" width="31.298em">
</mspace>
</mpadded>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mrow>
<mrow>
<mi mathvariant="bold">
                  r
                 </mi>
<mi mathvariant="bold">
                  e
                 </mi>
<mi mathvariant="bold">
                  t
                 </mi>
<mi mathvariant="bold">
                  u
                 </mi>
<mi mathvariant="bold">
                  r
                 </mi>
<mi mathvariant="bold">
                  n
                 </mi>
</mrow>
<mtext>
</mtext>
<msub>
<mi mathvariant="bold">
                  θ
                 </mi>
<mi mathvariant="bold">
                  t
                 </mi>
</msub>
</mrow>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mpadded height="0em" voffset="0em">
<mspace height="0.04em" mathbackground="black" width="31.298em">
</mspace>
</mpadded>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
           begin{aligned}
    &amp;rule{110mm}{0.4pt} 
    &amp;textbf{input} : gamma text{(lr)}, : tau
        text{(}beta_2text{ decay)}, : theta_0 text{(params)}, : f(theta) text{(objective)}, 
    &amp;hspace{15mm} : epsilon_1, epsilon_2 text{ (epsilons)}, : d text{(clipping threshold)}, 
    &amp;hspace{15mm} : lambda text{(weight decay)},
        : textit{maximize} 
    &amp;textbf{initialize} : : R_0 leftarrow 0 text{ (second moment row factor)}, 
    &amp;hspace{23mm} : C_0 leftarrow 0 text{ (second moment col factor)}, 
    &amp;hspace{23mm} : widehat{V}_0 leftarrow 0 text{ (second moment for vectors)} [-1.ex]
    &amp;rule{110mm}{0.4pt} 
    &amp;textbf{for} : t=1 : textbf{to} : ldots : textbf{do} 

    &amp;hspace{5mm}textbf{if} : textit{maximize}:                                       
    &amp;hspace{10mm}G_t leftarrow -nabla_{theta} f_t (theta_{t-1}) 
    &amp;hspace{5mm}textbf{else} 
    &amp;hspace{10mm}G_t leftarrow nabla_{theta} f_t (theta_{t-1}) 
    &amp;hspace{5mm}widehat{beta}_{2_t} leftarrow 1 - t^{tau} 
    &amp;hspace{5mm}rho_t leftarrow min(lr, frac{1}{sqrt{t}}) 
    &amp;hspace{5mm}alpha_t leftarrow max(epsilon_2,
        text{RMS}(theta_{t-1}))rho_t 
    &amp;hspace{5mm}theta_t leftarrow theta_{t-1} - gamma lambda theta_{t-1} 
    &amp;hspace{5mm}textbf{if} : text{dim}(G_t) &gt; 1:                                     
    &amp;hspace{10mm}R_t leftarrow widehat{beta}_{2_t}R_{t-1}+
        (1-widehat{beta}_{2_t})(G_t odot G_t) cdot 1_m 
    &amp;hspace{10mm}C_t leftarrow widehat{beta}_{2_t}C_{t-1}+
        (1-widehat{beta}_{2_t}) 1^top_n cdot (G_t odot G_t) 
    &amp;hspace{10mm}widehat{V}_t leftarrow
        frac{R_t cdot C_t}{max(1^top_n cdot R_t, epsilon_1)} 
    &amp;hspace{5mm}textbf{else} 
    &amp;hspace{10mm}widehat{V}_t leftarrow widehat{beta}_{2_t}widehat{V}_{t-1}+
        (1-widehat{beta}_{2_t}) cdot (G_t odot G_t) 
    &amp;hspace{5mm}U_t leftarrow
        frac{G_t}{max(sqrt{widehat{V}_t}, epsilon_1)} 
    &amp;hspace{5mm}widehat{U}_t  leftarrow frac{U_t}{max(1, frac{text{RMS}(U_t)}{d})} 
    &amp;hspace{5mm}theta_t leftarrow theta_{t-1} - alpha_t widehat{U}_t 

    &amp;rule{110mm}{0.4pt} [-1.ex]
    &amp;bf{return} :  theta_t [-1.ex]
    &amp;rule{110mm}{0.4pt} [-1.ex]
end{aligned}
          </annotation>
</semantics>
</math> -->
input : γ (lr) , τ ( β 2 decay) , θ 0 (params) , f ( θ ) (objective) , ϵ 1 , ϵ 2 (epsilons) , d (clipping threshold) , λ (weight decay) , maximize initialize : R 0 ← 0 (second moment row factor) , C 0 ← 0 (second moment col factor) , V ^ 0 ← 0 (second moment for vectors) for t = 1 to … do if maximize : G t ← − ∇ θ f t ( θ t − 1 ) else G t ← ∇ θ f t ( θ t − 1 ) β ^ 2 t ← 1 − t τ ρ t ← m i n ( l r , 1 t ) α t ← m a x ( ϵ 2 , RMS ( θ t − 1 ) ) ρ t θ t ← θ t − 1 − γ λ θ t − 1 if dim ( G t ) > 1 : R t ← β ^ 2 t R t − 1 + ( 1 − β ^ 2 t ) ( G t ⊙ G t ) ⋅ 1 m C t ← β ^ 2 t C t − 1 + ( 1 − β ^ 2 t ) 1 n ⊤ ⋅ ( G t ⊙ G t ) V ^ t ← R t ⋅ C t m a x ( 1 n ⊤ ⋅ R t , ϵ 1 ) else V ^ t ← β ^ 2 t V ^ t − 1 + ( 1 − β ^ 2 t ) ⋅ ( G t ⊙ G t ) U t ← G t m a x ( V ^ t , ϵ 1 ) U ^ t ← U t m a x ( 1 , RMS ( U t ) d ) θ t ← θ t − 1 − α t U ^ t r e t u r n θ t begin{aligned}
 &rule{110mm}{0.4pt} 
 &textbf{input} : gamma text{(lr)}, : tau
 text{(}beta_2text{ decay)}, : theta_0 text{(params)}, : f(theta) text{(objective)}, 
 &hspace{15mm} : epsilon_1, epsilon_2 text{ (epsilons)}, : d text{(clipping threshold)}, 
 &hspace{15mm} : lambda text{(weight decay)},
 : textit{maximize} 
 &textbf{initialize} : : R_0 leftarrow 0 text{ (second moment row factor)}, 
 &hspace{23mm} : C_0 leftarrow 0 text{ (second moment col factor)}, 
 &hspace{23mm} : widehat{V}_0 leftarrow 0 text{ (second moment for vectors)} [-1.ex]
 &rule{110mm}{0.4pt} 
 &textbf{for} : t=1 : textbf{to} : ldots : textbf{do} 

 &hspace{5mm}textbf{if} : textit{maximize}: 
 &hspace{10mm}G_t leftarrow -nabla_{theta} f_t (theta_{t-1}) 
 &hspace{5mm}textbf{else} 
 &hspace{10mm}G_t leftarrow nabla_{theta} f_t (theta_{t-1}) 
 &hspace{5mm}widehat{beta}_{2_t} leftarrow 1 - t^{tau} 
 &hspace{5mm}rho_t leftarrow min(lr, frac{1}{sqrt{t}}) 
 &hspace{5mm}alpha_t leftarrow max(epsilon_2,
 text{RMS}(theta_{t-1}))rho_t 
 &hspace{5mm}theta_t leftarrow theta_{t-1} - gamma lambda theta_{t-1} 
 &hspace{5mm}textbf{if} : text{dim}(G_t) > 1: 
 &hspace{10mm}R_t leftarrow widehat{beta}_{2_t}R_{t-1}+
 (1-widehat{beta}_{2_t})(G_t odot G_t) cdot 1_m 
 &hspace{10mm}C_t leftarrow widehat{beta}_{2_t}C_{t-1}+
 (1-widehat{beta}_{2_t}) 1^top_n cdot (G_t odot G_t) 
 &hspace{10mm}widehat{V}_t leftarrow
 frac{R_t cdot C_t}{max(1^top_n cdot R_t, epsilon_1)} 
 &hspace{5mm}textbf{else} 
 &hspace{10mm}widehat{V}_t leftarrow widehat{beta}_{2_t}widehat{V}_{t-1}+
 (1-widehat{beta}_{2_t}) cdot (G_t odot G_t) 
 &hspace{5mm}U_t leftarrow
 frac{G_t}{max(sqrt{widehat{V}_t}, epsilon_1)} 
 &hspace{5mm}widehat{U}_t leftarrow frac{U_t}{max(1, frac{text{RMS}(U_t)}{d})} 
 &hspace{5mm}theta_t leftarrow theta_{t-1} - alpha_t widehat{U}_t 

 &rule{110mm}{0.4pt} [-1.ex]
 &bf{return} : theta_t [-1.ex]
 &rule{110mm}{0.4pt} [-1.ex]
end{aligned}

​ input : γ (lr) , τ ( β 2 ​ decay) , θ 0 ​ (params) , f ( θ ) (objective) , ϵ 1 ​ , ϵ 2 ​ (epsilons) , d (clipping threshold) , λ (weight decay) , maximize initialize : R 0 ​ ← 0 (second moment row factor) , C 0 ​ ← 0 (second moment col factor) , V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)0 ​ ← 0 (second moment for vectors) for t = 1 to … do if maximize : G t ​ ← − ∇ θ ​ f t ​ ( θ t − 1 ​ ) else G t ​ ← ∇ θ ​ f t ​ ( θ t − 1 ​ ) β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ ← 1 − t τ ρ t ​ ← min ( l r , t ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 1 ​ ) α t ​ ← ma x ( ϵ 2 ​ , RMS ( θ t − 1 ​ )) ρ t ​ θ t ​ ← θ t − 1 ​ − γλ θ t − 1 ​ if dim ( G t ​ ) > 1 : R t ​ ← β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ R t − 1 ​ + ( 1 − β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ ) ( G t ​ ⊙ G t ​ ) ⋅ 1 m ​ C t ​ ← β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ C t − 1 ​ + ( 1 − β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ ) 1 n ⊤ ​ ⋅ ( G t ​ ⊙ G t ​ ) V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ ← ma x ( 1 n ⊤ ​ ⋅ R t ​ , ϵ 1 ​ ) R t ​ ⋅ C t ​ ​ else V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ ← β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t − 1 ​ + ( 1 − β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ ) ⋅ ( G t ​ ⊙ G t ​ ) U t ​ ← ma x ( V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ , ϵ 1 ​ ) G t ​ ​ U ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ ← ma x ( 1 , d RMS ( U t ​ ) ​ ) U t ​ ​ θ t ​ ← θ t − 1 ​ − α t ​ U ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ return θ t ​ ​

For further details regarding the algorithm we refer to [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/pdf/1804.04235)  . 

Parameters
:   * **params** ( *iterable*  ) – iterable of parameters or named_parameters to optimize
or iterable of dicts defining parameter groups. When using named_parameters,
all parameters in all groups should be named
* **lr** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – unlike other optimizers, Adafactor does not require a
learning rate, and Noam Shazeer and Mitchell Stern do not use lr at all.
Deviating from the paper, this implementation uses lr for applying weight
decay and as the maximum value for relative step size rho_t. Note that in
the paper, a constant of 0.01 is used as the maximum value for relative
step size, and so we set 0.01 as the default value. (default: 1e-2)
* **beta2_decay** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – the decay rate of beta2. beta2 standardly refers
to the coefficient used for computing the running average of the gradient
squared. (default: -0.8)
* **eps** ( *Tuple* *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]* *,* *optional*  ) – epsilon1 is the term added to the denominator
of the update calculation to improve numerical stability. This use of epsilon1
deviates from the algorithm written in the paper! See note below for more details.
epsilon2 is the term used to avoid having too small a weight update when applying
parameter scaling. (default: (None, 1e-3))
* **d** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – the clipping threshold, used to avoid larger-than-desired
updates.
* **weight_decay** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – weight decay coefficient (default: 1e-2)
* **foreach** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether foreach implementation of optimizer is used. Note
that the foreach implementation uses ~ sizeof(params) more peak memory than the
for-loop version due to the intermediates being a tensorlist vs just one tensor.
As Adafactor is commonly used when memory is prohibitive, Adafactor will default
to the slower single tensor for-loop implementation unless this flag is explicitly
True. This behavior is contrary to other optimizers, which will attempt defaulting
to foreach on CUDA for faster runtime. (default: None)
* **maximize** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – maximize the objective with respect to the
params, instead of minimizing (default: False)

Note 

The implementation of Adafactor subtly differs from Noam Shazeer and Mitchell Stern
and implementations in some other frameworks with its use of learning rate and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              ϵ
             </mi>
<mn>
              1
             </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            epsilon_1
           </annotation>
</semantics>
</math> -->ϵ 1 epsilon_1ϵ 1 ​  . 

Regarding the learning rate hyperparameter: Noam Shazeer and Mitchell Stern do not
use lr at all, as the stated algorithm uses <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              ρ
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            rho_t
           </annotation>
</semantics>
</math> -->ρ t rho_tρ t ​  and update clipping to
affect the step size. 

This implementation allows *lr* to influence the maximum value for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              ρ
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            rho_t
           </annotation>
</semantics>
</math> -->ρ t rho_tρ t ​  : 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                  ρ
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<mi>
                 m
                </mi>
<mi>
                 i
                </mi>
<mi>
                 n
                </mi>
<mo stretchy="false">
                 (
                </mo>
<mi>
                 l
                </mi>
<mi>
                 r
                </mi>
<mo separator="true">
                 ,
                </mo>
<mfrac>
<mn>
                  1
                 </mn>
<msqrt>
<mi>
                   t
                  </mi>
</msqrt>
</mfrac>
<mo stretchy="false">
                 )
                </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{aligned}
    &amp;hspace{5mm}rho_t leftarrow min(lr, frac{1}{sqrt{t}})
end{aligned}
           </annotation>
</semantics>
</math> -->
ρ t ← m i n ( l r , 1 t ) begin{aligned}
 &hspace{5mm}rho_t leftarrow min(lr, frac{1}{sqrt{t}})
end{aligned}

​ ρ t ​ ← min ( l r , t ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 1 ​ ) ​

This differs from Noam Shazeer and Mitchell Stern, who use a constant of 0.01 as
the maximum value of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              ρ
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            rho_t
           </annotation>
</semantics>
</math> -->ρ t rho_tρ t ​ 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                  ρ
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<mi>
                 m
                </mi>
<mi>
                 i
                </mi>
<mi>
                 n
                </mi>
<mo stretchy="false">
                 (
                </mo>
<mn>
                 0.01
                </mn>
<mo separator="true">
                 ,
                </mo>
<mfrac>
<mn>
                  1
                 </mn>
<msqrt>
<mi>
                   t
                  </mi>
</msqrt>
</mfrac>
<mo stretchy="false">
                 )
                </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{aligned}
    &amp;hspace{5mm}rho_t leftarrow min(0.01, frac{1}{sqrt{t}})
end{aligned}
           </annotation>
</semantics>
</math> -->
ρ t ← m i n ( 0.01 , 1 t ) begin{aligned}
 &hspace{5mm}rho_t leftarrow min(0.01, frac{1}{sqrt{t}})
end{aligned}

​ ρ t ​ ← min ( 0.01 , t ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 1 ​ ) ​

Noam Shazeer and Mitchell Stern do not enforce an opinion on how weight decay should
be computed, and so we use the learning rate as a coefficient for decoupled weight
decay, similar to what is suggested in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)  . 

Regarding the use of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              ϵ
             </mi>
<mn>
              1
             </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            epsilon_1
           </annotation>
</semantics>
</math> -->ϵ 1 epsilon_1ϵ 1 ​  : The implementation attempts to replicate the
presumed intention of Noam Shazeer and Mitchell Stern to use <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              ϵ
             </mi>
<mn>
              1
             </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            epsilon_1
           </annotation>
</semantics>
</math> -->ϵ 1 epsilon_1ϵ 1 ​  as
a stabilizing term when the squared gradient becomes small. 

This stabilization can be written as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                  R
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<msub>
<mover accent="true">
<mi>
                   β
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<msub>
<mn>
                   2
                  </mn>
<mi>
                   t
                  </mi>
</msub>
</msub>
<msub>
<mi>
                  R
                 </mi>
<mrow>
<mi>
                   t
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msub>
<mo>
                 +
                </mo>
<mo stretchy="false">
                 (
                </mo>
<mn>
                 1
                </mn>
<mo>
                 −
                </mo>
<msub>
<mover accent="true">
<mi>
                   β
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<msub>
<mn>
                   2
                  </mn>
<mi>
                   t
                  </mi>
</msub>
</msub>
<mo stretchy="false">
                 )
                </mo>
<mo stretchy="false">
                 (
                </mo>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ⊙
                </mo>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 +
                </mo>
<msub>
<mn>
                  1
                 </mn>
<mi>
                  n
                 </mi>
</msub>
<mo>
                 ⋅
                </mo>
<msubsup>
<mn>
                  1
                 </mn>
<mi>
                  m
                 </mi>
<mi mathvariant="normal">
                  ⊤
                 </mi>
</msubsup>
<mo stretchy="false">
                 )
                </mo>
<mo>
                 ⋅
                </mo>
<msub>
<mn>
                  1
                 </mn>
<mi>
                  m
                 </mi>
</msub>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                  C
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<msub>
<mover accent="true">
<mi>
                   β
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<msub>
<mn>
                   2
                  </mn>
<mi>
                   t
                  </mi>
</msub>
</msub>
<msub>
<mi>
                  C
                 </mi>
<mrow>
<mi>
                   t
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msub>
<mo>
                 +
                </mo>
<mo stretchy="false">
                 (
                </mo>
<mn>
                 1
                </mn>
<mo>
                 −
                </mo>
<msub>
<mover accent="true">
<mi>
                   β
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<msub>
<mn>
                   2
                  </mn>
<mi>
                   t
                  </mi>
</msub>
</msub>
<mo stretchy="false">
                 )
                </mo>
<msubsup>
<mn>
                  1
                 </mn>
<mi>
                  n
                 </mi>
<mi mathvariant="normal">
                  ⊤
                 </mi>
</msubsup>
<mo>
                 ⋅
                </mo>
<mo stretchy="false">
                 (
                </mo>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ⊙
                </mo>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 +
                </mo>
<msub>
<mn>
                  1
                 </mn>
<mi>
                  n
                 </mi>
</msub>
<mo>
                 ⋅
                </mo>
<msubsup>
<mn>
                  1
                 </mn>
<mi>
                  m
                 </mi>
<mi mathvariant="normal">
                  ⊤
                 </mi>
</msubsup>
<mo stretchy="false">
                 )
                </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mover accent="true">
<mi>
                   V
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<mfrac>
<mrow>
<msub>
<mi>
                    R
                   </mi>
<mi>
                    t
                   </mi>
</msub>
<mo>
                   ⋅
                  </mo>
<msub>
<mi>
                    C
                   </mi>
<mi>
                    t
                   </mi>
</msub>
</mrow>
<mrow>
<mi>
                   m
                  </mi>
<mi>
                   a
                  </mi>
<mi>
                   x
                  </mi>
<mo stretchy="false">
                   (
                  </mo>
<msubsup>
<mn>
                    1
                   </mn>
<mi>
                    n
                   </mi>
<mi mathvariant="normal">
                    ⊤
                   </mi>
</msubsup>
<mo>
                   ⋅
                  </mo>
<msub>
<mi>
                    R
                   </mi>
<mi>
                    t
                   </mi>
</msub>
<mo separator="true">
                   ,
                  </mo>
<msub>
<mi>
                    ϵ
                   </mi>
<mn>
                    1
                   </mn>
</msub>
<mo stretchy="false">
                   )
                  </mo>
</mrow>
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                  U
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<mfrac>
<msub>
<mi>
                   G
                  </mi>
<mi>
                   t
                  </mi>
</msub>
<mrow>
<mi>
                   m
                  </mi>
<mi>
                   a
                  </mi>
<mi>
                   x
                  </mi>
<mo stretchy="false">
                   (
                  </mo>
<msqrt>
<msub>
<mover accent="true">
<mi>
                      V
                     </mi>
<mo stretchy="true">
                      ^
                     </mo>
</mover>
<mi>
                     t
                    </mi>
</msub>
</msqrt>
<mo separator="true">
                   ,
                  </mo>
<msub>
<mi>
                    ϵ
                   </mi>
<mn>
                    1
                   </mn>
</msub>
<mo stretchy="false">
                   )
                  </mo>
</mrow>
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{aligned}
    &amp;hspace{5mm}R_t leftarrow widehat{beta}_{2_t}R_{t-1}+
        (1-widehat{beta}_{2_t})(G_t odot G_t + 1_n cdot 1^top_m) cdot 1_m 
    &amp;hspace{5mm}C_t leftarrow widehat{beta}_{2_t}C_{t-1}+
        (1-widehat{beta}_{2_t}) 1^top_n cdot (G_t odot G_t + 1_n cdot 1^top_m) 
    &amp;hspace{5mm}widehat{V}_t leftarrow
        frac{R_t cdot C_t}{max(1^top_n cdot R_t, epsilon_1)} 
    &amp;hspace{5mm}U_t leftarrow frac{G_t}{max(sqrt{widehat{V}_t}, epsilon_1)} 
end{aligned}
           </annotation>
</semantics>
</math> -->
R t ← β ^ 2 t R t − 1 + ( 1 − β ^ 2 t ) ( G t ⊙ G t + 1 n ⋅ 1 m ⊤ ) ⋅ 1 m C t ← β ^ 2 t C t − 1 + ( 1 − β ^ 2 t ) 1 n ⊤ ⋅ ( G t ⊙ G t + 1 n ⋅ 1 m ⊤ ) V ^ t ← R t ⋅ C t m a x ( 1 n ⊤ ⋅ R t , ϵ 1 ) U t ← G t m a x ( V ^ t , ϵ 1 ) begin{aligned}
 &hspace{5mm}R_t leftarrow widehat{beta}_{2_t}R_{t-1}+
 (1-widehat{beta}_{2_t})(G_t odot G_t + 1_n cdot 1^top_m) cdot 1_m 
 &hspace{5mm}C_t leftarrow widehat{beta}_{2_t}C_{t-1}+
 (1-widehat{beta}_{2_t}) 1^top_n cdot (G_t odot G_t + 1_n cdot 1^top_m) 
 &hspace{5mm}widehat{V}_t leftarrow
 frac{R_t cdot C_t}{max(1^top_n cdot R_t, epsilon_1)} 
 &hspace{5mm}U_t leftarrow frac{G_t}{max(sqrt{widehat{V}_t}, epsilon_1)} 
end{aligned}

​ R t ​ ← β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ R t − 1 ​ + ( 1 − β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ ) ( G t ​ ⊙ G t ​ + 1 n ​ ⋅ 1 m ⊤ ​ ) ⋅ 1 m ​ C t ​ ← β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ C t − 1 ​ + ( 1 − β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ ) 1 n ⊤ ​ ⋅ ( G t ​ ⊙ G t ​ + 1 n ​ ⋅ 1 m ⊤ ​ ) V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ ← ma x ( 1 n ⊤ ​ ⋅ R t ​ , ϵ 1 ​ ) R t ​ ⋅ C t ​ ​ U t ​ ← ma x ( V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ , ϵ 1 ​ ) G t ​ ​ ​

where the row and column factors of gradient squared <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              R
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            R_t
           </annotation>
</semantics>
</math> -->R t R_tR t ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              C
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            C_t
           </annotation>
</semantics>
</math> -->C t C_tC t ​  are left alone, and we apply <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              ϵ
             </mi>
<mn>
              1
             </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            epsilon_1
           </annotation>
</semantics>
</math> -->ϵ 1 epsilon_1ϵ 1 ​  at the final calculation of
the variance estimate <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mover accent="true">
<mi>
               V
              </mi>
<mo stretchy="true">
               ^
              </mo>
</mover>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            widehat{V}_t
           </annotation>
</semantics>
</math> -->V ^ t widehat{V}_tV ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​  and for the update <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              U
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            U_t
           </annotation>
</semantics>
</math> -->U t U_tU t ​  . 

This is in contrast to Noam Shazeer and Mitchell Stern and other frameworks which
apply <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              ϵ
             </mi>
<mn>
              1
             </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            epsilon_1
           </annotation>
</semantics>
</math> -->ϵ 1 epsilon_1ϵ 1 ​  to both row and column factors of the squared gradient, but
not in the calculations after: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                  R
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<msub>
<mover accent="true">
<mi>
                   β
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<msub>
<mn>
                   2
                  </mn>
<mi>
                   t
                  </mi>
</msub>
</msub>
<msub>
<mi>
                  R
                 </mi>
<mrow>
<mi>
                   t
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msub>
<mo>
                 +
                </mo>
<mo stretchy="false">
                 (
                </mo>
<mn>
                 1
                </mn>
<mo>
                 −
                </mo>
<msub>
<mover accent="true">
<mi>
                   β
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<msub>
<mn>
                   2
                  </mn>
<mi>
                   t
                  </mi>
</msub>
</msub>
<mo stretchy="false">
                 )
                </mo>
<mo stretchy="false">
                 (
                </mo>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ⊙
                </mo>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 +
                </mo>
<msub>
<mi>
                  ϵ
                 </mi>
<mn>
                  1
                 </mn>
</msub>
<msub>
<mn>
                  1
                 </mn>
<mi>
                  n
                 </mi>
</msub>
<mo>
                 ⋅
                </mo>
<msubsup>
<mn>
                  1
                 </mn>
<mi>
                  m
                 </mi>
<mi mathvariant="normal">
                  ⊤
                 </mi>
</msubsup>
<mo stretchy="false">
                 )
                </mo>
<mo>
                 ⋅
                </mo>
<msub>
<mn>
                  1
                 </mn>
<mi>
                  m
                 </mi>
</msub>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                  C
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<msub>
<mover accent="true">
<mi>
                   β
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<msub>
<mn>
                   2
                  </mn>
<mi>
                   t
                  </mi>
</msub>
</msub>
<msub>
<mi>
                  C
                 </mi>
<mrow>
<mi>
                   t
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msub>
<mo>
                 +
                </mo>
<mo stretchy="false">
                 (
                </mo>
<mn>
                 1
                </mn>
<mo>
                 −
                </mo>
<msub>
<mover accent="true">
<mi>
                   β
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<msub>
<mn>
                   2
                  </mn>
<mi>
                   t
                  </mi>
</msub>
</msub>
<mo stretchy="false">
                 )
                </mo>
<msubsup>
<mn>
                  1
                 </mn>
<mi>
                  n
                 </mi>
<mi mathvariant="normal">
                  ⊤
                 </mi>
</msubsup>
<mo>
                 ⋅
                </mo>
<mo stretchy="false">
                 (
                </mo>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ⊙
                </mo>
<msub>
<mi>
                  G
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 +
                </mo>
<msub>
<mi>
                  ϵ
                 </mi>
<mn>
                  1
                 </mn>
</msub>
<msub>
<mn>
                  1
                 </mn>
<mi>
                  n
                 </mi>
</msub>
<mo>
                 ⋅
                </mo>
<msubsup>
<mn>
                  1
                 </mn>
<mi>
                  m
                 </mi>
<mi mathvariant="normal">
                  ⊤
                 </mi>
</msubsup>
<mo stretchy="false">
                 )
                </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mover accent="true">
<mi>
                   V
                  </mi>
<mo stretchy="true">
                   ^
                  </mo>
</mover>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<mfrac>
<mrow>
<msub>
<mi>
                    R
                   </mi>
<mi>
                    t
                   </mi>
</msub>
<mo>
                   ⋅
                  </mo>
<msub>
<mi>
                    C
                   </mi>
<mi>
                    t
                   </mi>
</msub>
</mrow>
<mrow>
<msubsup>
<mn>
                    1
                   </mn>
<mi>
                    n
                   </mi>
<mi mathvariant="normal">
                    ⊤
                   </mi>
</msubsup>
<mo>
                   ⋅
                  </mo>
<msub>
<mi>
                    R
                   </mi>
<mi>
                    t
                   </mi>
</msub>
</mrow>
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mspace width="1.4226em">
</mspace>
<msub>
<mi>
                  U
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ←
                </mo>
<mfrac>
<msub>
<mi>
                   G
                  </mi>
<mi>
                   t
                  </mi>
</msub>
<msqrt>
<msub>
<mover accent="true">
<mi>
                     V
                    </mi>
<mo stretchy="true">
                     ^
                    </mo>
</mover>
<mi>
                    t
                   </mi>
</msub>
</msqrt>
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{aligned}
    &amp;hspace{5mm}R_t leftarrow widehat{beta}_{2_t}R_{t-1}+
                (1-widehat{beta}_{2_t})(G_t odot G_t + epsilon_1 1_n cdot 1^top_m) cdot 1_m 
    &amp;hspace{5mm}C_t leftarrow widehat{beta}_{2_t}C_{t-1}+
                (1-widehat{beta}_{2_t}) 1^top_n cdot (G_t odot G_t + epsilon_1 1_n cdot 1^top_m) 
    &amp;hspace{5mm}widehat{V}_t leftarrow frac{R_t cdot C_t}{1^top_n cdot R_t} 
    &amp;hspace{5mm}U_t leftarrow frac{G_t}{sqrt{widehat{V}_t}} 
end{aligned}
           </annotation>
</semantics>
</math> -->
R t ← β ^ 2 t R t − 1 + ( 1 − β ^ 2 t ) ( G t ⊙ G t + ϵ 1 1 n ⋅ 1 m ⊤ ) ⋅ 1 m C t ← β ^ 2 t C t − 1 + ( 1 − β ^ 2 t ) 1 n ⊤ ⋅ ( G t ⊙ G t + ϵ 1 1 n ⋅ 1 m ⊤ ) V ^ t ← R t ⋅ C t 1 n ⊤ ⋅ R t U t ← G t V ^ t begin{aligned}
 &hspace{5mm}R_t leftarrow widehat{beta}_{2_t}R_{t-1}+
 (1-widehat{beta}_{2_t})(G_t odot G_t + epsilon_1 1_n cdot 1^top_m) cdot 1_m 
 &hspace{5mm}C_t leftarrow widehat{beta}_{2_t}C_{t-1}+
 (1-widehat{beta}_{2_t}) 1^top_n cdot (G_t odot G_t + epsilon_1 1_n cdot 1^top_m) 
 &hspace{5mm}widehat{V}_t leftarrow frac{R_t cdot C_t}{1^top_n cdot R_t} 
 &hspace{5mm}U_t leftarrow frac{G_t}{sqrt{widehat{V}_t}} 
end{aligned}

​ R t ​ ← β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ R t − 1 ​ + ( 1 − β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ ) ( G t ​ ⊙ G t ​ + ϵ 1 ​ 1 n ​ ⋅ 1 m ⊤ ​ ) ⋅ 1 m ​ C t ​ ← β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ C t − 1 ​ + ( 1 − β ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ 2 t ​ ​ ) 1 n ⊤ ​ ⋅ ( G t ​ ⊙ G t ​ + ϵ 1 ​ 1 n ​ ⋅ 1 m ⊤ ​ ) V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ ← 1 n ⊤ ​ ⋅ R t ​ R t ​ ⋅ C t ​ ​ U t ​ ← V ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)t ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ G t ​ ​ ​

add_param_group ( *param_group* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L1066) 
:   Add a param group to the [`Optimizer`](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  s *param_groups* . 

This can be useful when fine tuning a pre-trained network as frozen layers can be made
trainable and added to the [`Optimizer`](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  as training progresses. 

Parameters
: **param_group** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – Specifies what Tensors should be optimized along with group
specific optimization options.

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L865) 
:   Load the optimizer state. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – optimizer state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.Adafactor.state_dict "torch.optim.Adafactor.state_dict")  .

Warning 

Make sure this method is called after initializing [`torch.optim.lr_scheduler.LRScheduler`](torch.optim.lr_scheduler.LRScheduler.html#torch.optim.lr_scheduler.LRScheduler "torch.optim.lr_scheduler.LRScheduler")  ,
as calling it beforehand will overwrite the loaded learning rates.

Note 

The names of the parameters (if they exist under the “param_names” key of each param group
in [`state_dict()`](#torch.optim.Adafactor.state_dict "torch.optim.Adafactor.state_dict")  ) will not affect the loading process.
To use the parameters’ names for custom cases (such as when the parameters in the loaded state dict
differ from those initialized in the optimizer),
a custom `register_load_state_dict_pre_hook`  should be implemented to adapt the loaded dict
accordingly.
If `param_names`  exist in loaded state dict `param_groups`  they will be saved and override
the current names, if present, in the optimizer state. If they do not exist in loaded state dict,
the optimizer `param_names`  will remain unchanged.

Example 

```
>>> model = torch.nn.Linear(10, 10)
>>> optim = torch.optim.SGD(model.parameters(), lr=3e-4)
>>> scheduler1 = torch.optim.lr_scheduler.LinearLR(
...     optim,
...     start_factor=0.1,
...     end_factor=1,
...     total_iters=20,
... )
>>> scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
...     optim,
...     T_max=80,
...     eta_min=3e-5,
... )
>>> lr = torch.optim.lr_scheduler.SequentialLR(
...     optim,
...     schedulers=[scheduler1, scheduler2],
...     milestones=[20],
... )
>>> lr.load_state_dict(torch.load("./save_seq.pt"))
>>> # now load the optimizer checkpoint after loading the LRScheduler
>>> optim.load_state_dict(torch.load("./save_optim.pt"))

```

register_load_state_dict_post_hook ( *hook*  , *prepend = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L829) 
:   Register a load_state_dict post-hook which will be called after [`load_state_dict()`](torch.optim.Optimizer.load_state_dict.html#torch.optim.Optimizer.load_state_dict "torch.optim.Optimizer.load_state_dict")  is called. It should have the
following signature: 

```
hook(optimizer) -> None

```

The `optimizer`  argument is the optimizer instance being used. 

The hook will be called with argument `self`  after calling `load_state_dict`  on `self`  . The registered hook can be used to
perform post-processing after `load_state_dict`  has loaded the `state_dict`  . 

Parameters
:   * **hook** ( *Callable*  ) – The user defined hook to be registered.
* **prepend** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If True, the provided post `hook`  will be fired before
all the already registered post-hooks on `load_state_dict`  . Otherwise,
the provided `hook`  will be fired after all the already registered
post-hooks. (default: False)

Returns
:   a handle that can be used to remove the added hook by calling `handle.remove()`

Return type
:   `torch.utils.hooks.RemoveableHandle`

register_load_state_dict_pre_hook ( *hook*  , *prepend = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L790) 
:   Register a load_state_dict pre-hook which will be called before [`load_state_dict()`](torch.optim.Optimizer.load_state_dict.html#torch.optim.Optimizer.load_state_dict "torch.optim.Optimizer.load_state_dict")  is called. It should have the
following signature: 

```
hook(optimizer, state_dict) -> state_dict or None

```

The `optimizer`  argument is the optimizer instance being used and the `state_dict`  argument is a shallow copy of the `state_dict`  the user
passed in to `load_state_dict`  . The hook may modify the state_dict inplace
or optionally return a new one. If a state_dict is returned, it will be used
to be loaded into the optimizer. 

The hook will be called with argument `self`  and `state_dict`  before
calling `load_state_dict`  on `self`  . The registered hook can be used to
perform pre-processing before the `load_state_dict`  call is made. 

Parameters
:   * **hook** ( *Callable*  ) – The user defined hook to be registered.
* **prepend** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If True, the provided pre `hook`  will be fired before
all the already registered pre-hooks on `load_state_dict`  . Otherwise,
the provided `hook`  will be fired after all the already registered
pre-hooks. (default: False)

Returns
:   a handle that can be used to remove the added hook by calling `handle.remove()`

Return type
:   `torch.utils.hooks.RemoveableHandle`

register_state_dict_post_hook ( *hook*  , *prepend = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L632) 
:   Register a state dict post-hook which will be called after [`state_dict()`](torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict "torch.optim.Optimizer.state_dict")  is called. 

It should have the following signature: 

```
hook(optimizer, state_dict) -> state_dict or None

```

The hook will be called with arguments `self`  and `state_dict`  after generating
a `state_dict`  on `self`  . The hook may modify the state_dict inplace or optionally
return a new one. The registered hook can be used to perform post-processing
on the `state_dict`  before it is returned. 

Parameters
:   * **hook** ( *Callable*  ) – The user defined hook to be registered.
* **prepend** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If True, the provided post `hook`  will be fired before
all the already registered post-hooks on `state_dict`  . Otherwise,
the provided `hook`  will be fired after all the already registered
post-hooks. (default: False)

Returns
:   a handle that can be used to remove the added hook by calling `handle.remove()`

Return type
:   `torch.utils.hooks.RemoveableHandle`

register_state_dict_pre_hook ( *hook*  , *prepend = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L600) 
:   Register a state dict pre-hook which will be called before [`state_dict()`](torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict "torch.optim.Optimizer.state_dict")  is called. 

It should have the following signature: 

```
hook(optimizer) -> None

```

The `optimizer`  argument is the optimizer instance being used.
The hook will be called with argument `self`  before calling `state_dict`  on `self`  .
The registered hook can be used to perform pre-processing before the `state_dict`  call is made. 

Parameters
:   * **hook** ( *Callable*  ) – The user defined hook to be registered.
* **prepend** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If True, the provided pre `hook`  will be fired before
all the already registered pre-hooks on `state_dict`  . Otherwise,
the provided `hook`  will be fired after all the already registered
pre-hooks. (default: False)

Returns
:   a handle that can be used to remove the added hook by calling `handle.remove()`

Return type
:   `torch.utils.hooks.RemoveableHandle`

register_step_post_hook ( *hook* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L579) 
:   Register an optimizer step post hook which will be called after optimizer step. 

It should have the following signature: 

```
hook(optimizer, args, kwargs) -> None

```

The `optimizer`  argument is the optimizer instance being used. 

Parameters
: **hook** ( *Callable*  ) – The user defined hook to be registered.

Returns
:   a handle that can be used to remove the added hook by calling `handle.remove()`

Return type
:   `torch.utils.hooks.RemovableHandle`

register_step_pre_hook ( *hook* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L556) 
:   Register an optimizer step pre hook which will be called before optimizer step. 

It should have the following signature: 

```
hook(optimizer, args, kwargs) -> None or modified args and kwargs

```

The `optimizer`  argument is the optimizer instance being used. If
args and kwargs are modified by the pre-hook, then the transformed
values are returned as a tuple containing the new_args and new_kwargs. 

Parameters
: **hook** ( *Callable*  ) – The user defined hook to be registered.

Returns
:   a handle that can be used to remove the added hook by calling `handle.remove()`

Return type
:   `torch.utils.hooks.RemovableHandle`

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L666) 
:   Return the state of the optimizer as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains two entries: 

* `state`  : a Dict holding current optimization state. Its content
:   differs between optimizer classes, but some common characteristics
hold. For example, state is saved per parameter, and the parameter
itself is NOT saved. `state`  is a Dictionary mapping parameter ids
to a Dict with state corresponding to each parameter.
* `param_groups`  : a List containing all parameter groups where each
:   parameter group is a Dict. Each parameter group contains metadata
specific to the optimizer, such as learning rate and weight decay,
as well as a List of parameter IDs of the parameters in the group.
If a param group was initialized with `named_parameters()`  the names
content will also be saved in the state dict.

NOTE: The parameter IDs may look like indices but they are just IDs
associating state with param_group. When loading from a state_dict,
the optimizer will zip the param_group `params`  (int IDs) and the
optimizer `param_groups`  (actual `nn.Parameter`  s) in order to
match state WITHOUT additional verification. 

A returned state dict might look something like: 

```
{
    'state': {
        0: {'momentum_buffer': tensor(...), ...},
        1: {'momentum_buffer': tensor(...), ...},
        2: {'momentum_buffer': tensor(...), ...},
        3: {'momentum_buffer': tensor(...), ...}
    },
    'param_groups': [
        {
            'lr': 0.01,
            'weight_decay': 0,
            ...
            'params': [0]
            'param_names' ['param0']  (optional)
        },
        {
            'lr': 0.001,
            'weight_decay': 0.5,
            ...
            'params': [1, 2, 3]
            'param_names': ['param1', 'layer.weight', 'layer.bias'] (optional)
        }
    ]
}

```

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( *closure = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/_adafactor.py#L121) 
:   Perform a single optimization step. 

Parameters
: **closure** ( *Callable* *,* *optional*  ) – A closure that reevaluates the model
and returns the loss.

zero_grad ( *set_to_none = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L996) 
:   Reset the gradients of all optimized [`torch.Tensor`](../tensors.html#torch.Tensor "torch.Tensor")  s. 

Parameters
: **set_to_none** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – instead of setting to zero, set the grads to None.
This will in general have lower memory footprint, and can modestly improve performance.
However, it changes certain behaviors. For example:
1. When the user tries to access a gradient and perform manual ops on it,
a None attribute or a Tensor full of 0s will behave differently.
2. If the user requests `zero_grad(set_to_none=True)`  followed by a backward pass, `.grad`  s
are guaranteed to be None for params that did not receive a gradient.
3. `torch.optim`  optimizers have a different behavior if the gradient is 0 or None
(in one case it does the step with a gradient of 0 and in the other it skips
the step altogether).

