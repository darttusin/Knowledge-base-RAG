RAdam 
==============================================

*class* torch.optim. RAdam ( *params*  , *lr = 0.001*  , *betas = (0.9, 0.999)*  , *eps = 1e-08*  , *weight_decay = 0*  , *decoupled_weight_decay = False*  , *** , *foreach = None*  , *maximize = False*  , *capturable = False*  , *differentiable = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/radam.py#L31) 
:   Implements RAdam algorithm. 

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
<msub>
<mi>
                 β
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
                 β
                </mi>
<mn>
                 2
                </mn>
</msub>
<mtext>
                (betas)
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
<mtext>
</mtext>
<mi>
                λ
               </mi>
<mtext>
                (weightdecay)
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
<mspace width="3.6989em">
</mspace>
<mi>
                ϵ
               </mi>
<mtext>
                (epsilon)
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext mathvariant="italic">
                decoupled_weight_decay
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
<msub>
<mi>
                 m
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
                ( first moment)
               </mtext>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 v
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
                ( second moment)
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
<mspace width="5.1215em">
</mspace>
<msub>
<mi>
                 ρ
                </mi>
<mi mathvariant="normal">
                 ∞
                </mi>
</msub>
<mo>
                ←
               </mo>
<mn>
                2
               </mn>
<mi mathvariant="normal">
                /
               </mi>
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
<mi>
                 β
                </mi>
<mn>
                 2
                </mn>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
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
<mspace width="1.7072em">
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
<mspace width="3.4143em">
</mspace>
<msub>
<mi>
                 g
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
<mspace width="1.7072em">
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
<mspace width="3.4143em">
</mspace>
<msub>
<mi>
                 g
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
<mspace width="1.7072em">
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
<mspace width="1.7072em">
</mspace>
<mtext mathvariant="bold">
                if
               </mtext>
<mtext>
</mtext>
<mi>
                λ
               </mi>
<mo mathvariant="normal">
                ≠
               </mo>
<mn>
                0
               </mn>
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
<mspace width="3.4143em">
</mspace>
<mtext mathvariant="bold">
                if
               </mtext>
<mtext>
</mtext>
<mtext mathvariant="italic">
                decoupled_weight_decay
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
<mspace width="5.1215em">
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
<mi>
                 t
                </mi>
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
<mspace width="3.4143em">
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
<mspace width="5.1215em">
</mspace>
<msub>
<mi>
                 g
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
                 g
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                +
               </mo>
<mi>
                λ
               </mi>
<msub>
<mi>
                 θ
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
<mspace width="1.7072em">
</mspace>
<msub>
<mi>
                 m
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
                 β
                </mi>
<mn>
                 1
                </mn>
</msub>
<msub>
<mi>
                 m
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
<mi>
                 β
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo stretchy="false">
                )
               </mo>
<msub>
<mi>
                 g
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
<mspace width="1.7072em">
</mspace>
<msub>
<mi>
                 v
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
                 β
                </mi>
<mn>
                 2
                </mn>
</msub>
<msub>
<mi>
                 v
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
<mi>
                 β
                </mi>
<mn>
                 2
                </mn>
</msub>
<mo stretchy="false">
                )
               </mo>
<msubsup>
<mi>
                 g
                </mi>
<mi>
                 t
                </mi>
<mn>
                 2
                </mn>
</msubsup>
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
<mspace width="1.7072em">
</mspace>
<mover accent="true">
<msub>
<mi>
                  m
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo stretchy="true">
                 ^
                </mo>
</mover>
<mo>
                ←
               </mo>
<msub>
<mi>
                 m
                </mi>
<mi>
                 t
                </mi>
</msub>
<mi mathvariant="normal">
                /
               </mi>
<mo fence="false" maxsize="1.2em" minsize="1.2em" stretchy="true">
                (
               </mo>
<mn>
                1
               </mn>
<mo>
                −
               </mo>
<msubsup>
<mi>
                 β
                </mi>
<mn>
                 1
                </mn>
<mi>
                 t
                </mi>
</msubsup>
<mo fence="false" maxsize="1.2em" minsize="1.2em" stretchy="true">
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
<mspace width="1.7072em">
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
<msub>
<mi>
                 ρ
                </mi>
<mi mathvariant="normal">
                 ∞
                </mi>
</msub>
<mo>
                −
               </mo>
<mn>
                2
               </mn>
<mi>
                t
               </mi>
<msubsup>
<mi>
                 β
                </mi>
<mn>
                 2
                </mn>
<mi>
                 t
                </mi>
</msubsup>
<mi mathvariant="normal">
                /
               </mi>
<mo fence="false" maxsize="1.2em" minsize="1.2em" stretchy="true">
                (
               </mo>
<mn>
                1
               </mn>
<mo>
                −
               </mo>
<msubsup>
<mi>
                 β
                </mi>
<mn>
                 2
                </mn>
<mi>
                 t
                </mi>
</msubsup>
<mo fence="false" maxsize="1.2em" minsize="1.2em" stretchy="true">
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
<mspace width="1.7072em">
</mspace>
<mtext mathvariant="bold">
                if
               </mtext>
<mtext>
</mtext>
<msub>
<mi>
                 ρ
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                &gt;
               </mo>
<mn>
                5
               </mn>
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
<mspace width="3.4143em">
</mspace>
<msub>
<mi>
                 l
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<mfrac>
<msqrt>
<mrow>
<mo stretchy="false">
                   (
                  </mo>
<mn>
                   1
                  </mn>
<mo>
                   −
                  </mo>
<msubsup>
<mi>
                    β
                   </mi>
<mn>
                    2
                   </mn>
<mi>
                    t
                   </mi>
</msubsup>
<mo stretchy="false">
                   )
                  </mo>
</mrow>
</msqrt>
<mrow>
<msqrt>
<msub>
<mi>
                    v
                   </mi>
<mi>
                    t
                   </mi>
</msub>
</msqrt>
<mo>
                  +
                 </mo>
<mi>
                  ϵ
                 </mi>
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
<mspace width="3.4143em">
</mspace>
<msub>
<mi>
                 r
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<msqrt>
<mfrac>
<mrow>
<mo stretchy="false">
                   (
                  </mo>
<msub>
<mi>
                    ρ
                   </mi>
<mi>
                    t
                   </mi>
</msub>
<mo>
                   −
                  </mo>
<mn>
                   4
                  </mn>
<mo stretchy="false">
                   )
                  </mo>
<mo stretchy="false">
                   (
                  </mo>
<msub>
<mi>
                    ρ
                   </mi>
<mi>
                    t
                   </mi>
</msub>
<mo>
                   −
                  </mo>
<mn>
                   2
                  </mn>
<mo stretchy="false">
                   )
                  </mo>
<msub>
<mi>
                    ρ
                   </mi>
<mi mathvariant="normal">
                    ∞
                   </mi>
</msub>
</mrow>
<mrow>
<mo stretchy="false">
                   (
                  </mo>
<msub>
<mi>
                    ρ
                   </mi>
<mi mathvariant="normal">
                    ∞
                   </mi>
</msub>
<mo>
                   −
                  </mo>
<mn>
                   4
                  </mn>
<mo stretchy="false">
                   )
                  </mo>
<mo stretchy="false">
                   (
                  </mo>
<msub>
<mi>
                    ρ
                   </mi>
<mi mathvariant="normal">
                    ∞
                   </mi>
</msub>
<mo>
                   −
                  </mo>
<mn>
                   2
                  </mn>
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
</mfrac>
</msqrt>
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
<mspace width="3.4143em">
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
<mi>
                 t
                </mi>
</msub>
<mo>
                −
               </mo>
<mi>
                γ
               </mi>
<mover accent="true">
<msub>
<mi>
                  m
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo stretchy="true">
                 ^
                </mo>
</mover>
<msub>
<mi>
                 r
                </mi>
<mi>
                 t
                </mi>
</msub>
<msub>
<mi>
                 l
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
<mspace width="1.7072em">
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
<mspace width="3.4143em">
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
<mi>
                 t
                </mi>
</msub>
<mo>
                −
               </mo>
<mi>
                γ
               </mi>
<mover accent="true">
<msub>
<mi>
                  m
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo stretchy="true">
                 ^
                </mo>
</mover>
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
      &amp;textbf{input} : gamma text{ (lr)}, : beta_1, beta_2
          text{ (betas)}, : theta_0 text{ (params)}, :f(theta) text{ (objective)}, :
          lambda text{ (weightdecay)}, :textit{maximize} 
      &amp;hspace{13mm} epsilon text{ (epsilon)}, textit{decoupled_weight_decay} 
      &amp;textbf{initialize} :  m_0 leftarrow 0 text{ ( first moment)},
          v_0 leftarrow 0 text{ ( second moment)}, 
      &amp;hspace{18mm} rho_{infty} leftarrow 2/(1-beta_2) -1 [-1.ex]
      &amp;rule{110mm}{0.4pt}  
      &amp;textbf{for} : t=1 : textbf{to} : ldots : textbf{do} 
      &amp;hspace{6mm}textbf{if} : textit{maximize}:                                       
      &amp;hspace{12mm}g_t leftarrow -nabla_{theta} f_t (theta_{t-1}) 
      &amp;hspace{6mm}textbf{else} 
      &amp;hspace{12mm}g_t leftarrow nabla_{theta} f_t (theta_{t-1}) 
      &amp;hspace{6mm} theta_t leftarrow theta_{t-1} 
      &amp;hspace{6mm} textbf{if} : lambda neq 0 
      &amp;hspace{12mm}textbf{if} : textit{decoupled_weight_decay} 
      &amp;hspace{18mm} theta_t leftarrow theta_{t} - gamma lambda theta_{t} 
      &amp;hspace{12mm}textbf{else} 
      &amp;hspace{18mm} g_t leftarrow g_t + lambda theta_{t} 
      &amp;hspace{6mm}m_t leftarrow beta_1 m_{t-1} + (1 - beta_1) g_t 
      &amp;hspace{6mm}v_t leftarrow beta_2 v_{t-1} + (1-beta_2) g^2_t 
      &amp;hspace{6mm}widehat{m_t} leftarrow m_t/big(1-beta_1^t big) 
      &amp;hspace{6mm}rho_t leftarrow rho_{infty} -
          2 t beta^t_2 /big(1-beta_2^t big) [0.1.ex]
      &amp;hspace{6mm}textbf{if} : rho_t &gt; 5 
      &amp;hspace{12mm} l_t leftarrow frac{sqrt{ (1-beta^t_2) }}{ sqrt{v_t} +epsilon  } 
      &amp;hspace{12mm} r_t leftarrow
sqrt{frac{(rho_t-4)(rho_t-2)rho_{infty}}{(rho_{infty}-4)(rho_{infty}-2) rho_t}} 
      &amp;hspace{12mm}theta_t leftarrow theta_t - gamma widehat{m_t} r_t l_t 
      &amp;hspace{6mm}textbf{else} 
      &amp;hspace{12mm}theta_t leftarrow theta_t - gamma widehat{m_t} 
      &amp;rule{110mm}{0.4pt} [-1.ex]
      &amp;bf{return} :  theta_t [-1.ex]
      &amp;rule{110mm}{0.4pt} [-1.ex]
 end{aligned}
          </annotation>
</semantics>
</math> -->
input : γ (lr) , β 1 , β 2 (betas) , θ 0 (params) , f ( θ ) (objective) , λ (weightdecay) , maximize ϵ (epsilon) , decoupled_weight_decay initialize : m 0 ← 0 ( first moment) , v 0 ← 0 ( second moment) , ρ ∞ ← 2 / ( 1 − β 2 ) − 1 for t = 1 to … do if maximize : g t ← − ∇ θ f t ( θ t − 1 ) else g t ← ∇ θ f t ( θ t − 1 ) θ t ← θ t − 1 if λ ≠ 0 if decoupled_weight_decay θ t ← θ t − γ λ θ t else g t ← g t + λ θ t m t ← β 1 m t − 1 + ( 1 − β 1 ) g t v t ← β 2 v t − 1 + ( 1 − β 2 ) g t 2 m t ^ ← m t / ( 1 − β 1 t ) ρ t ← ρ ∞ − 2 t β 2 t / ( 1 − β 2 t ) if ρ t > 5 l t ← ( 1 − β 2 t ) v t + ϵ r t ← ( ρ t − 4 ) ( ρ t − 2 ) ρ ∞ ( ρ ∞ − 4 ) ( ρ ∞ − 2 ) ρ t θ t ← θ t − γ m t ^ r t l t else θ t ← θ t − γ m t ^ r e t u r n θ t begin{aligned}
 &rule{110mm}{0.4pt} 
 &textbf{input} : gamma text{ (lr)}, : beta_1, beta_2
 text{ (betas)}, : theta_0 text{ (params)}, :f(theta) text{ (objective)}, :
 lambda text{ (weightdecay)}, :textit{maximize} 
 &hspace{13mm} epsilon text{ (epsilon)}, textit{decoupled_weight_decay} 
 &textbf{initialize} : m_0 leftarrow 0 text{ ( first moment)},
 v_0 leftarrow 0 text{ ( second moment)}, 
 &hspace{18mm} rho_{infty} leftarrow 2/(1-beta_2) -1 [-1.ex]
 &rule{110mm}{0.4pt} 
 &textbf{for} : t=1 : textbf{to} : ldots : textbf{do} 
 &hspace{6mm}textbf{if} : textit{maximize}: 
 &hspace{12mm}g_t leftarrow -nabla_{theta} f_t (theta_{t-1}) 
 &hspace{6mm}textbf{else} 
 &hspace{12mm}g_t leftarrow nabla_{theta} f_t (theta_{t-1}) 
 &hspace{6mm} theta_t leftarrow theta_{t-1} 
 &hspace{6mm} textbf{if} : lambda neq 0 
 &hspace{12mm}textbf{if} : textit{decoupled_weight_decay} 
 &hspace{18mm} theta_t leftarrow theta_{t} - gamma lambda theta_{t} 
 &hspace{12mm}textbf{else} 
 &hspace{18mm} g_t leftarrow g_t + lambda theta_{t} 
 &hspace{6mm}m_t leftarrow beta_1 m_{t-1} + (1 - beta_1) g_t 
 &hspace{6mm}v_t leftarrow beta_2 v_{t-1} + (1-beta_2) g^2_t 
 &hspace{6mm}widehat{m_t} leftarrow m_t/big(1-beta_1^t big) 
 &hspace{6mm}rho_t leftarrow rho_{infty} -
 2 t beta^t_2 /big(1-beta_2^t big) [0.1.ex]
 &hspace{6mm}textbf{if} : rho_t > 5 
 &hspace{12mm} l_t leftarrow frac{sqrt{ (1-beta^t_2) }}{ sqrt{v_t} +epsilon } 
 &hspace{12mm} r_t leftarrow
sqrt{frac{(rho_t-4)(rho_t-2)rho_{infty}}{(rho_{infty}-4)(rho_{infty}-2) rho_t}} 
 &hspace{12mm}theta_t leftarrow theta_t - gamma widehat{m_t} r_t l_t 
 &hspace{6mm}textbf{else} 
 &hspace{12mm}theta_t leftarrow theta_t - gamma widehat{m_t} 
 &rule{110mm}{0.4pt} [-1.ex]
 &bf{return} : theta_t [-1.ex]
 &rule{110mm}{0.4pt} [-1.ex]
 end{aligned}

​ input : γ (lr) , β 1 ​ , β 2 ​ (betas) , θ 0 ​ (params) , f ( θ ) (objective) , λ (weightdecay) , maximize ϵ (epsilon) , decoupled_weight_decay initialize : m 0 ​ ← 0 ( first moment) , v 0 ​ ← 0 ( second moment) , ρ ∞ ​ ← 2/ ( 1 − β 2 ​ ) − 1 for t = 1 to … do if maximize : g t ​ ← − ∇ θ ​ f t ​ ( θ t − 1 ​ ) else g t ​ ← ∇ θ ​ f t ​ ( θ t − 1 ​ ) θ t ​ ← θ t − 1 ​ if λ  = 0 if decoupled_weight_decay θ t ​ ← θ t ​ − γλ θ t ​ else g t ​ ← g t ​ + λ θ t ​ m t ​ ← β 1 ​ m t − 1 ​ + ( 1 − β 1 ​ ) g t ​ v t ​ ← β 2 ​ v t − 1 ​ + ( 1 − β 2 ​ ) g t 2 ​ m t ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ ← m t ​ / ( 1 − β 1 t ​ ) ρ t ​ ← ρ ∞ ​ − 2 t β 2 t ​ / ( 1 − β 2 t ​ ) if ρ t ​ > 5 l t ​ ← v t ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ + ϵ ( 1 − β 2 t ​ ) ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ ​ r t ​ ← ( ρ ∞ ​ − 4 ) ( ρ ∞ ​ − 2 ) ρ t ​ ( ρ t ​ − 4 ) ( ρ t ​ − 2 ) ρ ∞ ​ ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMzI0MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNDczLDI3OTMKYzMzOS4zLC0xNzk5LjMsNTA5LjMsLTI3MDAsNTEwLC0yNzAyIGwwIC0wCmMzLjMsLTcuMyw5LjMsLTExLDE4LC0xMSBINDAwMDAwdjQwSDEwMTcuNwpzLTkwLjUsNDc4LC0yNzYuMiwxNDY2Yy0xODUuNyw5ODgsLTI3OS41LDE0ODMsLTI4MS41LDE0ODVjLTIsNiwtMTAsOSwtMjQsOQpjLTgsMCwtMTIsLTAuNywtMTIsLTJjMCwtMS4zLC01LjMsLTMyLC0xNiwtOTJjLTUwLjcsLTI5My4zLC0xMTkuNywtNjkzLjMsLTIwNywtMTIwMApjMCwtMS4zLC01LjMsOC43LC0xNiwzMGMtMTAuNywyMS4zLC0yMS4zLDQyLjcsLTMyLDY0cy0xNiwzMywtMTYsMzNzLTI2LC0yNiwtMjYsLTI2CnM3NiwtMTUzLDc2LC0xNTNzNzcsLTE1MSw3NywtMTUxYzAuNywwLjcsMzUuNywyMDIsMTA1LDYwNGM2Ny4zLDQwMC43LDEwMiw2MDIuNywxMDQsCjYwNnpNMTAwMSA4MGg0MDAwMDB2NDBIMTAxNy43eiI+CjwvcGF0aD4KPC9zdmc+)​ θ t ​ ← θ t ​ − γ m t ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ r t ​ l t ​ else θ t ​ ← θ t ​ − γ m t ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMjRlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ibm9uZSIgdmlld2JveD0iMCAwIDEwNjIgMjM5IiB3aWR0aD0iMTAwJSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTUyOSAwaDVsNTE5IDExNWM1IDEgOSA1IDkgMTAgMCAxLTEgMi0xIDNsLTQgMjIKYy0xIDUtNSA5LTExIDloLTJMNTMyIDY3IDE5IDE1OWgtMmMtNSAwLTktNC0xMS05bC01LTIyYy0xLTYgMi0xMiA4LTEzeiI+CjwvcGF0aD4KPC9zdmc+)​ return θ t ​ ​

For further details regarding the algorithm we refer to [On the variance of the adaptive learning rate and beyond](https://arxiv.org/abs/1908.03265)  . 

This implementation provides an option to use either the original weight_decay implementation as in Adam
(where the weight_decay is applied to the gradient) or the one from AdamW (where weight_decay is applied
to the weight) through the decoupled_weight_decay option. When decoupled_weight_decay is set to False
(default), it uses the original Adam style weight decay, otherwise, it uses the AdamW style which
corresponds more closely to the [author’s implementation](https://github.com/LiyuanLucasLiu/RAdam)  in the RAdam paper. Further information
about decoupled weight decay can be found in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)  . 

Parameters
:   * **params** ( *iterable*  ) – iterable of parameters or named_parameters to optimize
or iterable of dicts defining parameter groups. When using named_parameters,
all parameters in all groups should be named
* **lr** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – learning rate (default: 1e-3)
* **betas** ( *Tuple* *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]* *,* *optional*  ) – coefficients used for computing
running averages of gradient and its square (default: (0.9, 0.999))
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – term added to the denominator to improve
numerical stability (default: 1e-8)
* **weight_decay** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – weight decay (L2 penalty) (default: 0)
* **decoupled_weight_decay** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to decouple the weight
decay as in AdamW to obtain RAdamW. If True, the algorithm does not
accumulate weight decay in the momentum nor variance. (default: False)
* **foreach** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether foreach implementation of optimizer
is used. If unspecified by the user (so foreach is None), we will try to use
foreach over the for-loop implementation on CUDA, since it is usually
significantly more performant. Note that the foreach implementation uses
~ sizeof(params) more peak memory than the for-loop version due to the intermediates
being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
parameters through the optimizer at a time or switch this flag to False (default: None)
* **maximize** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – maximize the objective with respect to the
params, instead of minimizing (default: False)
* **capturable** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether this instance is safe to
capture in a graph, whether for CUDA graphs or for torch.compile support.
Tensors are only capturable when on supported [accelerators](../torch.html#accelerators)  .
Passing True can impair ungraphed performance, so if you don’t intend to graph
capture this instance, leave it False (default: False)
* **differentiable** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether autograd should
occur through the optimizer step in training. Otherwise, the step()
function runs in a torch.no_grad() context. Setting to True can impair
performance, so leave it False if you don’t intend to run autograd
through this instance (default: False)

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
from a call to [`state_dict()`](#torch.optim.RAdam.state_dict "torch.optim.RAdam.state_dict")  .

Warning 

Make sure this method is called after initializing [`torch.optim.lr_scheduler.LRScheduler`](torch.optim.lr_scheduler.LRScheduler.html#torch.optim.lr_scheduler.LRScheduler "torch.optim.lr_scheduler.LRScheduler")  ,
as calling it beforehand will overwrite the loaded learning rates.

Note 

The names of the parameters (if they exist under the “param_names” key of each param group
in [`state_dict()`](#torch.optim.RAdam.state_dict "torch.optim.RAdam.state_dict")  ) will not affect the loading process.
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

step ( *closure = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/radam.py#L127) 
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

