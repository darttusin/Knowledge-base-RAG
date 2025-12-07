SGD 
==========================================

*class* torch.optim. SGD ( *params*  , *lr = 0.001*  , *momentum = 0*  , *dampening = 0*  , *weight_decay = 0*  , *nesterov = False*  , *** , *maximize = False*  , *foreach = None*  , *differentiable = False*  , *fused = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/sgd.py#L28) 
:   Implements stochastic gradient descent (optionally with momentum). 

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
                (weight decay)
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
<mspace width="3.6989em">
</mspace>
<mtext>
</mtext>
<mi>
                μ
               </mi>
<mtext>
                (momentum)
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
                (dampening)
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
</mtext>
<mrow>
<mtext>
</mtext>
<mtext mathvariant="italic">
                 nesterov,
                </mtext>
</mrow>
<mtext>
</mtext>
<mrow>
<mtext>
</mtext>
<mtext mathvariant="italic">
                 maximize
                </mtext>
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
<mspace width="1.4226em">
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
<mspace width="2.8453em">
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
<mi>
                μ
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
<mspace width="2.8453em">
</mspace>
<mtext mathvariant="bold">
                if
               </mtext>
<mtext>
</mtext>
<mi>
                t
               </mi>
<mo>
                &gt;
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
<mspace width="4.2679em">
</mspace>
<msub>
<mtext mathvariant="bold">
                 b
                </mtext>
<mi>
                 t
                </mi>
</msub>
<mo>
                ←
               </mo>
<mi>
                μ
               </mi>
<msub>
<mtext mathvariant="bold">
                 b
                </mtext>
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
<mi>
                τ
               </mi>
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
<mspace width="2.8453em">
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
<mspace width="4.2679em">
</mspace>
<msub>
<mtext mathvariant="bold">
                 b
                </mtext>
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
<mtext mathvariant="bold">
                if
               </mtext>
<mtext>
</mtext>
<mtext mathvariant="italic">
                nesterov
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
<mspace width="4.2679em">
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
                μ
               </mi>
<msub>
<mtext mathvariant="bold">
                 b
                </mtext>
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
<mspace width="2.8453em">
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
<mspace width="4.2679em">
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
<mtext mathvariant="bold">
                 b
                </mtext>
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
     &amp;textbf{input} : gamma text{ (lr)}, : theta_0 text{ (params)}, : f(theta)
         text{ (objective)}, : lambda text{ (weight decay)}, 
     &amp;hspace{13mm} :mu text{ (momentum)}, :tau text{ (dampening)},
     :textit{ nesterov,}:textit{ maximize} [-1.ex]
     &amp;rule{110mm}{0.4pt} 
     &amp;textbf{for} : t=1 : textbf{to} : ldots : textbf{do} 
     &amp;hspace{5mm}textbf{if} : textit{maximize}:                                       
     &amp;hspace{10mm}g_t leftarrow -nabla_{theta} f_t (theta_{t-1}) 
     &amp;hspace{5mm}textbf{else} 
     &amp;hspace{10mm}g_t leftarrow nabla_{theta} f_t (theta_{t-1}) 
     &amp;hspace{5mm}textbf{if} : lambda neq 0 
     &amp;hspace{10mm} g_t leftarrow g_t + lambda  theta_{t-1} 
     &amp;hspace{5mm}textbf{if} : mu neq 0 
     &amp;hspace{10mm}textbf{if} : t &gt; 1 
     &amp;hspace{15mm} textbf{b}_t leftarrow mu textbf{b}_{t-1} + (1-tau) g_t 
     &amp;hspace{10mm}textbf{else} 
     &amp;hspace{15mm} textbf{b}_t leftarrow g_t 
     &amp;hspace{10mm}textbf{if} : textit{nesterov} 
     &amp;hspace{15mm} g_t leftarrow g_{t} + mu textbf{b}_t 
     &amp;hspace{10mm}textbf{else} [-1.ex]
     &amp;hspace{15mm} g_t  leftarrow  textbf{b}_t 
     &amp;hspace{5mm}theta_t leftarrow theta_{t-1} - gamma g_t [-1.ex]
     &amp;rule{110mm}{0.4pt} [-1.ex]
     &amp;bf{return} :  theta_t [-1.ex]
     &amp;rule{110mm}{0.4pt} [-1.ex]
end{aligned}
          </annotation>
</semantics>
</math> -->
input : γ (lr) , θ 0 (params) , f ( θ ) (objective) , λ (weight decay) , μ (momentum) , τ (dampening) , nesterov, maximize for t = 1 to … do if maximize : g t ← − ∇ θ f t ( θ t − 1 ) else g t ← ∇ θ f t ( θ t − 1 ) if λ ≠ 0 g t ← g t + λ θ t − 1 if μ ≠ 0 if t > 1 b t ← μ b t − 1 + ( 1 − τ ) g t else b t ← g t if nesterov g t ← g t + μ b t else g t ← b t θ t ← θ t − 1 − γ g t r e t u r n θ t begin{aligned}
 &rule{110mm}{0.4pt} 
 &textbf{input} : gamma text{ (lr)}, : theta_0 text{ (params)}, : f(theta)
 text{ (objective)}, : lambda text{ (weight decay)}, 
 &hspace{13mm} :mu text{ (momentum)}, :tau text{ (dampening)},
 :textit{ nesterov,}:textit{ maximize} [-1.ex]
 &rule{110mm}{0.4pt} 
 &textbf{for} : t=1 : textbf{to} : ldots : textbf{do} 
 &hspace{5mm}textbf{if} : textit{maximize}: 
 &hspace{10mm}g_t leftarrow -nabla_{theta} f_t (theta_{t-1}) 
 &hspace{5mm}textbf{else} 
 &hspace{10mm}g_t leftarrow nabla_{theta} f_t (theta_{t-1}) 
 &hspace{5mm}textbf{if} : lambda neq 0 
 &hspace{10mm} g_t leftarrow g_t + lambda theta_{t-1} 
 &hspace{5mm}textbf{if} : mu neq 0 
 &hspace{10mm}textbf{if} : t > 1 
 &hspace{15mm} textbf{b}_t leftarrow mu textbf{b}_{t-1} + (1-tau) g_t 
 &hspace{10mm}textbf{else} 
 &hspace{15mm} textbf{b}_t leftarrow g_t 
 &hspace{10mm}textbf{if} : textit{nesterov} 
 &hspace{15mm} g_t leftarrow g_{t} + mu textbf{b}_t 
 &hspace{10mm}textbf{else} [-1.ex]
 &hspace{15mm} g_t leftarrow textbf{b}_t 
 &hspace{5mm}theta_t leftarrow theta_{t-1} - gamma g_t [-1.ex]
 &rule{110mm}{0.4pt} [-1.ex]
 &bf{return} : theta_t [-1.ex]
 &rule{110mm}{0.4pt} [-1.ex]
end{aligned}

​ input : γ (lr) , θ 0 ​ (params) , f ( θ ) (objective) , λ (weight decay) , μ (momentum) , τ (dampening) , nesterov, maximize for t = 1 to … do if maximize : g t ​ ← − ∇ θ ​ f t ​ ( θ t − 1 ​ ) else g t ​ ← ∇ θ ​ f t ​ ( θ t − 1 ​ ) if λ  = 0 g t ​ ← g t ​ + λ θ t − 1 ​ if μ  = 0 if t > 1 b t ​ ← μ b t − 1 ​ + ( 1 − τ ) g t ​ else b t ​ ← g t ​ if nesterov g t ​ ← g t ​ + μ b t ​ else g t ​ ← b t ​ θ t ​ ← θ t − 1 ​ − γ g t ​ return θ t ​ ​

Nesterov momentum is based on the formula from [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)  . 

Parameters
:   * **params** ( *iterable*  ) – iterable of parameters or named_parameters to optimize
or iterable of dicts defining parameter groups. When using named_parameters,
all parameters in all groups should be named
* **lr** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – learning rate (default: 1e-3)
* **momentum** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – momentum factor (default: 0)
* **dampening** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – dampening for momentum (default: 0)
* **weight_decay** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – weight decay (L2 penalty) (default: 0)
* **nesterov** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – enables Nesterov momentum. Only applicable
when momentum is non-zero. (default: False)
* **maximize** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – maximize the objective with respect to the
params, instead of minimizing (default: False)
* **foreach** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether foreach implementation of optimizer
is used. If unspecified by the user (so foreach is None), we will try to use
foreach over the for-loop implementation on CUDA, since it is usually
significantly more performant. Note that the foreach implementation uses
~ sizeof(params) more peak memory than the for-loop version due to the intermediates
being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
parameters through the optimizer at a time or switch this flag to False (default: None)
* **differentiable** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether autograd should
occur through the optimizer step in training. Otherwise, the step()
function runs in a torch.no_grad() context. Setting to True can impair
performance, so leave it False if you don’t intend to run autograd
through this instance (default: False)
* **fused** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether the fused implementation is used.
Currently, *torch.float64* , *torch.float32* , *torch.float16* , and *torch.bfloat16* are supported. (default: None)

Note 

The foreach and fused implementations are typically faster than the for-loop,
single-tensor implementation, with fused being theoretically fastest with both
vertical and horizontal fusion. As such, if the user has not specified either
flag (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
implementation when the tensors are all on CUDA. Why not fused? Since the fused
implementation is relatively new, we want to give it sufficient bake-in time.
To specify fused, pass True for fused. To force running the for-loop
implementation, pass False for either foreach or fused.

Example 

```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()

```

Note 

The implementation of SGD with Momentum/Nesterov subtly differs from
Sutskever et al. and implementations in some other frameworks. 

Considering the specific case of Momentum, the update can be written as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<msub>
<mi>
                 v
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  +
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                 =
                </mo>
<mi>
                 μ
                </mi>
<mo>
                 ∗
                </mo>
<msub>
<mi>
                  v
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
                  g
                 </mi>
<mrow>
<mi>
                   t
                  </mi>
<mo>
                   +
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msub>
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
<msub>
<mi>
                 p
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  +
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                 =
                </mo>
<msub>
<mi>
                  p
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 −
                </mo>
<mtext>
                 lr
                </mtext>
<mo>
                 ∗
                </mo>
<msub>
<mi>
                  v
                 </mi>
<mrow>
<mi>
                   t
                  </mi>
<mo>
                   +
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msub>
<mo separator="true">
                 ,
                </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{aligned}
    v_{t+1} &amp; = mu * v_{t} + g_{t+1}, 
    p_{t+1} &amp; = p_{t} - text{lr} * v_{t+1},
end{aligned}
           </annotation>
</semantics>
</math> -->
v t + 1 = μ ∗ v t + g t + 1 , p t + 1 = p t − lr ∗ v t + 1 , begin{aligned}
 v_{t+1} & = mu * v_{t} + g_{t+1}, 
 p_{t+1} & = p_{t} - text{lr} * v_{t+1},
end{aligned}

v t + 1 ​ p t + 1 ​ ​ = μ ∗ v t ​ + g t + 1 ​ , = p t ​ − lr ∗ v t + 1 ​ , ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             p
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            p
           </annotation>
</semantics>
</math> -->p pp  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             g
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            g
           </annotation>
</semantics>
</math> -->g gg  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             v
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            v
           </annotation>
</semantics>
</math> -->v vv  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             μ
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            mu
           </annotation>
</semantics>
</math> -->μ muμ  denote the
parameters, gradient, velocity, and momentum respectively. 

This is in contrast to Sutskever et al. and
other frameworks which employ an update of the form 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<msub>
<mi>
                 v
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  +
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                 =
                </mo>
<mi>
                 μ
                </mi>
<mo>
                 ∗
                </mo>
<msub>
<mi>
                  v
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 +
                </mo>
<mtext>
                 lr
                </mtext>
<mo>
                 ∗
                </mo>
<msub>
<mi>
                  g
                 </mi>
<mrow>
<mi>
                   t
                  </mi>
<mo>
                   +
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msub>
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
<msub>
<mi>
                 p
                </mi>
<mrow>
<mi>
                  t
                 </mi>
<mo>
                  +
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                 =
                </mo>
<msub>
<mi>
                  p
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 −
                </mo>
<msub>
<mi>
                  v
                 </mi>
<mrow>
<mi>
                   t
                  </mi>
<mo>
                   +
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msub>
<mi mathvariant="normal">
                 .
                </mi>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{aligned}
    v_{t+1} &amp; = mu * v_{t} + text{lr} * g_{t+1}, 
    p_{t+1} &amp; = p_{t} - v_{t+1}.
end{aligned}
           </annotation>
</semantics>
</math> -->
v t + 1 = μ ∗ v t + lr ∗ g t + 1 , p t + 1 = p t − v t + 1 . begin{aligned}
 v_{t+1} & = mu * v_{t} + text{lr} * g_{t+1}, 
 p_{t+1} & = p_{t} - v_{t+1}.
end{aligned}

v t + 1 ​ p t + 1 ​ ​ = μ ∗ v t ​ + lr ∗ g t + 1 ​ , = p t ​ − v t + 1 ​ . ​

The Nesterov version is analogously modified. 

Moreover, the initial value of the momentum buffer is set to the
gradient value at the first step. This is in contrast to some other
frameworks that initialize it to all zeros. One notable side effect
of this decision is that the first momentum value will not be scaled
by dampening. Dampening will be applied starting at the second step.

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
from a call to [`state_dict()`](#torch.optim.SGD.state_dict "torch.optim.SGD.state_dict")  .

Warning 

Make sure this method is called after initializing [`torch.optim.lr_scheduler.LRScheduler`](torch.optim.lr_scheduler.LRScheduler.html#torch.optim.lr_scheduler.LRScheduler "torch.optim.lr_scheduler.LRScheduler")  ,
as calling it beforehand will overwrite the loaded learning rates.

Note 

The names of the parameters (if they exist under the “param_names” key of each param group
in [`state_dict()`](#torch.optim.SGD.state_dict "torch.optim.SGD.state_dict")  ) will not affect the loading process.
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

step ( *closure = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/sgd.py#L105) 
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

