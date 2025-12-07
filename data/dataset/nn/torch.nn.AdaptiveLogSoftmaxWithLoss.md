AdaptiveLogSoftmaxWithLoss 
========================================================================================

*class* torch.nn. AdaptiveLogSoftmaxWithLoss ( *in_features*  , *n_classes*  , *cutoffs*  , *div_value = 4.0*  , *head_bias = False*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/adaptive.py#L20) 
:   Efficient softmax approximation. 

As described in [Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
Moustapha Cissé, David Grangier, and Hervé Jégou](https://arxiv.org/abs/1609.04309)  . 

Adaptive softmax is an approximate strategy for training models with large
output spaces. It is most effective when the label distribution is highly
imbalanced, for example in natural language modelling, where the word
frequency distribution approximately follows the [Zipf’s law](https://en.wikipedia.org/wiki/Zipf%27s_law)  . 

Adaptive softmax partitions the labels into several clusters, according to
their frequency. These clusters may contain different number of targets
each.
Additionally, clusters containing less frequent labels assign lower
dimensional embeddings to those labels, which speeds up the computation.
For each minibatch, only clusters for which at least one target is
present are evaluated. 

The idea is that the clusters which are accessed frequently
(like the first one, containing most frequent labels), should also be cheap
to compute – that is, contain a small number of assigned labels. 

We highly recommend taking a look at the original paper for more details. 

* `cutoffs`  should be an ordered Sequence of integers sorted
in the increasing order.
It controls number of clusters and the partitioning of targets into
clusters. For example setting `cutoffs = [10, 100, 1000]`  means that first *10* targets will be assigned
to the ‘head’ of the adaptive softmax, targets *11, 12, …, 100* will be
assigned to the first cluster, and targets *101, 102, …, 1000* will be
assigned to the second cluster, while targets *1001, 1002, …, n_classes - 1* will be assigned
to the last, third cluster.
* `div_value`  is used to compute the size of each additional cluster,
which is given as <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo fence="true">
              ⌊
             </mo>
<mfrac>
<mtext mathvariant="monospace">
               in_features
              </mtext>
<msup>
<mtext mathvariant="monospace">
                div_value
               </mtext>
<mrow>
<mi>
                 i
                </mi>
<mi>
                 d
                </mi>
<mi>
                 x
                </mi>
</mrow>
</msup>
</mfrac>
<mo fence="true">
              ⌋
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             leftlfloorfrac{texttt{in_features}}{texttt{div_value}^{idx}}rightrfloor
            </annotation>
</semantics>
</math> -->⌊ in_features div_value i d x ⌋ leftlfloorfrac{texttt{in_features}}{texttt{div_value}^{idx}}rightrfloor⌊ div_value i d x in_features ​ ⌋  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              i
             </mi>
<mi>
              d
             </mi>
<mi>
              x
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             idx
            </annotation>
</semantics>
</math> -->i d x idxi d x  is the cluster index (with clusters
for less frequent words having larger indices,
and indices starting from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->1 11  ).

* `head_bias`  if set to True, adds a bias term to the ‘head’ of the
adaptive softmax. See paper for details. Set to False in the official
implementation.

Warning 

Labels passed as inputs to this module should be sorted according to
their frequency. This means that the most frequent label should be
represented by the index *0* , and the least frequent
label should be represented by the index *n_classes - 1* .

Note 

This module returns a `NamedTuple`  with `output`  and `loss`  fields. See further documentation for details.

Note 

To compute log-probabilities for all classes, the `log_prob`  method can be used.

Parameters
:   * **in_features** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of features in the input tensor
* **n_classes** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of classes in the dataset
* **cutoffs** ( *Sequence*  ) – Cutoffs used to assign targets to their buckets
* **div_value** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – value used as an exponent to compute sizes
of the clusters. Default: 4.0
* **head_bias** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If `True`  , adds a bias term to the ‘head’ of the
adaptive softmax. Default: `False`

Returns
:   * **output** is a Tensor of size `N`  containing computed target
log probabilities for each example
* **loss** is a Scalar representing the computed negative
log likelihood loss

Return type
:   `NamedTuple`  with `output`  and `loss`  fields

Shape:
:   * input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mtext mathvariant="monospace">
                in_features
               </mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, texttt{in_features})
              </annotation>
</semantics>
</math> -->( N , in_features ) (N, texttt{in_features})( N , in_features )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mtext mathvariant="monospace">
                in_features
               </mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (texttt{in_features})
              </annotation>
</semantics>
</math> -->( in_features ) (texttt{in_features})( in_features )

* target: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N)
              </annotation>
</semantics>
</math> -->( N ) (N)( N )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               ()
              </annotation>
</semantics>
</math> -->( ) ()( )  where each value satisfies <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0
               </mn>
<mo>
                &lt;
               </mo>
<mo>
                =
               </mo>
<mtext mathvariant="monospace">
                target[i]
               </mtext>
<mo>
                &lt;
               </mo>
<mo>
                =
               </mo>
<mtext mathvariant="monospace">
                n_classes
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               0 &lt;= texttt{target[i]} &lt;= texttt{n_classes}
              </annotation>
</semantics>
</math> -->0 < = target[i] < = n_classes 0 <= texttt{target[i]} <= texttt{n_classes}0 <= target[i] <= n_classes

* output1: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N)
              </annotation>
</semantics>
</math> -->( N ) (N)( N )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               ()
              </annotation>
</semantics>
</math> -->( ) ()( )

* output2: `Scalar`

log_prob ( *input* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/adaptive.py#L281) 
:   Compute log probabilities for all <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext mathvariant="monospace">
              n_classes
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             texttt{n_classes}
            </annotation>
</semantics>
</math> -->n_classes texttt{n_classes}n_classes  . 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a minibatch of examples

Returns
:   log-probabilities of for each class <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                c
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               c
              </annotation>
</semantics>
</math> -->c cc  in range <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0
               </mn>
<mo>
                &lt;
               </mo>
<mo>
                =
               </mo>
<mi>
                c
               </mi>
<mo>
                &lt;
               </mo>
<mo>
                =
               </mo>
<mtext mathvariant="monospace">
                n_classes
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               0 &lt;= c &lt;= texttt{n_classes}
              </annotation>
</semantics>
</math> -->0 < = c < = n_classes 0 <= c <= texttt{n_classes}0 <= c <= n_classes  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext mathvariant="monospace">
                n_classes
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               texttt{n_classes}
              </annotation>
</semantics>
</math> -->n_classes texttt{n_classes}n_classes  is a
parameter passed to `AdaptiveLogSoftmaxWithLoss`  constructor.

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

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
<mtext mathvariant="monospace">
                  in_features
                 </mtext>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, texttt{in_features})
                </annotation>
</semantics>
</math> -->( N , in_features ) (N, texttt{in_features})( N , in_features )

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
<mtext mathvariant="monospace">
                  n_classes
                 </mtext>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, texttt{n_classes})
                </annotation>
</semantics>
</math> -->( N , n_classes ) (N, texttt{n_classes})( N , n_classes )

predict ( *input* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/adaptive.py#L300) 
:   Return the class with the highest probability for each example in the input minibatch. 

This is equivalent to `self.log_prob(input).argmax(dim=1)`  , but is more efficient in some cases. 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a minibatch of examples

Returns
:   a class with the highest probability for each example

Return type
:   output ( [Tensor](../tensors.html#torch.Tensor "torch.Tensor")  )

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
<mtext mathvariant="monospace">
                  in_features
                 </mtext>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, texttt{in_features})
                </annotation>
</semantics>
</math> -->( N , in_features ) (N, texttt{in_features})( N , in_features )

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  N
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N)
                </annotation>
</semantics>
</math> -->( N ) (N)( N )

