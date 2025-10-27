Published as a conference paper at ICLR 2025
BILINEAR MLPS ENABLE WEIGHT-BASED
MECHANISTIC INTERPRETABILITY
Michael T. Pearce* Thomas Dooms* Alice Rigg
Independent University of Antwerp Independent
pearcemt@ thomas.dooms@ rigg.alice0@
alumni.stanford.edu uantwerpen.be gmail.com
Jose Oramas Lee Sharkey
University of Antwerp, sqIRL/IDLab Apollo Research
jose .oramas@uantwerpen.be lee@apolloresearch.ai
ABSTRACT
A mechanistic understanding of how MLPs do computation in deep neural net-
works remains elusive. Current interpretability work can extract features from
hidden activations over an input dataset but generally cannot explain how MLP
weights construct features. One challenge is that element-wise nonlinearities
introduce higher-order interactions and make it difficult to trace computations
through the MLP layer. In this paper, we analyze bilinear MLPs, a type of
Gated Linear Unit (GLU) without any element-wise nonlinearity that neverthe-
less achieves competitive performance. Bilinear MLPs can be fully expressed in
terms of linear operations using a third-order tensor, allowing flexible analysis of
the weights. Analyzing the spectra of bilinear MLP weights using eigendecom-
position reveals interpretable low-rank structure across toy tasks, image classifi-
cation, and language modeling. We use this understanding to craft adversarial
examples, uncover overfitting, and identify small language model circuits directly
from the weights alone. Our results demonstrate that bilinear layers serve as an
interpretable drop-in replacement for current activation functions and that weight-
based interpretability is viable for understanding deep-learning models.
1 INTRODUCTION
Multi-layer perceptrons (MLPs) are an important component of many deep learning models, in-
cluding transformers (Vaswani et al}, 2017). Unfortunately, element-wise nonlinearities obscure the
relationship between weights, inputs, and outputs, making it difficult to trace a neural network’s
decision-making process. Consequently, MLPs have previously been treated as undecomposable
components in interpretability research [2021).
While early mechanistic interpretability literature explored neural network weights
2017}, 2020} [Voss et al, 2021} [Elhage et al} 2021), activation-based approaches dominate contem-
porary research (Petsiuk et al| 2018} [Ribeiro et al| [2016; [Simonyan et al 2014} [Montavon et al,
@D. In particular, most recent studies on transformers use sparse dictionary learning (SDL) to
decompose latent representations into an overcomplete basis of seemingly interpretable atoms @
[ningham et al] 2024} [Bricken et al| 2023b; Marks et al], 2024} [Dunefsky et al] [2024). However,
SDL-based approaches only describe which features are present, not how they are formed or what
their downstream effect is. Previous work has approximated interactions between latent dictionaries
using linear and gradient-based attribution (Marks et al}}[2024; [Ge et al}}[2024), but these approaches
offer weak guarantees of generalization. To ensure faithfulness, ideally, we would be able to capture
nonlinear feature interactions in circuits that are grounded in the model weights.
“Equal contribution
Code at: https://github.com/tdooms/bilinear-decomposition/
Published as a conference paper at ICLR 2025
One path to better circuit discovery is to use more inherently interpretable architectures. Previous
work constrains the model into using human-understandable components or concepts
[2019), but this typically requires labeled training data for the predefined con-
cepts and involves a trade-off in accuracy compared to learning the best concepts for performance
(Henighan et al [2023). Ideally, we could more easily extract and interpret the concepts that mod-
els naturally learn rather than force the model to use particular concepts. To this end, [Sharkey]
(2023) suggested that bilinear layers [Cin et al] 2015); [Li et al| (2017); [Chrysos et al] (2021) of
the form g(z) = (Wx) ® (V) are intrinsically interpretable because their computations can be
expressed in terms of linear operations with a third order tensor. This enables the use of tensor
or matrix decompositions to directly understand the weights. Moreover, bilinear layers outperform
ReLU-based transformers in language modeling (Shazeer} [2020) and have performance only slightly
below SwiGLU, which is prevalent in competitive models today [2023).
Tensor decompositions have long been studied in machine learning (Cichocki et al| [2015} [Panagakis|
et al] 2021} [Sidiropoulos et al | [2017) where most applications are based on an input dataset. Using
decompositions to extract features directly from the weights of tensor-based models is less explored.
Here, we show that bilinear MLPs can be decomposed into functionally relevant, interpretable com-
ponents by directly decomposing the weights, without using inputs. These decompositions reveal
a low-rank structure in bilinear MLPs trained across various tasks. In summary, this paper demon-
strates that bilinear MLPs are an interpretable drop-in replacement for ordinary MLPs in a wide
range of settings. Our contributions are as follows:
1. Infsection 3] we introduce several methods to analyze bilinear MLPs. One method decom-
poses the weights into a set of eigenvectors that explain the outputs along a given set of
directions in a way that is fully equivalent to the layer’s original computations.
2. Infsection 4] we showcase the eigenvector decomposition across multiple image classifica-
tion tasks, revealing an interpretable low-rank structure. Smaller eigenvalue terms can be
truncated while preserving performance. Using the eigenvectors, we see how regularization
reduces signs of overfitting in the extracted features and construct adversarial examples.
3. Finally, in [section 3| we analyze how bilinear MLPs compute output features from input
features, both derived from sparse dictionary learning (SDL). We highlight a small circuit
that flips the sentiment of the next token if the current token is a negation (“not”). We
also find that many output features are well-correlated with low-rank approximations. This
gives evidence that weight-based interpretability can be viable in large language models.
2 BACKGROUND
Throughout, we use conventional notation as in (2016). Scalars are denoted by s,
vectors by v, matrices by M, and third-order tensors by T. The entry in row ¢ and column j of a
matrix M is a scalar and therefore denoted as m;;. We denote taking row 7 or column ;j of a matrix
by m;. and m.; respectively. We use © to denote an element-wise product and - to denote a
product of tensors along the specified axis.
Defining bilinear MLPs. Modern Transformers [2023) feature Gated Linear Units
(GLUs), which offer a performance gain over standard MLPs for the same number of parameters
Shazeer], [2020% [Dauphin et al} 2017). GLU activations consist of the component-wise product
of two linear up-projections of size (dhidden, dinput), W and V', one of which is passed through a
nonlinear activation function o((Equation I). The hidden activations g(z) then pass through a down-
projection P of size (douput: dhidden) - We Omit biases for brevity.
9(@) = (Wa) ©o(V) o
GLU(2) = Plg(=))
A bilinear layer is a GLU variant that omits the nonlinear activation function o Bilinear layers beat
ordinary ReLU MLPs and perform almost as well as SwiGLU on language modeling tasks
[2020). We corroborate these findings in[Appendix 1| and show that bilinear layers achieve equal loss
when keeping training time constant and marginally worse loss when keeping data constant.
Published as a conference paper at ICLR 2025
Interaction matrices and the bilinear tensor. A bilinear MLP parameterizes the pairwise interac-
tions between inputs. One way to see this is by looking at how a single output g(x), is computed.
o(@) = (Wa) o (Va)
T T
9(@)a = (woz) (v,T)
= 2T (w,vh)z
We call the (dinput, dinput) Matrix wg; vg" = By an interaction matrix since it defines how each pair
of inputs interact for a given output dimension a.
The collection of interaction matrices across the output axis can be organized into the third-order
bilinear tensor, B, with elements bai; = waivaj, illustrated in[Figure TA. The bilinear tensor allows
us to easily find the interaction matrix for a specific output direction w of interest by taking a product
along the output axis, u oy B, equal to a weighted sum over the neuron-basis interaction matrices,
>4 UaWa:V,,. As written, B has size (dhigden, dinput: dinpur) but we will typically multiply the down-
projection P into B resulting in a (douput. dinput dinput) Size tensor.
Simplifications due to symmetry. Because an interaction matrix is always evaluated with two
copies of the input @, it contains redundant information that does not contribute to the activation.
Any square matrix can be expressed uniquely as the sum of a symmetric and anti-symmetric matrix.
1 1
Bu: = 5(Bu:+ BL) + 5(Bu: — BL})
—_ —_
B Banti
However, evaluating an anti-symmetric matrix A with identical inputs yields 0 and can be omitted:
2T Az = 2T (—AT)z = (2" Ax)T = 0.
Therefore, only the symmetric part B3%™ contributes. From here on, we drop the -*¥" superscript
and assume the symmetric form for any interaction matrix or bilinear tensor (bsij = %(wmva] +
Waj Vgi)). Symmetric matrices have simpler eigendecompositions since the eigenvalues are all real-
valued, and the eigenvectors are orthogonal by the spectral theorem.
Incorporating biases. If a bilinear layer has biases, we can augment the weight matrices to adapt
our approach. Given activations of the form g(x) = (Wx+b,)©(Vx+b,), define W’ = [W; by],
V' = [V;by], and @’ = [z, 1]. Then, g(x) = (W'z’) © (V'z') in a bilinear layer with biases.
In [subsection 4.3] we study a toy classification task using a model trained with biases, illustrating
how biases can be interpreted using the same framework. For the rest of our experiments, we used
models without biases for simplicity, as it did not harm performance. See[Appendix L] for details.
3 ANALYSIS METHODS
Since bilinear MLPs can be expressed in terms of a third-order tensor, B, they can be flexibly
analyzed using techniques from linear algebra, such as decompositions and transformations. The
choice of analysis approach depends on what additional information, in terms of previously obtained
input or output features, is provided.
3.1 INPUT/OUTPUT FEATURES — DIRECT INTERACTIONS
If we have already obtained meaningful sets of features for the bilinear MLP’s inputs and outputs,
for example from a set of latent feature dictionaries '™ and F°", then we can directly study the in-
teractions between these features and understand how the output features are constructed from input
ones. We can transform the bilinear tensor into the feature basis via bave = 3,k fai'" bijk fy] foi-
For a given set of sparse input and output activations, only a small subset of the interactions (with
a, b, c all active) will contribute, and the statistics of these active interactions can be studied.
For dictionaries obtained from sparse autoencoders (SAEs) we can instead use the output SAE’s
encoder directions in the transformation: base = 37,1 €4t bijk f,! fci- Then the output activations
are 20" = ReLU(Y_,; babe2i'2y") in terms of the input directions z™. I we use this
approach to identify the top relevant interactions for features in a language me
Published as a conference paper at ICLR 2025
A) Bilinear layer Bilinear tensor B) Multiply by output Tnteraction Matrix Eigendecomposition
ecor
s . N
" X B ‘m — ‘: T
\4 B H % N\ it Q -
- m 4
o@)=WzoVz g@)= bu; ziz; U o B=Q= A\ vl
Figure 1: A) Two ways to represent a bilinear layer, via an elementwise product or the bilinear
tensor. B) Diagram of the eigendecomposition technique. Multiplying the bilinear tensor by a
desired output direction w produces an interaction matrix @ that can be decomposed into a set of
eigenvectors v and associated eigenvalues \;.
3.2 OUTPUT FEATURES — EIGENDECOMPOSITION
Given a set of meaningful features for the MLP outputs, we can identify the most important input
directions that determine the output feature activations. The output features could come from a
dictionary, from the unembedding (shown in[section 4), or from the decompilation of later layers.
The interaction matrix, @ = u -y B for a given output feature w can be decomposed into a set
of eigenvectors (Figure I). Since @ can be considered symmetric without loss of generality (see
[section 2), the spectral theorem gives
d
Q=" Nwu] )
i
with a set of d (the rank of W, V') orthonormal eigenvectors v; and real-valued eigenvalues \;. In
the eigenvector basis, the output in the u-direction is
d
TQe =5 A (v ) 3
' Qx Z (vl'z) 3)
act for v,
where each term can be considered the activation for the eigenvector v; of size (dinpu). That is, the
bilinear layer’s outputs are quadratic in the eigenvector basis.
The eigenvector basis makes it easy to identify any low-rank structure relevant to . The top eigen-
vectors by eigenvalue magnitude give the best low-rank approximation to the interaction matrix Q
for a given rank. And since the eigenvectors diagonalize Q, there are no cross-interactions between
eigenvectors that would complicate the interpretation of their contributions to w.
3.3 NO FEATURES — HIGHER-ORDER SVD
If we have no prior features available, it is still possible to determine the most important input and
output directions of B through a higher-order singular value decomposition (HOSVD). The simplest
approach that takes advantage of the symmetry in B is to reshape the tensor by flattening the two
input dimensions to produce a (douput, dfi\pm) shaped matrix and then do a standard singular value
decomposition (SVD). Schematically, this gives
Buuinxin = Y05 ) © g
:
where g can still be treated as an interaction matrix and further decomposed into eigenvectors as
described above. We demonstrate this approach for an MNIST model in [Appendix D}
4 IMAGE CLASSIFICATION: INTERPRETING VISUAL FEATURES
‘We consider models trained on the MNIST dataset of handwritten digits and the Fashion-MNIST
dataset of clothing images. This is a semi-controlled environment that allows us to evaluate the
Published as a conference paper at ICLR 2025
A) B) : :
=3 =
. L=2*® Egenvector=v, Input=x | Activation = A(yT?
Hih
- Hh AT i r'b';l'\.
L
Figure 2: A) Eigenvector activations are quadratic in the input and have a large magnitude if an
input aligns with the positive (blue) regions or the negative (red) regions, but not both. B) Top
eigenvectors for single-layer MNIST and Fashion-MNIST models, revealing the most significant
patterns learned for each class. In MNIST, eigenvectors represent components of the target class,
while Fashion-MNIST eigenvectors function as localized edge detectors. Best viewed in color.
interpretability of eigenvectors computed using the methods in[subsection 3.2} This section analyses
a shallow feedforward network (FEN) consisting of an embedding projection, a bilinear layer, and a
classification head; see[Appendix G for details.
First, we qualitatively survey the eigenvectors and highlight the importance of regularization in
feature quality. Second, we consider the consistency of eigenvectors across training runs and sizes.
Third, we turn toward an algorithmic task on MNIST, where we compare the ground truth with
the extracted eigenvectors. Lastly, we use these eigenvectors to construct adversarial examples,
demonstrating their causal importance.
4.1 QUALITATIVE ASSESSMENT: TOP EIGENVECTORS APPEAR INTERPRETABLE
The eigenvectors are derived using the unembedding directions for the digits as the output directions
w to obtain interaction matrices @ = -ou B that are then decomposed following subsection 3.2} So
each unembedding direction (digit) has a corresponding set of eigenvectors, although we may refer
to the full collection as the eigenvectors of the layer or model.
‘We can visualize them by projecting them into the input space using the embedding wei%h(s. Be-
cause the activation of an eigenvector v with eigenvalue )\; is quadratic in the input, A\(v!x)?, the
sign of the eigenvector v is arbitrary. The quadratic leads to XOR-like behavior where high overlap
with an eigenvector’s positive regions (blue) or the negative regions (red)—but not both—leads to
large activation magnitude, while the overall sign is determined by the eigenvalue (Figure 2)A).
For MNIST, the top positive eigenvector for each output class emphasizes a curve segment specific
to its digit or otherwise resembles a prototypical class image (Figure 2B). Top eigenvectors for
FMNIST function as localized edge detectors, focusing on important edges for each clothing article,
such as the leg gap for trousers. The localized edge detection relies on the XOR-like behavior of the
eigenvector’s quadratic activation.
LU
/mu
y P %
Figure 3: The top four positive (top) and negative (bottom) eigenvectors for the digit 5, ordered
from left to right by importance. Their eigenvalues are highlighted on the left. Only 20 positive and
20 negative eigenvalues (out of 512) are shown on the left images. Eigenvectors tend to represent
semantically and spatially coherent structures.
Published as a conference paper at ICLR 2025
("
norm=0 orm=1
Figure 4: Top eigenvector for models trained with varying Gaussian input noise. For reference, the
norm of an average digit is about 0.3; adding noise with a norm of 1 results in a heavily distorted
but discernible digit. Finally, the test accuracy for each model is shown at the top.
Only a small fraction of eigenvalues have non-negligible magnitude (Figure 3). Different top eigen-
vectors capture semantically different aspects of the class. For example, in the spectrum for digit
5, the first two positive eigenvectors detect the 5°s horizontal top stroke but at different positions,
similar to Gabor filters. The next two positive eigenvectors detect the bottom segment. The negative
eigenvectors are somewhat less intuitive but generally correspond to features that indicate the digit
is not a five, such as an upward curve in the top right quadrant instead of a horizontal stroke. In
we study this technique towards explaining an input prediction. Details of the training
setup are outlined in[Appendix G| while similar plots for other digits can be found in[Appendix A}
Because we can extract features directly from model weights, we can identify overfitting in image
models by visualizing the top eigenvectors and searching for spatial artifacts. For instance, the
eigenvectors of unregularized models focus on certain outlying pixels (Figure 4). We found adding
dense Gaussian noise to the inputs [2023a) to be an effective model regularizer,
producing bilinear layers with more intuitively interpretable features. Increasing the scale of the
added noise results produces more digit-like eigenvectors and results in a lower-rank eigenvalue
spectrum (Appendix E). These results indicate that our technique can qualitatively help uncover
overfitting or other unwanted behavior in models. Furthermore, it can be used to evaluate the effect
of certain regularizers and augmentation techniques, as explored in [Appendix B}
4.2 QUANTITATIVE ASSESSMENT: EIGENVECTORS LEARN CONSISTENT PATTERNS
One important question in machine learning is whether models learn the same structure across train-
ing runs (Liet al} [2016) and across model sizes (Frankle & Carbin] 2019). In this section, we study
both and find that eigenvectors are similar across runs and behave similarly across model sizes.
Furthermore, we characterize the impact of eigenvector truncation on classification accuracy.
Both the ordering and contents of top eigenvectors are very consistent across runs. The cosine
similarities of the top eigenvector are between 0.8 and 0.9 depending on size (Figure 3). Generally,
) B)
cmooi I
[— FTp———
Figure 5: A) The similarity between ordered eigenvectors of the same model size averaged over all
digits. This shows that equally sized models leam similar features. B) Resulting accuracy after only
retaining the n most important eigenvalues (per digit). Both plots are averaged over 5 runs with the
90% confidence interval shown.
Published as a conference paper at ICLR 2025
increasing model sizes results in more similar top eigenvectors. Further, truncating all but the top
few eigenvectors across model sizes yields very similar classification accuracy. This implies that,
beyond being consistently similar, these eigenvectors have a comparable impact on classification. In
we further study the similarity of eigenvectors between sizes and show that retaining
only a handful of eigenvectors results in minimal accuracy drops (0.01%).
4.3 COMPARING WITH GROUND TRUTH: EIGENVECTORS FIND COMPUTATION
To perform a ground-truth assessment of eigenvectors, we consider a task from a mechanistic in-
terpretability challenge, where the goal was to determine the labeling function (training objective)
from a model [Casper| . Specifically, the challenge required reverse-engineering a binary im-
age classifier trained on MNIST, where the label is based on the similarity to a specific target image.
The model predicted ‘True’ if the input has high cosine similarity to this target or high similarity to
the complement (one minus the grayscale) of that target and ‘False’ otherwise. This target is chosen
as an instance of a ‘1°.
Previous work (Stefan Heimersheim| 2023) reverse-engineered this through a combination of meth-
ods, all requiring careful consideration and consisting of non-trivial insights. Furthermore, the meth-
ods required knowledge of the original dataset and a hint of what to look for. While our method does
not work on the original architecture, we show that we do not require such knowledge and can extract
the original algorithm from the weights alone.
‘We perform our decomposition on the output difference (True — False) since this is the only mean-
ingful direction before the softmax. This consistently reveals one high positive eigenvalue; the rest
are (close to) zero (Figure 6). The most positive eigenvector is sufficient for completing the task;
it computes the exact similarity we want. If the input is close to the target, the blue region will
match; if it is close to the complement, the red will match; if both are active simultaneously, they
will somewhat cancel out. The remaining two eigenvectors are separated as they seem to overfit the
data slightly; the negative eigenvector seems to penalize diagonal structures.
Contrary to other models, this task greatly benefited from including biases. This arises from the
fact that the model must not only compute similarity but also make its binary decision based on a
learned threshold. If no bias is provided, the model attempts to find quadratic invariances in the
data, which don’t generalize well, especially given the important but sensitive role of this threshold
in classification. Here, the bias (shown in the bottom corner ofms a negative
contribution. The role of biases in bilinear layers is further discussed in[A
{8\
Target
Bl Bias
Figure 6: Eigenvalues and eigenvectors of a model trained to classify based on similarity to a target.
The most important eigenvector (top-left) is a generalizing solution; the other features sharpen the
decision boundary based on the training dataset. The latter features disappear with increased regu-
larization. On the right, the target digit is shown along with the learned bias from the model.
4.4 ADVERSARIAL MASKS: GENERAL ATTACKS FROM WEIGHTS
To demonstrate the utility of weight-based decomposition, we construct adversarial masks for the
MNIST model without training or any forward passes. These masks are added to the input, leading
to misclassification as the adversarial digit. The effect is similar to steering, but the intervention is
at the input instead of the model internals.
Published as a conference paper at ICLR 2025
it ~+ orgnal
w T
Adversarisl Misclassfed Random st st
Example. Mask Hosk st s
; g s £os
. 2] oo
e 04 {| & aoveranl £o1
2 Bangam
“Nasc sty Caskssao
Figure 7: Examples of an adversarial mask constructed from the given eigenvector along for models
trained A) with Gaussian noise regularization (std 0.15) and B) without regularization. The average
accuracy and the rate of misclassification as the adversarial digit show stronger effects for adversarial
masks than random baselines. In B), the mask is only applied to the outer edge of pixels that are
active on less than 1% of samples.
‘We construct the adversarial masks from the eigenvectors for specific digits. One complication is
that the eigenvectors can have nontrivial cosine similarity with each other, so an input along a single
eigenvector direction could potentially activate multiple eigenvectors across different digits. To help
avoid this, we construct the mask m; for a given eigenvector v;, as the corresponding row of the
pseudoinverse (V+);; for a set of eigenvectors V' (specifically the top 10 positive). In an analogy to
key-value pairs, the pseudoinverses effectively act like keys that activate with more specificity than
the eigenvectors themselves, since vj. - (V)i = d;5.
We construct an adversarial mask from an eigenvector for the digit 3 (Figure 7JA). Even though the
original eigenvector resembles the digit, the pseudoinverse-based mask does not (see [Appendix M|
for more examples). The accuracy, averaged over masks from the top three eigenvectors, drops
significantly more than the baseline of randomly permuting the mask despite regularizing the model
during training using dense Gaussian noise with a standard deviation of 0.15. The corresponding
rise in misclassification indicates effective steering towards the adversarial digit.
(2019) observe that adversarial examples can arise from predictive but non-robust features
of the data, perhaps explaining why they often transfer to other models. Our construction can be
seen as a toy realization of this phenomenon because the masks correspond to directions that are
predictive of robust features but are not robust. We construct masks that only exploit the patterns
of over-fitting found on the outer edge of the image for a model trained without regularization
(Figure 7B). Since we can find this over-fitting pattern from the eigenvectors, in a general way, we
can construct the mask by hand instead of optimizing it.
5 LANGUAGE: FINDING INTERACTIONS BETWEEN SAE FEATURES
Each output of a bilinear layer is described by weighted pairwise interactions between their input
features. Previous sections show that this can be successfully leveraged to trace between a bilinear
layer’s inputs and outputs. Here, we turn towards tracing between latent feature dictionaries ob-
tained by training sparse autoencoders (SAEs) on the MLP inputs or outputs for a 6-layer bilinear
transformer trained on TinyStories (Eldan & Li| 2023) (see training details in[Appendix G).
5.1 SENTIMENT NEGATION CIRCUIT
‘We focus on using the eigendecomposition to identify low-rank, single-layer circuits in a bilinear
transformer. We cherry-pick and discuss one such circuit that takes input sentiment features and
semantically negates them. Unlike previous work on sparse feature circuits (Marks et al | 2024) that
relies on gradient-based linear approximations, we identify nonlinear interactions grounded in the
layer’s weights that contribute to the circuit’s computation.
Published as a conference paper at ICLR 2025
A B a0
- - .z o =
B £
E) L]
e N l g O
— ) i 20 30 @
Top positive eigenvector Output featre activation
Figure 8: The sentiment negation circuit that computes the not-good and not-bad output features.
A) The interaction submatrix containing the top 15 interactions. B) The projection of top interacting
features onto the top eigenvectors using cosine similarity. The symbols for different clusters match
the labels in A. Clusters coincide with the projection of meaningful directions such as the difference
in “bad” vs “good” token unembeddings and the MLP’s input activations for the input “[BOS] not”.
C) The not-good feature activation compared to its approximation by the top two eigenvectors.
The sentiment negation circuit computes the activation of two opposing output features in layer 4
(index 1882 and 1179) that form a fully linear subspace. The cosine similarity of their decoder
vectors is -0.975. Based on their top activations, the output features activate on negation tokens
(“not”, “never”, “wasn’t”) and boosts either positive sentiment tokens (“good”, “safe”, “nice”) or
negative sentiment tokens (“bad”, “hurt”, “sad”), so we denote the two features as the not-good and
the not-bad features respectively. See[Appendix O] for the top activations of all features mentioned.
Focusing on the not-good feature, the top interactions for computing its activations resemble an
AND-gate (Figure 8A). Input features that boost negative sentiment tokens (blue squares) have
strong positive interactions with negation token features (green triangles), but both have negligi-
ble self-interactions. So, both types of input features are needed to activate the not-good feature
and flip the boost from negative to positive sentiment. The one positive sentiment feature (orange
downward triangle) interacts with the opposite sign. The interactions shown are significantly larger
than the typical cross-interactions with a standard deviation of 0.004
The eigenvalue spectrum has one large positive (0.62) and one large negative value (-0.66) as outliers
(Figure 27). We can see the underlying geometry of the circuit computation by projecting the input
features onto these eigenvectors (Figure 8). By itself, a positive sentiment feature (blue squares)
would equally activate both eigenvectors and cancel out, but if a negation feature is also present,
the positive eigenvector is strongly activated. The activation based on only these two eigenvectors,
following [Equation 3} has a good correlation (0.66) with the activation of the not-good feature,
particularly at large activation values (0.76), conditioned on the not-good feature being active.
5.2 LOW-RANK APPROXIMATIONS OF OUTPUT FEATURE ACTIVATIONS
The top eigenvectors can be used to approximate the activations of the SAE output features using a
truncated form of [Equation 3] To focus on the more meaningful tail of large activations, we compute
the approximation’s correlation conditioned on the output feature being active. The correlations
of inactive features are generally lower because they are dominated by ‘noise’. We evaluate this
on three bilinear transformers at approximately 2/3 depth: a 6-layer TinyStories (‘ts-tiny’) and two
FineWeb models with 12 and 16 layers (‘fw-small’ and ‘fw-medium’).
‘We find that features are surprisingly low-rank, with the average correlation starting around 0.65 for
approximations by a single eigenvector and rising steadily with additional eigenvectors (Figure 9A).
Most features have a high correlation (> 0.75) even when approximated by just two eigenvectors
(Figure 9B). Scatter plots for a random sample of features show that the low-rank approximation
often captures the tail dependence well (Figure 91). Interestingly, we find the approximation to
drastically improve with longer SAE training times while other metrics change only slightly. This
indicates a ‘hidden’ transition near convergence and is further discussed in [Appendix H} Overall,
these results suggest that the interactions that produce large output activations are low-rank, making
their interpretability potentially easier.
Published as a conference paper at ICLR 2025
A) Feature activation approximation B) Approx. by top 2 sgenvectars o)
“ -
" / . e
Figure 9: Activation correlations with low-rank approximations for differently-sized transformers.
A) Average correlation over output features computed over every input where the feature is active.
B) The distribution of active-only correlations for approximations using the top two eigenvectors.
C) Scatter plots for a random set of nine output features on ‘fw-medium’. Approximations use the
top two eigenvectors. Low correlation scores generally only occur on low-activation features.
6 DISCUSSION
Summary. This paper introduces a novel approach to weight-based interpretability that leverages
the close-to-linear structure of bilinear layers. A key result is that we can identify the most im-
portant input directions that explain the layer’s output along a given direction using an eigenvector
decomposition. The top eigenvectors are often interpretable, for example for MNIST they function
as edge-detectors for strokes specific to each digit. The lack of element-wise nonlinearity in bilinear
MLPs allows us to transform their weights into interaction matrices that connect input to output
features and then extract the low-rank structure. In language models, we find that many SAE output
features are well-approximated by low-rank interaction matrices, particularly at large activations.
‘We highlighted one example of an extracted low-rank circuit that flips the sentiment of the next to-
ken if the current token is a negation (“not”). The behavior of this circuit can be easily understood in
terms of the top eigenvectors, whereas finding a similar circuit in conventional MLPs would be more
difficult. Overall, our results demonstrate that bilinear MLPs offer intrinsic interpretability that can
aid in feature and circuit extraction.
Implications. The main implication of our work is that weight-based interpretability is viable, even
for large language models. Bilinear MLPs can replace conventional MLPs in transformers with min-
imal cost while offering intrinsic interpretability due to their lack of element-wise nonlinearities and
close-to-linear structure. Current circuit analysis techniques rely on gradient-based approximations
(Syed et al}, 2023} [2024) or use transcoders [2024) to approximate
MLPs. Both approaches depend on an input dataset, potentially leading to poor performance out-
of-distribution, and they may not fully capture the nonlinear computations in MLPs. In contrast,
bilinear MLPs can be transformed into explicit feature interaction matrices and decomposed in a
way fully equivalent to the original computations. Extracting interactions more directly from the
weights should lead to better, more robust circuits. Weight-based interpretability may also offer
better safety guarantees since we could plausibly prove bounds on a layer’s outputs by quantifying
the residual weights not captured in a circuit’s interactions.
Limitations. Application of our methods typically relies on having a set of meaningful output direc-
tions available. In shallow models, the unembedding directions can be used, but in deeper models,
we rely on features derived from sparse autoencoders that are dependent on an input dataset. An-
other limitation is that, although the eigenvalue spectra are often low-rank and the top eigenvectors
appear interpretable, there are no guarantees the eigenvectors will be monosemantic. We expect that
for high-rank spectra, the orthogonality between eigenvectors may limit their interpretability. Ap-
plying sparse dictionary learning approaches to decompose the bilinear tensor may be a promising
way to relax the orthogonality constraint and find interpretable features from model weights.
ACKNOWLEDGEMENTS
‘We are grateful to Narmeen Oozeer, Nora Belrose, Philippe Chlenski, and Kola Ayonrinde for help-
ful feedback on the draft. We are grateful to the AI Safety Camp| program where this work first
10
Published as a conference paper at ICLR 2025
started and to the (MATS) program that supported Michael and
Alice while working on this project. We thank CoreWeave for providing compute for the finetuning
experiments. This research received funding from the Flemish Government under the “Onderzoek-
sprogramma Artificiéle Intelligentie (AI) Vlaanderen” programme.
CONTRIBUTIONS
Michael performed the bulk of the work on the MNIST analysis and provided valuable insights
across all presented topics. Thomas worked on the Language Models section and was responsible
for code infrastructure. The paper was written in tandem, each focusing on their respective section.
ETHICS STATEMENT
This paper proposes no advancements to the state-of-the-art in model capabilities. Rather, it provides
new methods to analyze the internals of models to increase our understanding. The only misuse the
authors envision is using this technique to leak details about the dataset that the model has learned
more efficiently. However, this can be avoided by using this technique during safety evaluation.
REPRODUCIBILITY STATEMENT
‘We aspired to make this work as reproducible as possible. First, (among others) aims
to provide detailed and sufficient descriptions to independently recreate our training setups. Second,
our code (currently public but not referenced for anonymity) contains separate files that can be
used to generate the figures in this paper independently. We used seeds across training runs so that
recreated figures would be equivalent. Third, all models that are compute-intensive to train, such as
the SAEs and the LMs, will be shared publicly. Lastly, we will publish an interactive demo, which
will allow independent analysis of the figures in [Appendix_A] [Appendix B and [Appendix O] in a
way this document cannot.
REFERENCES
Trenton Bricken, Rylan Schaeffer, Bruno Olshausen, and Gabriel Kreiman. Emergence of sparse
representations from noise. In International Conference on Machine Learning, pp. 3148-3191.
PMLR, 2023a.
Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly,
Nicholas L. Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu,
Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Alex Tamkin, Karina Nguyen,
Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, and Chris Olah.
Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Trans-
former Circuits Thread, October 2023b. URL ht tps://transformer—circuits.pub/|
2023 /monosemant ic- features/index.htmll
Stephen Casper. Eis vii: A challenge for mechanists, 2023. URL https://
www.alignment forum.orq/s/a6bne2ve5ut urEEQK7 /p/KSHoLzOscwdnv44Ts| Al
Alignment Forum, Part 7 of the Engineer’s Interpretability Sequence, posted on February 18,
2023.
Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett, Jonathan Su, and Cynthia Rudin.
This looks like that: Deep learning for interpretable image recognition, 2019. URL ht B
[//arxiv.orqg/abs/1806.10574l
Grigorios G Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Jiankang Deng, Yannis Panagakis,
and Stefanos Zafeiriou. Deep polynomial neural networks. IEEE transactions on pattern analysis
and machine intelligence, 44(8):4021-4034, 2021.
Andrzej Cichocki, Danilo Mandic, Lieven De Lathauwer, Guoxu Zhou, Qibin Zhao, Cesar Caiafa,
and HUY ANH PHAN. Tensor decompositions for signal processing applications: From two-
way to multiway component analysis. IEEE Signal Processing Magazine, 32(2):145-163, March
11
Published as a conference paper at ICLR 2025
2015. ISSN 1053-5888. doi: 10.1109/msp.2013.2297439. URL http: //dx.doi.org/10.]
/MSP.2013.2297439)
Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse
Autoencoders Find Highly Interpretable Features in Language Models. ICLR, January
2024. doi: 10.48550/arXiv.2309.08600. URL http://arxiv.org/abs/2309.08600.
arXiv:2309.08600 [cs].
Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated
convolutional networks, 2017.
Jacob Dunefsky, Philippe Chlenski, and Neel Nanda. Transcoders enable fine-
grained interpretable circuit analysis for language models, 2024. URL
ttps://www.alignmentforum.org/posts/Ymk jnWt ZGLbHRbzrP |
transcoders—enable-fine-grained-interpretable-circuith
Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak
coherent english?, 2023.
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann,
Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep
Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt,
Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and
Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread,
2021. https://transformer-circuits.pub/2021/framework/index.html.
Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: networks, 2019. URL https://arxiv.orqg/abs/1803.03635.
Finding sparse, trainable neural
Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever,
Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders, 2024. URL https :|
[//arxiv.orq/abs/2406.04093.
Xuyang Ge, Fukang Zhu, Wentao Shu, Junxuan Wang, Zhengfu He, and Xipeng Qiu. Auto-
matically identifying local and global circuits with linear computation graphs. arXiv preprint
arXiv:2405.13868, 2024.
Tan Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1.
MIT Press, 2016.
Tom Henighan, Shan Carter, Tristan Hume, Nelson Elhage, Robert Lasenby, Stanislav Fort,
Nicholas Schiefer, and Christopher Olah. Superposition, memorization, and double descent.
Transformer Circuits Thread, 2023. URL https://transformer—circuits.pub/| [2023/toy—double-descent/index.html,
Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, and Aleksander
Madry. Adversarial examples are not bugs, they are features. Advances in neural information
processing systems, 32, 2019.
Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and
Percy Liang. Concept bottleneck models, 2020. URL https://arxiv.org/abs/2007.]
04612,
Yanghao Li, Naiyan Wang, Jiaying Liu, and Xiaodi Hou. Factorized bilinear models for image
recognition. In Proceedings of the IEEE international conference on computer vision, pp. 2079—
2087, 2017.
Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John Hopcroft. Convergent learning: Do
different neural networks learn the same representations?, 2016. URL!
b 0754
Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji. Bilinear cnn models for fine-grained
visual recognition. In Proceedings of the IEEE international conference on computer vision, pp.
1449-1457, 2015.
Published as a conference paper at ICLR 2025
Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller.
Sparse feature circuits: Discovering and editing interpretable causal graphs in language models.
arXiv preprint arXiv:2403.19647, 2024.
Grégoire Montavon, Wojciech Samek, and Klaus-Robert Miiller. Methods for interpreting and un-
derstanding deep neural networks. Digital Signal Processing, 73:1-15, February 2018. ISSN
1051-2004. doi: 10.1016/j.dsp.2017.10.011. URL http://dx.doi.org/10.1016/7.]
dsp. 20 0.0
Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. Feature visualization. Distill, 2017. doi:
10.23915/distill.00007. https://distill.pub/2017/feature-visualization.
Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter.
Zoom In: An Introduction to Circuits. Distill, March 2020. URL https://distill.pub
2020/circuits/zoom-inl
Yannis Panagakis, Jean Kossaifi, Grigorios G. Chrysos, James Oldfield, Mihalis A. Nicolaou, Anima
Anandkumar, and Stefanos Zafeiriou. Tensor methods in computer vision and deep learning.
Proceedings of the IEEE, 109(5):863-890, May 2021. ISSN 1558-2256. doi: 10.1109/jproc.
2021.3074329. URLhttp://dx.doi.orq/10.1109/JPROC.2021.3074329,
Guilherme Penedo, Hynek Kydlitek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin
Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the
finest text data at scale, 2024. URLhttps://arxiv.org/abs/2406.17557,
Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of
black-box models, 2018. URL https://arxiv.org/abs/1806.07421.
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “why should i trust you?”: Explaining
the predictions of any classifier, 2016. URL https://arxiv.org/abs/1602.04938,
Lee Sharkey. A technical note on bilinear layers for interpretability. 2023.
Noam Shazeer. Glu variants improve transformer, 2020.
Nicholas D. Sidiropoulos, Lieven De Lathauwer, Xiao Fu, Kejun Huang, Evangelos E. Papalexakis,
and Christos Faloutsos. Tensor decomposition for signal processing and machine learning. JEEE
Transactions on Signal Processing, 65(13):3551-3582, 2017. doi: 10.1109/TSP.2017.2690524.
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks:
Visualising image classification models and saliency maps, 2014. URL https://arxiv.|
Marius Hobbhahn Stefan Heimersheim. solving-the-mechanistic-interpretability-challenges,
2023. URL https://www.alignment forum.orqg/posts/sTel8dNIDGywu9Dz6/| [solving-the-mechanistic-interpretability-challenges—eis-viil Ac-
cessed: 2024-09-02.
Aaquib Syed, Can Rager, and Arthur Conmy. Attribution patching outperforms automated circuit
discovery, 2023. URL|https://arxiv.org/abs/2310.10348,
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Niko-
lay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher,
Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy
Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshom,
Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel
Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee,
Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra,
Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi,
Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh
Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, lliyan Zarov, Yuchen
Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic,
Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models,
2023.
Published as a conference paper at ICLR 2025
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Fukasz Kaiser, and Illia Polosukhin. ~Attention is All you Need. NeurIPS, 30, 2017. URL
https://arxiv.orqg/abs/1706.03762
Chelsea Voss, Nick Cammarata, Gabriel Goh, Michael Petrov, Ludwig Schubert, Ben Egan,
Swee Kiat Lim, and Chris Olah. Visualizing weights. Distill, 2021. doi: 10.23915/distill.00024.
007. https:/distill.pub/2020/circuits/visualizing-weights.
Published as a conference paper at ICLR 2025
A EIGENSPECTRA: SHOWING EIGENVECTORS ACROSS DIGITS
The following are plots showing multiple positive and negative eigenvectors for certain digits. Posi-
tive features either tend to look for specific patterns in the target class (first eigenvector of 2, match-
ing the bottom part) or tend to match an archetypal pattern (second eigenvector of 6, matching the
whole digit). Negative eigenvectors tend to look for a specific part that would change the class of
the digit. For instance, if the pattern highlighted by the first negative eigenvector of 4 were on, it
would most likely be a 9.
Ty | . - .
= % g
20
g ’.'.. - '.'El'
0.5 2
Figure 10: eigenvectors for digit 2.
015
012
iy,
§ -
0.00 >
W
o
1l
i :}
20
015 013 0.10
/
R
=
20
0.12 0.14 0.7
\\
N
o
24
Figure 12: eigenvectors for digit 6.
Published as a conference paper at ICLR 2025
B REGULARIZATION & AUGMENTATION: ABLATIONS & OBSERVATIONS
Following the observation that regularization improves feature interpretability for image classifiers,
we study several popular regularization and augmentation techniques. In summary, input noise
sparsifies the features, while geometric transformations blur the features. Some popular techniques,
such as dropout, have little impact on features.
B.1 REGULARIZATION
Input noise has the largest impact on features from any of the explored techniques. Specifically,
we found dense Gaussian noise (already depicted in [Figure 4) to provide the best trade-off between
feature interpretability and accuracy. We also considered sparse salt-and-pepper noise (blacking or
whiting out pixels), which resulted in both lower accuracy and interpretability. Lastly, we explored
Perlin noise, which is spatially correlated and produces smooth patches of perturbation. However,
this performed worst of all, not fixing the overfitting.
Model noise adds random Gaussian noise to the activations. This had no measurable impact on any
of our experiments. However, this may simply be because our models are quite shallow.
Weight decay generally acts as a sparsifier for eigenvalues but does not significantly impact the
eigenvectors. This is extremely useful as it can zero out the long tail of unimportant eigenvalues,
strongly reducing the labor required to analyze a model fully (more details in [Appendix E).
Dropout did not seem to impact our models. Overfitting was still an issue, even for very high values
(> 0.5). We suspect this may change in larger or capacity-constrained models.
B.2 AUGMENTATION
Translation stretches features in all directions, making them smoother. The maximal shift (right)
is about 7 pixels in each direction, which is generally the maximal amount without losing impor-
tant information. Interestingly, translation does not avoid overfitting but rather results in smoother
overfitting patches. High translation results in split features, detecting the same pattern in different
locations (Figure T3). This also results in a higher rank.
Rotation affects the features in the expected manner. Since rotating the digit zero does not signifi-
cantly impact features, we consider the digit 5, which has a mix of rotation invariance and variance.
Again, it does not stop the model from learning overfitting patches near the edges without noise.
These features become broader with increased rotation.
Blur does not significantly affect features beyond making them somewhat smoother. Again, it still
overfits certain edge pixels in a blurred manner without noise.
0.14 E .- = »
oz - =
009 - |
e — -——
00 —
o - -
20
H P i
T
0.11 0.13 20
Figure 13: Important eigenvectors for a model trained with high translation regularization (7 pixels
on either side). Similar patterns manifest as multiple eigenvectors at different locations.
All these augmentations are shown separately in [Figure T4 Combining augmentations has the ex-
pected effect. For instance, blurring and rotation augmentation yield smooth and curvy features.
16
Published as a conference paper at ICLR 2025
0.4 norm
Noise
0 rorm
"
o
7 pixels
=2 =
Ny L
sa.0% 2% s EX s75%
Noise:
0.4 norm
- - - - * ']
= v v " - .
et et et ._'.._! _'_l
%
£ LEE o S, ok
aI .n Ea
Figure 14: Important eigenvectors for models with different hyperparameters.
Published as a conference paper at ICLR 2025
C EXPLAINABILITY: A SMALL CASE STUDY WITH EIGENVECTORS
‘While this paper focuses on bilinear layers for interpretability, the proposed techniques can also be
used for post-hoc explainability to understand what has gone wrong. Our explanations are generally
not as human-friendly as other methods but are fully grounded in the model’s weights. This section
explores explaining two test-set examples, one correctly classified and one incorrectly.
The figures are divided into three parts. The left line plots indicate the sorted eigenvector activation
strengths for the digits with the highest logits. The middle parts visualize the top positive and
negative eigenvectors for each digit. The right displays the input under study and the related logits.
The first example, a somewhat badly drawn five, results in about equal positive activations for the
output classes 5, 6, and 8 (which all somewhat match this digit). The negative eigenvectors are most
important in this classification, where class 5 is by far the least suppressed. This is an interesting
example of the model correctly classifying through suppression.
The second example, a seven-y looking two, is actually classified as a 7. From looking at the top
eigenvectors of the digit 2 (shown in [Figure 10), we see that the more horizontal top stroke and
more vertical slanted stroke activates the top eigenvector for the digit 7 more strongly than the 2-
eigenvectors that look for more curved and slanted strokes. The negative eigenvectors are not very
important in this incorrect classification.
s s s input
28 3.20 _= = —
— %
000
000
H 6 8 logits
-0.72 " -
ok -
2,08 [ -
251 10
Figure 15: Study of a correctly classified 5. The output is strongly influenced by negative eigenvec-
tors, resulting in strong suppression for the other digits.
= .
" - Ak
:;;; : 9 - =
o~ Ik =
o 2 7 8 logits
0.00 7
R L] 2
04 & . L
] O
087 - LE L
Figure 16: Study of a misclassified 2. The model mostly classifies twos based on the bottom line
and top curve, which are both only partially present.
18
Published as a conference paper at ICLR 2025
D HOSVD: FINDING THE MOST IMPORTANT SHARED FEATURES
In the case that no output features are available or we wish to find the dominant output directions,
we can use HOSVD on the B tensor (described in [subsection 3.3). Intuitively, this reveals the
most important shared features. We demonstrate this approach on the same MNIST model used in
Instead of contributing to a single output dimension, each interaction matrix can contribute to an
arbitrary direction, shown at the bottom right ("contributions™). Further, the importance of the con-
tributions is determined by their singular value, which is shown at the top right. The remainder of
the visualization shows the top eigenvalues and the corresponding eigenvectors.
The most important output direction separates digits with a prominent vertical line (1, 4, and 7)
from digits with a prominent horizontal line (5 specifically). Similarly, the second most important
direction splits horizontal from vertical lines but is more localized to the top half. Specifically, it
splits by the orientation of the top stroke (whether it starts/ends left or right).
+ eigenvalues + eigenvectors singular value
0.34 0.63
& H “’ ¢
o 20 o
o
- elgenvalues - elgenvectors contributions
= 12 = ii;
S8 &N
Figure 17: The most important output direction of an MNIST model roughly splits digits by its
horizontality or verticality.
+ eigenvalues + eigenvectors singular value
039 = -=_ - 0ss
) .-l‘
-y -
0 20 o
- eigenvalues - eigenvectors. contributions
20
Figure 18: The second most important output direction of an MNIST model splits digits according
to the orientation of its top stroke.
‘We observe that the output directions uncovered through HOSVD somewhat correspond to mean-
ingful concepts, albeit sometimes dominated by a specific digit (such as 5 and 6). Less significant
directions often highlight specific portions of digits that seem meaningful but are more challenging
to describe. In summary, while in the case of MNIST, the results are not particularly more inter-
pretable than decomposing according to digits, we believe this technique increases in utility (but
also computational cost) as the number of output classes increases.
19
Published as a conference paper at ICLR 2025
E SPARSITY: WEIGHT DECAY VERSUS INPUT NOISE
Throughout, we make the claims that input noise helps create cleaner eigenvectors and that weight
decay results in lower rank; this appendix aims to quantify these claims. To quantify near-sparsity,
we use (Ll/L2)2, which can be seen as a continuous version of the Ly norm, accounting for near-
zero values. We analyze both the eigenvalues, indicating the effective rank, and the top eigenvectors,
indicating the effective pixel count.
As visually observed in[Figure 4] this analysis (left of [Figure T9) shows that input noise plays a large
role in determining the eigenvector sparsity; weight decay does not. On the other hand, input noise
increases the number of important eigenvectors while weight decay decreases it. Intuitively, input
noise results in specialized but more eigenvectors, while weight decay lowers the rank.
Eigenvector Sparsity Eigenvalue Sparsity
Input Noise Input Noise
Figure 19: Measuring the approximate Lo norm of eigenvalues (left) and the top 5 eigenvectors
(right) with varying Gaussian input noise and weight decay.
F TRUNCATION & SIMILARITY: A COMPARISON ACROSS SIZES
This appendix contains a brief extension of the results presented in[Figure 5
Tunton Agess Szes Sy sro Eoumueciors I
H I
) B O 0 = - El Al . " " ol
Figure 20: A) The accuracy drop when truncating the model to a limited number of eigenvectors.
The first few eigenvectors result in a similar drop across sizes. Narrow models tend to naturally
remain more low-rank. In general, few eigenvectors are necessary to recover the full accuracy. B)
Inter-model size similarity of eigenvectors (using 300 as a comparison point). The top features for
similarly-sized models are mostly similar. C) A similarity comparison between all model sizes for
the top eigenvector.
20
Published as a conference paper at ICLR 2025
G EXPERIMENTAL SETUPS: A DETAILED DESCRIPTION
This section contains details about our architectures used and hyperparameters to help reproduce
results. More information can be found in our code [currently not referenced for anonymity].
G.1 IMAGE CLASSIFICATION SETUP
The image classification models in this paper consist of three parts: an embedding, the
bilinear layer, and the head/unembedding, as shown in The training hyperparameters
are found in However, since the model is small, these parameters (except input noise;
) do not affect results much.
Figure 21: The architecture of the MNIST model.
MNIST Training Parameters
input noise norm 0.5
weight decay 1.0
learning rate 0.001
batch size 2048
optimizer AdamW
schedule cosine annealing
epochs 20-100
Table 1: Training setup for the MNIST models, unless otherwise stated in the text.
G.2 LANGUAGE MODEL SETUP
The language model used in[section 3]is a 6-layer modern transformer model (Touvron et al}} [2023)
where the SWiGLU is replaced with a bilinear MLP (Figure 22). The model has about 33 million
parameters. The training setup is detailed in[Table 3] As the training dataset, we use a simplified
and cleaned version of TinyStories ) that remedies the following issues.
Contamination: About 20% of stories are exact duplicates (in both train and test).
Corruption: Some stories contain strange symbol sequences akin to data corruption.
Furthermore, we use a custom interpretability-first BPE tokenizer. The tokenizer is lower-case only,
splits on whitespace, and has a vocabulary of only 4096 tokens.
st (Narm — attn
E
00
B {‘" ! o
Norm e
r >
poqaun
Figure 22: Simplified depiction of a bilinear transformer model. A model dimension of 512 is used,
and an expansion factor of 4 (resulting in 2048 hidden dimensions in the MLP).
21
Published as a conference paper at ICLR 2025
TinyStories Training Parameters
weight decay 0.1
batch size 512
context length 256
learning rate 0.001
optimizer AdamW
schedule linear decay
epochs 5
tokens +2B
initialisation 2pt2
Table 2: Tinystories training setup. Omitted parameters are the HuggingFace defaults.
The models used in the experiments shown in
are trained of the FineWeb dataset (Penedo
. These follow the architecture of GPT2-
all (12 layers) and GPT2-medium (16 layers)
but have bilinear MLPs.
Their parameter count is 162M and 335M, respectively. Both use the
Mixtral tokenizer.
FineWeb Training Parameters
weight decay 0.1
batch size 512
context length 512
learning rate 6e-4
optimizer AdamW
schedule linear decay
tokens + 32B
initialisation 2pt2
Table 3: FineWeb training setup. Omitted parameters are the HuggingFace defaults.
G.3 SPARSE AUTOENCODER SETUP
All discussed SAEs use a TopK activation function, as described in [Gao et al]
k = 30 to strike a good balance between sparseness and reconstruction 1os
narrow dictionaries (4x expansion) for simplicity. The exact hyperparameters and the attained loss added (Equation 4) across layers is shown in[Figure 23]
). We found
n 5|studies quite
are shown in [Table 4}
Ladded =
Lypatch — Letean
Lpatch — Liclean 4
Letean ¢
Figure 23: Loss added for the mlp_out and resid_mid SAEs across layers.
22
Published as a conference paper at ICLR 2025
SAE Training Parameters
expansion 4x
k 30
batch size 4096
learning rate le-4
optimizer AdamW
schedule cosine annealing
tokens =+ 150M
buffer size +2M
normalize decoder True
tied encoder init True
encoder bias False
Table 4: SAE training hyperparameters.
H CORRELATION: ANALYZING THE IMPACT OF TRAINING TIME
[Figure 9] shows how a few eigenvectors capture the essence of SAE features. This section discusses
the impact of SAE quality, measured through training steps, on the resulting correlation. In short,
we find features of SAEs that are trained longer are better approximated with few eigenvectors.
‘We train 5 SAEs with an expansion factor of 16 on the output of the MLP at layer 12 of the ‘fw-
medium’ model. Each is trained twice as long as the last. The feature approximation correlations
are computed over 100K activations; features with less than 10 activations (of which there are less
than 1000) are considered dead and not shown. The reconstruction error and loss recovered between
SAE:s differ only by 10% while the correlation mean changes drastically (Table 3). The correlation
distribution is strongly bimodal for the ‘under-trained” SAEs (shown in[f with darker col-
ors). Given more training time, this distribution shifts towards higher correlations. The activation
frequencies of features are mostly uncorrelated with their approximations.
Training steps (relative) 1 2 4 8 16
Normalized MSE 0.17 0.16 0.16 0.15 0.15
Loss recovered 0.60 0.61 0.65 0.65 0.66
Correlation mean 0.17 028 042 052 059
Table 5: The SAE metrics along with the mean of the correlations shown in[Figure 24 The correla-
tion improves strongly with longer training, while other metrics only change marginally.
Count.
o 025 05
Correlation (active only)
Figure 24: Feature approximation correlations using two eigenvectors across SAEs with different
training times. Darker is shorter, and bright is longer. This shows a clear bimodal distribution for
‘under-trained” SAEs, which vanishes upon longer training, indicating some form of convergence.
23
Published as a conference paper at ICLR 2025
I BILINEAR TRANSFORMERS: A LOSS COMPARISON
‘While experiments on large models show bilinear layers to only marginally lag behind SwiGLU
(Shazeer} 2020), this section quantifies this accuracy trade-off through compute efficiency. We per-
formed our experiments using a 6-layer transformer model trained on TinyStories. For these exper-
iments, we use d_model = 512 and d_hidden = 2048, resulting in roughly 30 million parameters.
However, we have found these results to hold across all sizes we tried.
‘ Bilinear ReGLU SwiGLU
constant epochs 1.337 1.332 1.321
constant time 1.337 1.337 1.336
Table 6: The loss of language models with varying MLP activation functions. Bilinear layers are
6% less data efficient but equally compute efficient.
Considering the data efficiency (constant epochs), both SwiGLU and ReGLU marginally beat the
bilinear variant. Concretely, SwiGLU attains the same final loss of the bilinear variant in 6% less
epochs. On the other hand, when considering compute efficiency (constant time), we see that these
differences vanishfl Consequently, if data is abundant, there is little disadvantage to using bilinear
layers over other variants.
J FINETUNING: YOUR TRANSFORMER IS SECRETLY BILINEAR
Many state-of-the-art open-source models use a gated MLP called SwiGLU (Touvron et al, 2023).
This uses the following activation function Swishg(z) = z - sigmoid(8z). We can vary the 3
parameter to represent common activation functions. If 3 = 1, that corresponds to SiLU activation,
used by many current state-of-the-art models. 5 = 1.7 approximates a GELU and 3 = 0 is simply
linear, corresponding to our setup. Consequently, we can fine-tune away the gate by interpolating 3
from its original value to zero. This gradually converts an ordinary MLP into its bilinear variant.
To demonstrate how this approach performs, we fine-tuned TinyLlama-1.1B, a 1.1 billion-parameter
transformer model pretrained on 3 trillion tokens of data, using a single A40 GPU. For simplicity,
we trained on a slice of FineWeb data. Due to computing constraints, we only tried a single schedule
that linearly interpolates towards 3 = 0 during the first 30% (120M tokens) and then fine-tunes for
the remaining 70% (280M tokens). We compare this to a baseline that does not vary 3 during fine-
tuning, corresponding to continued training. We use this baseline to compensate for the difference
in the pretraining distribution of TinyLlama (consisting of a mixture of RedPajama and StarCoder
data). This shows that this fine-tuning process increases the loss by about (0.05) but seems to benefit
from continued training (Figure 25). We plan to extend this result with a more thorough search
for an improved schedule, which will probably result in a lower final loss. We also expect longer
training runs to close to gap even further.
w5 o an - s o
Figure 25: Comparison of fine-tuning versus a baseline over the course of 400M tokens. The loss
difference is noticeable but decreases quickly with continued training.
! An improvement in implementation efficiency, such as fusing kernels, may change this fact.
24
Published as a conference paper at ICLR 2025
K TENSOR DECOMPOSITIONS: EXTRACTING SHARED FEATURES
Given a complete or over-complete set of m u-vectors, we can re-express B in terms of the eigen-
vectors, which amounts to a change of basis. To avoid multiple contributions from similar u-vectors,
we have to use the pseudo-inverse, which generalizes the inverse for non-square matrices. Taking
the u-vectors as the columns of U, the pseudo-inverse U+ satisfies UU* = I, as long as U is full
rank (equal to d). Then
m
B=) uj®Q )
E m d
= ZZ)\(ICJ) U ® Ui} @ Uik} (6)
ki
where u}’c are the rows of Ut and Q, = i Mk,iyVik,iy @ Uik, is the eigendecomposition of
the interaction matrix corresponding for uk :. We can then recover the interaction matrices from
Q. = U -ou B using the fact that wy, - ui, = Ok (Kronecker delta). Note that the eigenvectors
within a single output direction k are orthogonal but will overlap when comparing across different
output directions.
L BILINEAR LAYERS: A PRACTICAL GUIDE
Bilinear layers are inherently quadratic; they can only model the importance of pairs of features,
not single features. Interestingly, we haven’t found this to be an issue in real-world tasks. However,
linear structure is important for some toy tasks and, therefore, merits some reflection. Without
modification, bilinear layers will model this linear relation as a quadratic function. To resolve this,
we can add biases to the layer as follows: BL(z) = (Wx +b) ® (V& + ¢). In contrast to ordinary
layers, where the bias acts as a constant value, here it acts as both a linear and a constant value. This
becomes apparent when expanded:
BL(z) = Wz © Va + (cWz +bVz) + cb
‘We disambiguate by calling the terms ‘interaction’, ‘linear’, and ‘constant’. Theoretically, this is
very expressive; all binary gates and most mathematical operations can be approximated quite well
with it. In practice, the training process often fails to leverage this flexibility and degenerates to
using quadratic invariances instead of learned constants.
M ADVERSARIAL MASKS: ADDITIONAL FIGURE
A) [FE v, kg
e
~B)
g
Figure 26: More examples of adversarial masks constructed from specific eigenvectors for models
with A) no regularization, B) Gaussian noise regularization with std 0.15, and C) Gaussian noise
regularization with std 0.3.
N LANGUAGE: FURTHER DETAILS FOR FEATURE CIRCUITS
25
Published as a conference paper at ICLR 2025
=
Selfinteraction Croszinteractions
—— Interacton Matrix
=~ Random Symmetric Matrix
Counts
Eigenvalue
03 02 -01 00 01 02 03 04 0 w0 20 w0 40 50
Interaction Eigenvector index
Figure 27: A) Histogram of self-interactions and cross-interactions. B) The eigenvalue spectrum
for the sentiment negation feature discussed in [section 5] The dashed red line gives spectrum for a
random symmetric matrix with Gaussian distributed entries with the same standard deviation.
O INPUT FEATURES OF THE NEGATION FEATURE
Output Feature (1882): not good
Hly wanted to climb th ladder ,but her mommy old her to stay. away from it bcmmuw.n. safe . lilylistened to her
they looked everywhere . the trainwas lost . the rain stayed inthe hole. and. “m. happy . the end . [EOS]
the manwas angry he grabbed sue .who screamed and cried . sue was very troubled Anz- U © wach the phy md
and
illcoutdn U find her beloved jacket . she feliso sad that m- plying it her | ends and [Sopped| |
b dog . and mexsid 1 ddn Utk the big dog . he wam [Wronest ke me oy
sick . they called the doctor , who said he was very | 1 . sadly . the small boy
‘-rzmvz red and he passed
Output Feature (1179): not bad
metothepk car1 > e car ddn 1 o ond bty it [f] e s o oy
s bee sk e it bt mommnyesplaned that the bes s vindly and wousd [RB o e s was cumons
s prowd ofmsetr. hewas scaed 1o cotetthe emons bt be i~ ] e wp . he coneaeaan
became riends. they went_on_many sivntres ogter. hebearwas st ey bt din [ mind e was ooy o
made many new friends and had lots of fun . the lile rog was always very busy b'umz-mrgm the cow s advice
o sy e g i g the sowman i cod_and et ttom does [ mindne whes e sovman
Table 7: Output SAE features in the negation feature discussed in
26
Published as a conference paper at ICLR 2025
Input Feature (326): crashing and breaking
but . as the lady was camying the birdic across he road aidn them both . the birdic
takes an egg and aps iton the edge of thebowl . but he tap s m-._.. breaks inhis -
and ;.mln. feels dizzy from e peed . he does mot the his
Input Feature (1376): dangerous actions
o sweete . [you an - vown e ion - Y [ o por e
g
it thm tom was wored e sid” e we shoud novtouch the _baoons | [T Y [ w5 | mashe
waswortedad wid o sam_ you mustn g0 i I.-..- < buam
Input Feature (1636): nervous / worried
bigadvenure e deidd 0 ake a1 o th . joewas soexied. but ) [ 8 e woried w0 e
o peopl doim yoga o was st and he clapped withexcemen b he [RRGY (NG + ¢ womy - one
forest and soon he spotted alovely tree with lots of delicious fruit . he was excited , but
F
Input Feature (123): negative attribute
mas;nm\.m.ummmmmmgll... n o M
[BOS] once upon atime ,there was a beach . the beach was very filthy.
f
10y hello. she ran to the door and knocked on it but it was lthy .-.. the new kid opened
Table 8: SAE features that contribute to the negation feature discussed in|
Published as a conference paper at ICLR 2025
Input Feature (990): a bad turn
she wentup to him and asked him why he was crying . the litle boy .I.l. my i m
wiiticboy ayimg . shewenttoim sndsskedwhy he s sad o o boy [0 [ o e oy nd ian -
h . spotasked. " whyae you sad.lile bid 7 "the bid replied ll. my favorite .. the
Input Feature (1929): inability to do something / failure
* Ulisento him - again and - L obu e water wouldn umml.. pleae amd aked nicely .
couldn” treach it on_ the counter and o high I._I.. ame ino the
Input Feature (491): body parts
abucket and a shovel and start 1o il the bucket withsoil . they press the soilhard with - they tum the bucket
day he hopped up 1o a magazine that was ying on the ground . he poked it mm.-.xm‘i it elt hard
spideron e coting . she waned ot 50 e rscht wp to_pinch 0wt R [nges v e sider s
Input Feature (947 : bad ending (seriously)
Input Feature (882): being positive
e was a musician who wandered through the countryside . he ws quite weak . but ..-. make it tothe
finger ©she yelped and quickly pulled away . the lite girlwas abitscaredbut ...- she did ot
was tll and had long . bony fingers . the It boy was abit scared . but ...- he went
Input Feature (240): negation of attribute
it skl the ters whst s e was 0 xcid o konow o v Jf] 0 e g
xplosbig unne snd ss whathe ight i e oo ot it v ] st e sndve st
‘0 swimming in the ocean . he asked his mom 1 take him . his mom explained that u\uA.
afe 10 swim alone and she
Input Feature (766
inability to perform physical actions
otone iy, e ot dck ad s ot i by hurt nd ecouan [ play it s s amymrs
[BOS] one day . aprettybird hun its wing it could . fly . akind girl amed
Lo play with him 100 _ one day . the puppy fell down and hurthis leg hzcnuld. play and run ke before
Input Feature (1604): avoiding bad things
in more et vy 0 e s playing more ety making st wout” ] st 0 iy e et
et he sl snd sy was ol o ol ft th et 1t ot [ k. e prvay showed
jane notced the ice was very icy . mary suggested they wear gloves . s tht the cold would . hur their hands . jane agreed
Input Feature (1395): positive ending
the okd man watched the pony with a smilk on his face . the pony was happy mfl.- scared . she had found a
and e sccing o n the woods and choosing 0 sy andspprcitent. e B8 g o] e . eosy
sam was happy . and 50 was the bl now Axmxmmszllmumwxymgnhzr-. they roled and bounced
Table 9: SAE features that contribute to the negation feature discussed in continued).
28