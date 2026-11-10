## Logarithm base sets the information unit

Base-two logarithms give bits; natural logarithms give nats. The underlying distribution does not change when the unit changes. A fair binary distribution has entropy one bit or ln2≈.693147 nats.

$$
H_{\rm nats}=H_{\rm bits}\ln2,\qquad H_{\rm bits}=H_{\rm nats}/\ln2.
$$

## Normalize logits into probabilities

Neural-network logits may be negative and need not sum to one. Before calculating probability entropy, softmax produces normalized masses. Subtracting the maximum logit before exponentiation avoids overflow.

$$
p_i=\frac{e^{\ell_i}}{\sum_j e^{\ell_j}},\qquad \log Z=m+\log\sum_j e^{\ell_j-m},\quad m=\max_j\ell_j.
$$

Adding the same constant to every logit cancels between numerator and denominator, preserving probabilities while changing the normalizer. Compare (0,0) and (4,4): both give (.5,.5), while log Z shifts by four.

## Separate example: masses versus density heights

The plot shows a continuous Uniform[0,a] distribution, separate from softmax. Density is 1/a and probability is area. At a=.25, density is four but total area remains 4×.25=1. Density heights may exceed one because they are not probabilities.

$$
h(X)=-\int_0^a\frac1a\ln\frac1a\,dx=\ln a.
$$

Differential entropy at a=.25 is about −1.38629 nats. Discrete entropy's nonnegativity does not extend to densities in this way. Rescaling coordinates by positive c changes differential entropy according to h(cX)=h(X)+ln c.

## Specify bin width to connect discrete and continuous values

Divide the interval into K equal bins of width Δ=a/K. Each has probability 1/K, so bin-index entropy is ln K. For this uniform example, the following is an exact identity:

$$
H_{\rm bin}=h(X)-\ln\Delta=\ln K.
$$

Blue shows full density; orange outlines the first bin. Its probability is area (1/a)Δ=1/K, not height. Fine-quantization formulas for general nonuniform densities may instead be approximations requiring additional conditions.

## Record conventions in AI experiments

Report whether losses are sums or per-token or per-sample averages, which log base is used, whether values are logits or normalized probabilities, and whether the model describes continuous densities or discrete masses. Identically named quantities need not be directly comparable when these choices differ. Logit controls affect the discrete panel; a and K affect the separate continuous-bin illustration.

## Exercises with explanations

1. How many bits are in one nat? **1/ln2≈1.443 bits.**
2. Is density height four an impossible probability? **No: integrate its area.**
3. Does changing a at fixed K=4 change bin-index entropy? **No: four equiprobable bins retain entropy ln4.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
