## May a derivative pass through an expectation?

AI objectives often average losses over randomness. Moving a derivative through that expectation requires justification. The lab compares two functions under a fixed Uniform[0,1] law: a smooth quadratic and a moving-event indicator. The parameter stays between 0.25 and 0.75, so centered difference quotients do not cross the interval boundaries.

$$
\frac d{d\theta}E[g_\theta(X)]\stackrel{?}{=}E[\partial_\theta g_\theta(X)].
$$

## Recast the question using difference quotients

For the smooth example, the centered quotient equals 2θx² exactly. In a neighborhood of θ it has an integrable common bound, so dominated convergence justifies interchange. Expected loss is θ²/3 and its derivative 2θ/3, giving one third at θ=0.5. A useful sufficient condition is almost-everywhere convergence of the quotients together with one integrable bound for all sufficiently small h.

$$
D_h(x)=\frac{g_{\theta+h}(x)-g_{\theta-h}(x)}{2h},\qquad |D_h(x)|\le G(x),\quad G\in L^1.
$$

## An almost-everywhere derivative is not enough

The moving indicator has expectation θ, hence derivative one. At fixed θ, the pointwise derivative is zero for almost every x other than the moving boundary. Its expectation is therefore zero. The centered quotient is a spike of width 2h and height 1/(2h). It converges to zero almost everywhere while keeping integral one, just like a concentrating-mass counterexample.

$$
E[\mathbf1_{(0,\theta)}(X)]=\theta,\quad\frac d{d\theta}E[g_\theta]=1,\quad E[\partial_\theta g_\theta]=0.
$$

## Apply the conditions to an actual model

There is no integrable common dominator for these quotients. Existence of the derivative for almost every x at each fixed θ is insufficient. If the sampling density also changes with θ, its derivative or moving-support contributions must be handled too. Reparameterization supplies a fixed-noise computation path but does not automatically supply integration hypotheses. Endpoint conventions or assigning a derivative at one exceptional point do not repair the integral mismatch.

$$
\lim_{h\to0}\int D_h\,dP=1\ne0=\int\lim_{h\to0}D_h\,dP\quad\text{in the indicator example}.
$$

## Exercises with explanations

1. What if the spike width halves? **Height doubles, preserving integral one.**
2. Can changing a derivative at one point fix the counterexample? **No: one point has measure zero.**
3. Does a smooth-looking plot justify interchange? **No: verify actual integrability conditions.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
