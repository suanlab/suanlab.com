## Pull an output event back

To find the law of Y=g(X), ask which input outcomes satisfy g(X)≤y. This event-based method is more fundamental than an inverse-function formula and still works for many-to-one or constant maps. The lab compares Y=aX and Y=aX² with integer scale a. X is uniform on [0,1] in linear mode and [−1,1] in square mode. The plot shows the output CDF.

$$
F_Y(y)=P(g(X)\le y).
$$

## Why the Jacobian is absolute

For a differentiable one-to-one transform, preserving mass in a small interval introduces the absolute derivative of the inverse. If X is uniform on [0,1] and Y=2X, the support doubles in width and density halves. Negative a reverses orientation but cannot make density negative. In several dimensions, the corresponding factor is the absolute determinant of the inverse Jacobian.

$$
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|.
$$

## Squaring has two preimages

For Y=X² and 0<y<1, both positive and negative square roots map to y. Each branch has input density one half and inverse derivative magnitude 1/(2√y). Adding them gives 1/(2√y). At y=0.25, density is one and the CDF is one half. Keeping only one inverse branch loses half the total mass.

$$
f_Y(y)=\sum_{x:g(x)=y}\frac{f_X(x)}{|g^{\prime}(x)|}=\frac1{2\sqrt y},\quad F_Y(y)=\sqrt y\quad(0<y<1).
$$

## Boundaries and constant transforms

The square-mode density diverges near zero but remains integrable. At the boundary the interface gives no finite density readout and flags the unbounded limit. Choosing a density value at one point does not change probabilities. At a=0, the output is always zero: point mass one and no Lebesgue density. Dividing by a zero Jacobian is invalid. Density calculations in invertible generative models rely on these assumptions; the formula does not apply unchanged to every neural network.

$$
a=0\implies P(Y=0)=1,\qquad a\ne0:\;f_{aX}(y)=\frac1{|a|}f_X(y/a).
$$

## Exercises with explanations

1. What is the linear density for a=−2? **One half inside (−2,0).**
2. Does squaring have one inverse branch? **No: two at positive interior values.**
3. Does an unbounded density imply an atom at zero? **No: nonzero-scale point probability remains zero.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
