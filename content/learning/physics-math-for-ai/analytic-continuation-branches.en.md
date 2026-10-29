## A logarithm needs a branch choice

For z=re^{iθ},r>0, all values

$$
\log z=\log r+i(\theta+2\pi k),\qquad k\in\mathbb Z
$$

exponentiate to z. A branch consistently selects values. The usual principal holomorphic branch excludes the negative real axis and uses angles in (−π,π). Assigning a boundary angle does not make the function continuous across the cut. Probe endpoints avoid that cut.

## Continue angle along a path

Start at positive r and reach angle θ after k extra turns. The continuous lifted angle is Θ=θ+2πk:

$$
L_k=\ln r+i(\theta+2\pi k),\qquad e^{L_k}=z.
$$

This is a lift along a specified path, not a global single-valued logarithm on the punctured plane. A closed turn adds 2πi, obstructing such a global continuous choice.

## Square-root monodromy

Halving the lifted logarithm gives

$$
w_k=\sqrt r\exp\left(\frac i2(\theta+2\pi k)\right)=(-1)^k w_0.
$$

An odd turn flips sign; two turns restore the root. At endpoint i the principal root is (1+i)/√2, and an extra turn gives its negative. After two turns the root returns while the logarithm still differs by 4πi.

## Local uniqueness versus global path independence

The identity theorem supports uniqueness of holomorphic continuation on connected overlaps under its hypotheses. Continuing around a multiply connected domain can still encounter a global single-valuedness obstruction. Local uniqueness and independence of every continuation path are different claims.

Similarly, the Taylor series of 1/(1−z) at zero converges only for |z|<1, whereas the rational expression exists away from z=1. A series disk and a function’s extended domain need not coincide.

## Read the two paths

Blue shows z, orange the continued root w. They are different variables displayed on equal-unit complex axes. Overlapping paths do not erase winding information.

Software using complex log or sqrt must state branch conventions. Local automatic derivatives do not remove a discontinuous branch choice across a cut. Phase unwrapping likewise separates principal from accumulated phase, with extra sampling and noise issues.

## Exercises and answers

1. One continued-log turn adds **2πi**.
2. A square root returns after **two turns**.
3. The endpoint alone **does not specify the continued logarithm without winding information**.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
