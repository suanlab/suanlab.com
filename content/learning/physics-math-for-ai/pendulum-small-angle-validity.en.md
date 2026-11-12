## A useful approximation needs a question and an error budget

The statement that a pendulum has a constant period is useful for small oscillations, but incomplete as a validity claim. Which amplitude is allowed? Are we predicting a period, a restoring force, or the angle at a much later time? How large an error can the application tolerate? A model can answer one question accurately while missing another.

This lesson will compare the period of an ideal nonlinear pendulum with two small-angle formulas. The reference is a declared parent model: a point mass on a massless rigid link of length L, moving in a vertical plane in uniform gravity, without drag or pivot friction. Solving that model accurately does not establish its accuracy for every real pendulum. Material flexibility, air resistance and other omitted effects remain separate physical assumptions.

## Start from the nonlinear restoring law

Let theta be the angular displacement from the stable downward position, measured in radians. The equation is

$$
\theta''(t)+\frac{g}{L}\sin\theta(t)=0.
$$

Release the pendulum from theta(0)=alpha with zero angular velocity. For a nonzero amplitude below pi, the motion oscillates between minus alpha and plus alpha. The angular restoring term is sin(theta), whereas the linearized model replaces it with theta. That replacement is justified locally by the Taylor expansion sin(theta)=theta-theta cubed/6 plus higher terms.

A degree number must be converted to radians before evaluating these expansions. Substituting an input such as thirty directly into a formula derived for radians produces a different and incorrect approximation. Dimension-one angles still need a declared angular convention.

## Derive a period reference through energy

Multiply the equation by angular velocity and integrate. The conserved energy per mass is

$$
\frac{L^2}{2}(\theta')^2+gL(1-\cos\theta)
=gL(1-\cos\alpha).
$$

On a quarter swing, separating time from angle gives an integral with a turning-point singularity. The substitution sin(theta/2)=sin(alpha/2)*sin(phi) transforms it to a smooth integrand for amplitudes strictly below pi:

$$
T=4\sqrt{L/g}\,K(k),\qquad k=\sin(\alpha/2),
\qquad K(k)=\int_0^{\pi/2}\frac{d\phi}{\sqrt{1-k^2\sin^2\phi}}.
$$

Here k is the elliptic modulus. Some numerical libraries instead accept the parameter m=k squared. Confusing these two conventions is a software error, not a failure of the physical approximation. An independent quadrature of this integral provides a useful check on a special-function implementation.

## Compare two approximations to the same observable

The linearized equation predicts

$$
T_0=2\pi\sqrt{L/g}.
$$

The first amplitude correction is

$$
T_2=T_0\left(1+\frac{\alpha^2}{16}\right).
$$

Both formulas concern period. The second is an asymptotic correction as the amplitude approaches zero; it is not a general replacement for the nonlinear motion. Higher-order terms eventually matter. At a ninety degree release, for example, the reference ratio T/T0 is approximately 1.180340599. The linear formula underestimates the reference period by about 15.28 percent when the denominator is the reference period, while the corrected formula still underestimates it by about 2.21 percent.

Saying the nonlinear period is about 18.03 percent longer than T0 uses a different denominator. Both numerical descriptions can be consistent, but an error criterion must choose one denominator explicitly. Otherwise a reader cannot reproduce a claimed tolerance.

## Work through three amplitude scales

At five degrees, T/T0 is approximately 1.00047617249. The linear relative error using reference T is approximately 0.000475945854, or0.04759 percent. The quadratic correction reduces that error to about 0.00002077 percent under the same parent model.

At thirty degrees, T/T0 is approximately 1.01740879760. The linear relative error is about1.71109 percent, so it fails a one percent period requirement. The corrected formula has relative error about 0.02694 percent, passing that particular requirement at this amplitude.

At ninety degrees, the corrected formula fails a one percent requirement as well. Adding a term improves an approximation in some regime; it does not make it uniformly accurate over all physically possible inputs. These examples should be accompanied by the requested tolerance rather than a universal label such as small or large.

## Handle the equilibrium and the separatrix honestly

At exactly zero release angle with zero initial velocity, the pendulum stays at equilibrium. There is no nontrivial observed oscillation whose period can be measured. The limiting small oscillation period T0 is well defined, but it should be labeled as a limit rather than a measured period of the stationary trajectory.

As alpha approaches pi from below, the ideal period grows without bound. An exactly upright pendulum released perfectly at rest is another stationary solution, unstable to perturbations, rather than an ordinary finite-period oscillation. The ideal separatrix separates qualitatively different motions. A control range bounded below pi avoids treating a divergent integral as an ordinary finite number, while the text should still explain that limit.

## Compute accurately without hiding numerical error

For positive arguments the arithmetic-geometric mean iterates arithmetic and geometric averages toward a common limit. Its relationship to K gives another evaluation route:

$$
K(k)=\frac{\pi}{2\operatorname{AGM}(1,\sqrt{1-k^2})}.
$$

For amplitudes below pi, the complementary modulus equals cos(alpha/2) and remains positive. The mathematical iteration provides converging upper and lower sequences. Floating-point implementations still have rounding error, so numerical output should not be called an exact certificate merely because the underlying identity is exact.

The relevant separation is between numerical evaluation error in the parent-model reference and truncation error of T0 or T2 against that reference. Comparing an AGM result with independent high-precision quadrature helps test the former. Computing period differences tests the latter. Neither test supplies empirical evidence about drag or the rigidity of a real apparatus.

## A period graph is not an exact nonlinear trajectory

Replacing a cosine's period with the nonlinear period does not make the cosine the nonlinear pendulum solution. The actual waveform also changes with amplitude. A visualization can show a force-angle graph and a period-amplitude graph without claiming an exact time waveform.

This distinction matters for phase-sensitive prediction. A small period error accumulates over many cycles, and a period correction alone may still miss the instantaneous angle. Accuracy of a period, a peak angle and a long-time trajectory are different observable-specific claims. A useful research report states which one was evaluated and over what time interval.

## Connect the validity question to learned surrogates

A surrogate trained on small amplitudes may learn the harmonic period extremely well while extrapolating poorly toward the separatrix. A tiny training loss inside one amplitude range does not establish physical accuracy outside it. The chosen input range and error metric are part of the model's specification.

Dimensional scaling offers a useful check: under this ideal model changing L or g rescales both the reference and approximate periods through sqrt(L/g), while the normalized ratio depends on amplitude. This does not mean every real pendulum follows that scaling once omitted effects become important. A physically meaningful benchmark separates parent-model approximation, numerical evaluation and empirical model discrepancy.

## Research extension: Bound a period approximation and its accumulated return-time error

### Replace a small numerical difference with a controlled remainder

Keep the ideal pendulum assumptions and an oscillation amplitude strictly between zero and pi radians. Write m=sin²(alpha/2), so 0<m<1. Here m is the elliptic parameter, not the pendulum mass; the modulus in the earlier integral is sqrt(m). Expanding the integrand in nonnegative binomial powers and integrating its sine powers gives a positive series for the period ratio R=T/T_0:

$$
R=\sum_{n=0}^{\infty}a_nm^n,\qquad
a_n=\left({\binom{2n}{n}\over4^n}\right)^2,\qquad
{a_{n+1}\over a_n}=\left({2n+1\over2n+2}\right)^2<1.
$$

The underlying elliptic expansion and AGM relation are recorded in [DLMF 19.5](https://dlmf.nist.gov/19.5) and [19.8](https://dlmf.nist.gov/19.8). The following explicit geometric remainder follows directly from the decreasing coefficients. For S_J retaining powers zero through J, every later coefficient is at most a_(J+1), hence

$$
0\le R-S_J\le B_J={a_{J+1}m^{J+1}\over1-m},\qquad
S_J=\sum_{n=0}^{J}a_nm^n.
$$

At alpha=60 degrees, m=1/4 exactly, so both interval endpoints are rational numbers. The code checks six such bounds against a separately evaluated 90-digit AGM value. The rational bound is established by the series argument; the finite-precision AGM check is an independent numerical consistency check, not a proof that its rounded value is exact. Near the separatrix m approaches one and this particular bound becomes less efficient.

### A good period is not automatically a good long-time clock

Let a chosen approximation have period T_Q=T_0 Q, with Q positive. At the Nth numbered return to the release phase, the time discrepancy is N times the one-period discrepancy. Combining this exact clock statement with the interval above gives

$$
\left|NT-NT_Q\right|=NT_0|R-Q|
\le NT_0\max\left(|S_J-Q|,|S_J+B_J-Q|\right).
$$

Truncation order J and event count N are independent choices. A timing budget therefore needs a number of cycles as well as a per-period tolerance. For the 60-degree pendulum, R is about 1.073182007. After 100 corresponding cycles, the linear period's timing mismatch is about 7.3182 T_0. Using the alpha-squared corrected period reduces it to about 0.4643 T_0, but does not eliminate accumulation.

![Timing mismatch at numbered return events grows linearly with event count for a 60-degree pendulum; the corrected small-angle period reduces the drift relative to the linear period.](/assets/learning/physical-period-clock-en.svg)

This compares periodic clocks. It does not replace the nonlinear motion by a cosine or certify displacement error between returns. At zero amplitude the actual resting state has no nontrivial return period; T_0 remains a limiting quantity. At the upright equilibrium the oscillatory family also ends. Those exclusions from the original lesson remain essential.

| Claim | Information required | What remains unproved |
|---|---|---|
| Bound the period ratio | Amplitude and an elliptic-series remainder | Accuracy of a guessed waveform |
| Bound numbered return times | Period bound and the event count | Uniform position error between events |
| Apply the ideal prediction experimentally | Adequate gravity, geometry and drag assumptions | Empirical validity from a dimension check alone |

**Worked question:** Can a fixed one-percent period rule guarantee a fixed absolute timing error for arbitrarily many cycles? **Answer:** No. Unless the period is exact, the numbered-event error grows linearly with cycle count. Choose a time horizon and observable before assigning a tolerance. The code uses exact fractions for the rational series bounds and explicitly labels the corrected-period clock as a numerical calculation.

```python
from fractions import Fraction as F
from decimal import Decimal as D,localcontext
from math import comb,pi
with localcontext() as ctx:
    ctx.prec=90
    a,b=D(1),(D(3)/4).sqrt()
    for _ in range(20): a,b=(a+b)/2,(a*b).sqrt()
    reference=1/a
    for order in [1,2,4,8,16,32]:
        coeff=[F(comb(2*n,n),4**n)**2 for n in range(order+2)]
        lower=sum(coeff[n]*F(1,4)**n for n in range(order+1))
        width=coeff[-1]*F(1,4)**(order+1)/F(3,4)
        lo=D(lower.numerator)/D(lower.denominator)
        hi=D((lower+width).numerator)/D((lower+width).denominator)
        assert lo<reference<hi
        print('order',order,'interval width',float(width))
    ratio=float(reference)
    for Q in [1,1+(pi/3)**2/16]:
        print('100 numbered returns, mismatch in T0 units',100*abs(ratio-Q))
```

## Exercises with worked solutions

1. **Exercise:** Why use radians in the correction alpha squared/16? **Solution:** Its Taylor derivation assumes the radian convention; degree values require conversion by pi/180.
2. **Exercise:** Does a stationary zero amplitude release have a measured period T0? **Solution:** No. T0 is the limiting small oscillation period, not an observed cycle of that stationary state.
3. **Exercise:** At thirty degrees, does T0 meet a one percent period criterion? **Solution:** No. Its relative error against the reference is approximately 1.71109 percent.
4. **Exercise:** Does T2 meet that criterion at ninety degrees? **Solution:** No. Its relative error is approximately 2.21360 percent.
5. **Exercise:** Can a cosine with the corrected period be labeled the exact nonlinear trajectory? **Solution:** No. Matching the period does not reproduce the nonlinear waveform.
6. **Exercise:** Does agreement between AGM and quadrature validate the no-drag assumption? **Solution:** No. It checks two evaluations of the same ideal-model reference; empirical model validity needs other evidence.

## Sources and further study

[NIST DLMF on the pendulum](https://dlmf.nist.gov/22.19) gives the nonlinear equation and elliptic-function period. [NIST DLMF on the arithmetic-geometric mean](https://dlmf.nist.gov/19.8) gives the complete-elliptic-integral identity. The observable-specific comparisons and draft worked values above were independently evaluated for this lesson. 
