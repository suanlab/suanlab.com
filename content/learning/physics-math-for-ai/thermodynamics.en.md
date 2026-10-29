## From microscopic probabilities to physical quantities

Statistical mechanics connects probabilities of microscopic states with macroscopic energy and entropy. This connection needs a physical ensemble and units. A generic neural-network loss called “energy” does not automatically acquire temperature, heat or thermodynamic meaning.

Consider equilibrium with a thermal reservoir, one ground microstate of energy zero and $g$ excited microstates each of energy $\Delta>0$. Use units with Boltzmann’s constant $k_B=1$, so temperature has the corresponding energy units. The finite model is exact under these assumptions; it contains no phase transition or spatial interactions.

## Define the canonical weights

A microstate of energy $E_i$ receives weight $e^{-E_i/T}$. The partition function sums these weights:

$$
Z=1+g e^{-\Delta/T},\qquad
p_0=1/Z,\quad p_j=e^{-\Delta/T}/Z\ (j=1,\ldots,g).
$$

The excited energy-level occupancy is $P_1=g e^{-\Delta/T}/Z$. It sums $g$ microscopic probabilities. This distinction is crucial when counting entropy.

| Quantity | Definition |
| --- | --- |
| $Z$ | Dimensionless normalization |
| $U$ | Mean energy $\Delta P_1$ |
| $F$ | Helmholtz free energy $-T\log Z$ |
| $S$ | Entropy $-\sum_i p_i\log p_i$ in units $k_B=1$ |

## Work the two-state reference

With $\Delta=T=1,g=1$, $Z=1+e^{-1}\approx1.367879$. Excited occupancy is $1/(1+e)\approx0.268941$, so $U\approx0.268941$. Free energy is about $-0.313262$ and entropy about 0.582203.

Substitute $\log p_i=-E_i/T-\log Z$ into entropy to derive

$$
S=U/T+\log Z=(U-F)/T.
$$

This identity follows from the specified canonical probabilities. It is not a free identification of every information-theoretic entropy with heat divided by temperature.

## Degeneracy changes what must be counted

For $g>1$, let $H_{\rm binary}$ be the entropy of ground versus excited **energy occupancy**. All excited microstates have equal conditional probability $1/g$, giving

$$
S=H_{\rm binary}+P_1\log g.
$$

Thus the binary occupancy distribution loses information about which excited microstate is present. The lab displays both quantities, allowing an explicit check. Equal energies do not make distinct microstates disappear from the state count.

## Temperature limits

As $T\to0^+$, the unique ground state dominates and entropy tends to zero in this model. As $T\to\infty$, all $g+1$ microstates become equally likely, excited occupancy tends to $g/(g+1)$ and entropy to $\log(g+1)$.

For $g=1$, high-temperature excited occupancy tends to 1/2, not 1. Raising temperature does not mean every system is certainly excited. The reservoir exchanges energy; the system’s instantaneous energy fluctuates even though its equilibrium probabilities are stationary.

## Read the plotted curves

Blue is excited-level occupancy versus temperature; orange is dimensionless entropy versus temperature. They are different quantities with explicitly stated axes, not two estimates of the same probability. The table evaluates the selected temperature, whereas the curves show a temperature sweep for the current gap and degeneracy.

The curve grid covers positive temperatures only. Low-temperature floating-point saturation is a numerical representation issue, not a proof of a new physical limit. Moments and free energy are computed from the finite-state formula rather than estimated from drawn samples.

## AI connections and research limits

Energy-based models use normalized exponential scores and partition functions. The algebra resembles canonical probabilities, but physical thermodynamics requires actual state energies, reservoir conditions and units. Sampling a learned distribution is not automatically a physical thermal experiment.

A useful research extension adds interacting degrees of freedom, then compares exact enumeration on a small system with sampling. Check normalization, mean energy and autocorrelation. Do not claim a finite two-level example reproduces a thermodynamic phase transition or establishes universal laws of learning.

## Exercises and source

1. At high temperature with $g=3$, find excited occupancy. **Answer:** 3/4.
2. Show that $S-H_{\rm binary}=P_1\log g$ and verify it in the degenerate preset.
3. Explain why multiplying every unnormalized weight by one constant leaves probabilities unchanged but requires care when interpreting the zero of energy and free energy.
4. State which assumptions would change for an isolated system rather than a canonical reservoir.

For canonical probabilities and energy moments, see [MIT Statistical Physics: Micro-canonical and Canonical Ensembles](https://web.mit.edu/esfaki/www/8.08_files/canonical.pdf). The finite-level calculation above is this book’s specified teaching model.
