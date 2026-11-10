## What changes when an order parameter jumps?

The preceding Curie–Weiss landscape develops two symmetry-related minima continuously as coupling passes its critical value. Other systems can acquire a distant competing minimum before their original minimum loses local stability. At equilibrium the preferred state then changes discontinuously when the two depths cross. Watching one locally stable solution would miss that crossing.

Think of three valleys: a central valley and two outer valleys. Lowering the outer valleys can make all three equally deep while each still has positive curvature. Equal depth is a global statement; curvature is a local one. A later disappearance of the central valley is another event. This chapter makes those events explicit in a scalar sixth-order Landau theory.

We study a specified phenomenological potential and its minimizers. We do not claim that every material has this polynomial, that a finite positive-temperature integral is singular, or that a static valley determines a switching time. The aim is to learn how symmetry, polynomial coefficients, multiple roots and global comparison work together.

## Specify the potential, its domain and its units

Take a dimensionless scalar order parameter m on the whole real line. Divide a physical free-energy density by a fixed positive scale f₀, so Φ and the displayed coefficients are dimensionless. Here b is the quartic coefficient and H is the conjugate field. This b is not the external field b of the previous chapter.

At H=0 the theory is invariant under m→−m. The even powers express that symmetry. The term −Hm breaks it. Positive c guarantees confinement: Φ grows to positive infinity at either end, so a global minimum exists. The calculator supports |a|,|b|,|H|≤4 and 0.25≤c≤4. Its computational coefficient bounds are not a physical bound |m|≤1.

For example, large negative quadratic or quartic coefficients can move a minimum outside the spin-magnetization interval. Clipping it to ±1 would solve a different constrained problem. A plot must include all certified roots and describe its viewing range without turning that range into the mathematical domain.

$$
\Phi(m)=\frac a2m^2+\frac b4m^4+\frac c6m^6-Hm,\qquad c>0,\qquad m\in\mathbb R,\qquad f(m)=f_0\Phi(m).
$$

## Stationarity requires a fifth-degree equation

Let g be the derivative of Φ. Every global minimum is an interior real root of this quintic. The curvature is its derivative. A positive curvature classifies a strict local minimum; a negative one classifies a local maximum. Zero curvature requires higher derivatives.

The quintic has at most five distinct real roots, but fewer distinct roots can retain algebraic multiplicity five. At a fold, two simple stationary roots become one double root. At a quartic minimum, g can have a triple zero. At a pure sextic minimum it has a quintuple zero. Counting plotted intersections without multiplicity loses the distinction.

Draw Φ and g together. A local maximum of Φ is a downward crossing of g, while a minimum is an upward crossing. A double zero of g touches zero without changing sign and produces a stationary inflection of Φ. Small derivative values near such a point do not by themselves certify a nearby minimum.

$$
g(m)=cm^5+bm^3+am-H,\qquad \Phi^{\prime\prime}(m)=g^{\prime}(m)=5cm^4+3bm^2+a.
$$

## At zero field reduce the problem to squared amplitude

At H=0, m=0 is always stationary. Every other stationary root has y=m²>0 solving the quadratic below. Each positive y gives two roots; negative y gives no real m. When y=0 it must not be counted again as two additional roots.

The exact discriminant decides whether these amplitudes exist. A repeated positive y at the discriminant zero produces two distinct double stationary roots at opposite magnetizations. The origin remains a separate root. If a=0, the origin instead acquires multiplicity three or five, depending on b.

At a nonzero stationary root eliminate a from the energy and curvature. These formulas are useful for identifying which amplitude is a minimum and whether its free energy is below the central value zero. They also avoid treating a minimum's mere existence as proof that it is globally preferred.

$$
cy^2+by+a=0,\qquad y=m^2>0,\qquad D=b^2-4ac.
$$

$$
\left.\Phi\right|_{g=0,H=0}=-\frac b4y^2-\frac c3y^3,\qquad \left.\Phi^{\prime\prime}\right|_{g=0,H=0}=2y(b+2cy).
$$

## A positive quartic term gives a continuous onset

With b>0 and H=0, a>0 gives the unique central minimum. When a becomes negative, the origin becomes a maximum and two outer global minima appear continuously. Close to zero the quartic and quadratic terms dominate, giving the amplitude scaling below.

The curvature at the central minimum is a, so its branch response is 1/a. The softening curvature anticipates a large response before the continuous critical point. At a=0 the positive quartic term still makes the origin a strict minimum, even though its quadratic curvature is zero.

The square-root exponent describes this scalar mean-field potential with the stated nonzero quartic coefficient. Spatial fluctuations, disorder or different symmetries can change the effective description. An interactive graph should connect the sign of a, the slope of g near zero and the appearance of both minima; it should not label a finite smooth rounding as a proof of criticality.

$$
|m_*|\sim\sqrt{-\frac ab}\quad(a\uparrow0,\ b>0,\ H=0),\qquad \left.\frac{dm}{dH}\right|_{m=0}=\frac1a\quad(a>0).
$$

## Removing the quartic term changes the leading balance

Set b=0 and keep c positive. For a<0, balancing the quadratic and sextic terms gives a fourth-root amplitude rather than a square root. At a=b=H=0 the origin is a strict sextic minimum. Its stationarity equation has multiplicity five.

Under a small field at that exact preparation, the response follows a fifth root. The local derivative diverges as the field vanishes. A numerical Hessian that reports zero cannot distinguish this point from a quartic critical minimum or a flat maximum; the leading nonzero derivative and its sign are necessary.

This is the mean-field tricritical preparation of this polynomial family. Realizing tricritical behavior requires tuning the appropriate parameters and checking the theory's validity. A small positive quartic coefficient eventually dominates sufficiently close to zero, so it cannot be silently replaced by exact zero just because its displayed value is tiny.

$$
|m_*|=\left(-\frac ac\right)^{1/4}\quad(a<0,b=H=0),\qquad m_*=\operatorname{sgn}(H)\left(\frac{|H|}{c}\right)^{1/5}\quad(a=b=0).
$$

## A negative quartic term permits first-order coexistence

Now let b<0. Positive c keeps the theory bounded below, while the negative quartic term can produce outer valleys even when the origin is still a local minimum. Equating the ordered stationary value to the central value zero gives the coexistence coefficient and nonzero amplitude below.

The factorization proves a global statement: at coexistence Φ is nonnegative for every real m and vanishes exactly at three points. All three are global minima. Their positive curvatures do not vanish when equilibrium preference switches. The order parameter jumps between zero and a finite amplitude when a crosses this value at zero field.

For a above coexistence, the central minimum wins wherever the outer minima still exist. Below coexistence, the ordered pair wins and the central minimum can remain metastable until a reaches zero. The word metastable describes local stability with a higher free energy; no physical residence time has been computed.

$$
a_* =\frac{3b^2}{16c},\qquad y_*=m_*^2=-\frac{3b}{4c},\qquad \Phi_{a_*,b,c,0}(m)=\frac c6m^2(m^2-y_*)^2\geq0\quad(b<0).
$$

## Spinodals mark loss of a local branch

At zero field, the ordered roots first appear at the discriminant-zero value a=b²/(4c). Their double stationary roots are folds. The central minimum loses its stability at a=0. The coexistence value lies strictly between these two boundaries for b<0.

| Decreasing a at H=0, b<0 | Local landscape | Equilibrium preference |
| --- | --- | --- |
| Above b²/(4c) | Central minimum only | Central |
| Between coexistence and b²/(4c) | Central and two outer minima | Central |
| At 3b²/(16c) | Three equally deep minima | Three coexist |
| Between 0 and coexistence | Three minima | Ordered pair |
| Below 0 | Central maximum and outer minima | Ordered pair |

At the ordered spinodal itself, the outer minimum and maximum have merged into stationary inflections. At a=0 the origin is a flat maximum when b<0. The open intervals in the table deliberately leave those boundary classifications separate. A plotted branch that persists past coexistence represents metastable continuation, not the equilibrium order parameter.

$$
a_{\mathrm{dis}}=0< a_*=\frac{3b^2}{16c}<a_{\mathrm{ord}}=\frac{b^2}{4c}\quad(b<0),\qquad m_{\mathrm{ord}}^2=-\frac b{2c}.
$$

## A field tilts the landscape and unfolds degeneracies

For H>0, a negative candidate cannot be globally minimal: reflecting it to positive m lowers its free energy. However, two different positive local minima can still compete. Field sign alone does not prove which positive minimum wins. The model therefore combines sign-based elimination with actual free-energy comparisons.

A nonzero-field fold solves g=g′=0. Parameterizing it by m gives the equations below. A higher degeneracy also requires g″=0. Such a point can occur at nonzero m and nonzero H, so a zero-curvature critical minimum need not sit at the origin.

Under H→−H all stationary roots reflect and the free-energy ordering reflects with them. This is an effective check on both a numerical solver and a visualization. A tiny tilt can select one state even when displayed free energies look equal; decimal equality is not a proof of coexistence.

$$
\Phi(-m)-\Phi(m)=2Hm,\qquad a_{\mathrm{sp}}(m)=-3bm^2-5cm^4,\qquad H_{\mathrm{sp}}(m)=-2bm^3-4cm^5.
$$

## Response and temperature need their own preparation

On a simple stationary branch, implicit differentiation gives dm/dH=1/Φ″. It is positive on minima and negative on maxima. At an odd multiple minimum the limiting derivative can be positive infinity; at a fold there is no single-valued smooth branch through the point. These cases require different labels.

The coefficient a is not automatically temperature. For a concrete thermal interpretation choose a(T)=α(T−T₀), α>0, with b,c and f₀ independent of T and H=0. Assume the coexistence temperature T* is positive; it is given below. If an arbitrary common background free-energy density is added, it changes absolute entropy but not the entropy jump between the competing phases.

With this explicit preparation the entropy contribution follows differentiation of the minimized physical free energy. The ordered phase has a lower entropy contribution than the disordered one; heating across coexistence absorbs the latent heat density below. If b,c,f₀ also vary with temperature, their derivatives must be included. These formulas should not be attached to an arbitrary coefficient slider without those assumptions.

$$
\left.\frac{dm}{dH}\right|_{\mathrm{branch}}=\frac1{\Phi^{\prime\prime}(m)}\quad\text{for a simple stationary root}.
$$

$$
T_*=T_0+\frac{3b^2}{16\alpha c},\qquad s_{\mathrm{dis}}-s_{\mathrm{ord}}=\frac{f_0\alpha}{2}m_*^2,\qquad L=T_*\frac{f_0\alpha}{2}m_*^2.
$$

## A scalar barrier is not a nucleation theory

For each local maximum, compare its potential with the adjacent minima. Their positive differences are barriers of the specified one-coordinate landscape. At coexistence a central minimum can have different escape directions, and the two outer minima do not share a single neighboring maximum. The calculator records both endpoint indices for each barrier.

Converting a free-energy density difference into a total free energy needs an explicit volume and a uniform-state assumption. Real spatial transitions may proceed through droplets and interfaces with lower barriers than moving the entire system uniformly. Their gradient energy, geometry and dynamics are missing here.

Likewise, following a metastable branch until a spinodal can sketch a possible quasistatic continuation. It does not predict an experimental hysteresis loop, a nucleation rate or a sampler's mixing time. Those require a dynamical model and an observation protocol.

$$
\Delta_{i\to j}=\Phi(m_j^{\mathrm{max}})-\Phi(m_i^{\mathrm{min}})>0,\qquad \Delta F=f_0V\,\Delta_{i\to j}\quad\text{for a uniform volume }V.
$$

## Certify roots and energy comparisons separately

The numerical model reads the actual binary64 coefficients as exact rational numbers. A rational polynomial gcd separates repeated-root structure. Sturm sign variations count real roots on intervals; exact bisection isolates every distinct real root. Multiplicity is retained rather than reconstructed by clustering nearby numeric answers.

For each isolated root the leading nonzero stationary derivative gives local classification. The free energy is enclosed using its exact midpoint value and a derivative bound times the bracket's half-width. Exact zero-field symmetry and the coexistence factorization establish ties algebraically. General overlapping energy bounds at the precision ceiling leave a set of global candidates and an unresolved status.

A complete root set and a resolved global selection are separate flags. A small residual cannot replace either proof. Nonzero values outside the numeric display range retain signed bounds; a zero-curvature fold gets no invented finite inverse curvature. Independent Decimal references use analytic curvature turning points and monotone intervals, rather than copying the Sturm algorithm.

$$
|\Phi(m_*)-\Phi(\bar m)|\leq\frac{u-\ell}{2}\sup_{m\in[\ell,u]}|g(m)|,\qquad \bar m=\frac{\ell+u}{2},\qquad m_*\in[\ell,u].
$$

## Connect coexistence and stability to the experiment

Begin with **a=3/16,b=-1, three coexisting minima**. In the free-energy view identify the three valleys; then switch to the derivative and identify all five stationary roots. The root table uses the same marker numbers. Orange-filled markers indicate certified global minima; an unfilled stationary marker alone does not establish local stability.

| Preparation to compare | Predict before changing it | Read in the result |
| --- | --- | --- |
| Ordered fold a=1/4 → coexistence a=3/16 | Does an outer root's appearance already make it globally preferred? | Distinguish double stationary inflections from five simple roots with three equal global minima. |
| Quartic critical → tricritical → flat maximum | Does zero curvature specify the local shape? | Compare multiplicities three, five and three, their roles and the leading nonzero stationary derivative. |
| Small field → coexistence coefficients with smallest field | Does a tiny tilt leave an exact tie? | Compare global roles and energy excesses; retain a positive splitting even when rounded energies match. |

Use **Root details** for curvature and energy excess, **Zoom root** for a local drawing, and **Full view** to restore the broad range. The order parameter lives on the real line; the plot range is not a constraint |m|≤1. Commit a,b,c,H with Enter or blur. Here b is the quartic coefficient and H is the field. The precision ceiling controls arithmetic, not temperature. The named unresolved example retains two global candidates at its cap; a complete root set does not imply a resolved global comparison. Static barrier rows identify an adjacent minimum and maximum, without specifying a switching time. Manual example steps remain available when motion is paused.

## Six worked checks

### 1. Prove the three-way coexistence

Set a=3/16,b=−1,c=1,H=0. Locate all minima and maxima and find the barriers.

**Explanation.** The potential factors as m²(m²−3/4)²/6. It is nonnegative everywhere, so zero and ±√3/2 are global minima of value zero. The other stationary roots are ±1/2, each a maximum of height 1/96. Their curvatures are −1/4; the central and outer minimum curvatures are 3/16 and 3/4. There are five distinct simple roots and four directed comparisons from an adjacent minimum to a maximum.

### 2. Separate three characteristic coefficients

Keep b=−1,c=1,H=0 and decrease a. Which coefficient indicates coexistence, which the ordered fold, and which the central stability boundary?

**Explanation.** They are 3/16,1/4 and 0, respectively. At a=1/4 the stationary roots are zero and ±1/√2. The outer roots have multiplicity two and are stationary inflections, while zero is the unique minimum. At a=3/16 all three minima coexist. At a=0 the origin has multiplicity three and is a flat maximum. A single curvature-zero rule cannot combine these events correctly.

### 3. Compare quartic and sextic criticality

At H=0 compare a=0,b=1,c=1 with a=b=0,c=1. Then give each a small positive field.

**Explanation.** Both origins are strict minima with zero curvature. In the first preparation g begins with m³ and the root scales as H^(1/3); in the second g=m⁵ and the root is H^(1/5). Their stationary multiplicities are three and five. At the smallest positive binary64 field the sextic root is around 2.18×10^−65, and its negative free energy remains nonzero. Higher-order information distinguishes the two preparations.

### 4. Construct a tilted double root

Choose a=−5/16,b=0,c=1,H=−1/8. Verify the stationary root m=1/2 and classify it.

**Explanation.** g(1/2)=1/32−5/32+1/8=0 and g′(1/2)=5/16−5/16=0. The second derivative of g is 20m³=5/2 there, so the root has multiplicity two. The potential has a stationary inflection. Changing H slightly above this value creates two nearby positive stationary roots; changing it below removes that pair. There is also a separate negative global minimum, so counting the double root as a stable minimum gives a wrong picture.

### 5. Find a higher-order minimum away from zero

Use a=15/16,b=−5/2,c=3,H=1/4. Test m=1/2.

**Explanation.** g,g′ and g″ vanish there, while g‴=30>0. In fact g=3(m−1/2)³(m²+3m/2+2/3), whose quadratic factor is positive everywhere because its discriminant is negative. Thus the potential decreases before 1/2 and increases after it. This is its unique global minimum, with stationary multiplicity three, zero curvature and energy −5/128. A nonzero field does not rule out a tuned higher-order minimum.

### 6. Interpret a tiny bias and a latent heat separately

At three-way coexistence give the system the smallest positive binary64 H. Then consider instead the explicit preparation a(T)=α(T−T₀) at H=0.

**Explanation.** The positive outer minimum is selected under the tiny field. The negative and central local minima survive with positive free-energy excesses; they do not become equal simply because a display rounds away their splitting. In the thermal preparation, the entropy jump requires α and f₀, and the latent heat density also requires T*. These physical quantities are not supplied by the field selection experiment. The two questions share a landscape but have different control protocols.

## Build an experiment that keeps these distinctions visible

A useful teaching view links a full potential curve, a derivative curve and a root table. Start with b=−1,c=1,H=0 and decrease a through 1/4,3/16 and zero. Highlight local minima and the selected global set separately. Add a zoom near a fold and retain the multiplicity label when two markers become visually indistinguishable.

A second experiment compares positive-quartic, zero-quartic and negative-quartic families. A third adds a signed field and examines root reflection and energy selection. Reduced-motion users should receive the same final states without needing to watch an animation. Exact boundaries and adjacent binary64 presets should use named selections; a coarse slider is insufficient for those comparisons.

The registered reader connects these preparations, a cancellable calculation worker and saved valid settings. The standalone theory and numerical checks alone do not establish correct cancellation, saved-state recovery or browser rendering; those require integration checks. [Tong's statistical field theory notes](https://www.damtp.cam.ac.uk/user/tong/sft/sfthtml/S1.html) provide broader background on Landau reasoning and its limitations. The exact sixth-order preparation, rational root certificates and worked checks here are specified independently.

For AI research, compare local optimization trajectories with global minima, inspect implicit gradients near singular curvature and distinguish an effective objective from a probabilistic model. Next topics on correlations, fluctuations and detailed balance will add the ensemble and dynamics information that this static scalar landscape does not contain.
