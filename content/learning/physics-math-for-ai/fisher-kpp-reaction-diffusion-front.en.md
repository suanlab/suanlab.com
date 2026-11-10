## A growing front is more than a spreading cloud

A small population can spread into an unoccupied region while reproducing behind its leading edge. A concentration peak may then rise or a populated region may expand even after boundary transport has been accounted for. Diffusion alone cannot explain that increase: a source is needed. The Fisher–KPP equation combines diffusive spreading with growth that slows near a carrying concentration. A uniform background flow adds advection without changing the relative propagation mechanism.

We study one exact traveling profile. Its smooth formula makes it possible to inspect the field, differentiate it and integrate real finite observation windows. It does not replace a solver for arbitrary initial data. In particular, the speed of this special profile must not be presented as the universal speed selected by every initial condition. Separating the profile's assumptions from broader front theory is part of understanding what an analytic example can establish.

## Assumptions, units and the open-system interpretation

The spatial domain is the entire line. Diffusivity D and growth rate r are strictly positive constants. Carrying concentration K and tube area A are positive. The prescribed background velocity U is uniform. Concentration is nonnegative, and the exact profile remains between zero and K. This lesson does not model the resource, substrate or energy supply that permits growth. The source represents exchange or conversion involving variables outside the model; it does not assert creation of matter in a closed system.

| Quantity | Unit | Meaning |
| --- | --- | --- |
| c, K | kg/m³ | modeled material concentration and carrying concentration |
| D | m²/s | diffusivity |
| r | 1/s | low-concentration growth rate |
| U | m/s | prescribed bulk velocity |
| j | kg/(m² s) | signed total flux |
| S | kg/(m³ s) | growth source |

$$
c_t+Uc_x=Dc_{xx}+rc\left(1-\frac{c}{K}\right),\qquad j=Uc-Dc_x,\qquad c_t+j_x=S,\quad S=rc\left(1-\frac{c}{K}\right).
$$

The source is positive between zero and K, and zero at either endpoint. It is largest at half the carrying concentration. Local growth alone does not determine the motion of the entire front; spatial transport and the surrounding profile also matter.

## Scale the equation before interpreting the wave

Remove uniform advection by moving with the bulk flow. Measure distance in diffusion-growth lengths and time in growth times. Concentration divided by K is dimensionless. The transformed equation has no physical coefficients, making it easier to recognize the special traveling solution and restore its units afterward.

$$
u=\frac{c}{K},\qquad \xi=\sqrt{\frac rD}(x-Ut),\qquad \tau=rt,\qquad u_\tau=u_{\xi\xi}+u(1-u).
$$

The length scale is the square root of D divided by r, not D divided by r itself. The corresponding velocity scale is the square root of D times r. A dimensional check is a useful safeguard against inserting a dimensionless wave speed directly into a plot labeled meters per second.

## Construct the exact logistic-square profile

Let x-zero locate a convenient reference point at time zero. The profile moves to the right relative to the bulk flow. Define its inverse length lambda, its relative speed v and its laboratory speed as follows.

$$
\lambda=\sqrt{\frac r{6D}},\qquad v=5\sqrt{\frac{Dr}{6}},\qquad X(t)=x_0+(U+v)t,\qquad z=\lambda(x-X(t)),\qquad c=\frac{K}{(1+e^z)^2}.
$$

Far to the left, concentration approaches K. Far to the right, it approaches zero. The profile has no finite sharp edge. Changing K scales concentrations but leaves lambda and v unchanged. Increasing D broadens the front and increases its relative speed; increasing r narrows it and also increases its relative speed. These effects follow from this family, with the other parameters held fixed.

## Verify the spatial derivatives explicitly

Introduce complementary logistic factors w and s. Keeping both factors will also help with numerical stability near a plateau. Differentiate the actual concentration rather than defining a derivative indirectly from the PDE one hopes to verify.

$$
w=\frac1{1+e^z},\qquad s=1-w,\qquad c=Kw^2,\qquad c_x=-2\lambda cs,\qquad c_{xx}=2\lambda^2cs(3s-1).
$$

The first derivative is negative throughout the finite transition. The second derivative changes sign, so diffusion can locally increase or decrease concentration. A positive D does not imply that the term D times c_xx is positive everywhere. This distinction helps explain why a local reaction rate cannot by itself predict the local time derivative.

## Verify the time derivative and the equation

The profile translates without changing shape. Its time derivative is minus laboratory speed times its spatial derivative. Rewriting the source through s makes the substitution transparent because one minus w squared equals s times one plus w.

$$
c_t=-(U+v)c_x,\qquad S=rcs(2-s),\qquad -vc_x=Dc_{xx}+S.
$$

For positive c and s, divide the last equality by c times s. Substitution of the proposed lambda and v then verifies the identity for every location. This direct check establishes the exact family; an independently differentiated reference is still needed to check that a software implementation evaluates the family correctly.

$$
2v\lambda=2D\lambda^2(3s-1)+r(2-s)=\frac{5r}{3}.
$$

The advection term disappears only from the relative-motion identity. It remains present in laboratory motion and in the total boundary flux. A stationary laboratory profile can therefore have active diffusion, growth and advection in balance.

## The reference point is the quarter level

At z equal to zero, the denominator is four. Consequently X is a quarter-concentration reference, not the half-height position. Any fixed relative level between zero and one has its own position, and all these positions move with the same laboratory speed.

$$
c(X(t),t)=\frac K4,\qquad x_\alpha(t)=X(t)+\frac1\lambda\log\left(\alpha^{-1/2}-1\right),\qquad \frac{dx_\alpha}{dt}=U+v.
$$

For the half level, the logarithmic shift is negative, placing it to the left of X. A visualization can show quarter, half and another level simultaneously to make the difference visible. An absolute concentration threshold behaves differently if K is changed; specify whether a tracked threshold is absolute or relative before comparing experiments.

## A worked local balance

Take D equal to one sixth square meter per second, r equal to one per second, K equal to four kilograms per cubic meter and U equal to zero. At the reference point, lambda is one inverse meter and concentration is one kilogram per cubic meter. Substitution gives a complete local balance.

$$
\lambda=1,\quad v=\frac56,\quad c=1,\quad c_x=-1,\quad c_{xx}=\frac12,\quad c_t=\frac56,\quad Dc_{xx}=\frac1{12},\quad S=\frac34.
$$

Thus diffusion contributes one twelfth and growth contributes three quarters in concentration-per-time units. Their sum is five sixths, matching the actual time derivative. These values are local, not integrated masses. Multiplying the source at this one point by an arbitrary window width is generally not the exact source integral, because the profile changes across the window.

## Finite-window storage includes integrated growth

For a fixed physical window, integrate the conservation form. The right-minus-left boundary flux is signed net outflow. The source must be integrated over the same window and multiplied by the same physical area as the storage.

$$
M_{[a,b]}=A\int_a^bc\,dx,\qquad \frac{dM_{[a,b]}}{dt}+A[j(b)-j(a)]=A\int_a^bS\,dx.
$$

A rise in stored mass can come from inward transport, positive growth, or both. In the computational model, the concentration, actual time derivative and growth source are integrated separately; boundary fluxes are evaluated at the endpoints. Quadrature agreement is a diagnostic, not a certified error bound for arbitrary functions. The accompanying independent references differentiate and integrate the analytic field at high precision without calling the product model.

## Why total mass on the entire line diverges

The left plateau extends indefinitely and approaches positive K. The entire-line mass is therefore infinite. This is a different setup from the finite-mass Gaussian cloud, even though both examples use diffusion. A finite display interval cannot turn this analytic front into a finite-mass whole-line state.

$$
\lim_{x\to-\infty}c(x,t)=K>0,\qquad A\int_{-\infty}^{\infty}c(x,t)\,dx=+\infty.
$$

Report the global mass as divergent rather than returning a finite integral over the plotted window with a global-mass label. Finite-window budgets remain well defined and useful. A model for a finite resource reservoir or an initially localized population would need different initial data, boundaries or additional equations.

## Plateau deficits and front tails need stable observations

On the left, concentration may round to K while the true deficit and growth remain positive. Computing one minus rounded c over K then produces a false zero source. On the right, the concentration can underflow even though a scaled source is still representable. Store logarithmic concentration and evaluate complementary logistic factors through stable softplus formulas.

$$
\operatorname{softplus}(z)=\log(1+e^z),\quad \log w=-\operatorname{softplus}(z),\quad \log s=-\operatorname{softplus}(-z),\quad K-c=Ks(1+w),\quad S=rcs(1+w).
$$

The written definition of softplus needs a stable implementation: direct exponentiation at large positive z can overflow. Evaluate its equivalent branches using z plus a small logarithmic correction for positive z. Preserve the stable deficit separately from subtraction of rounded concentrations. If a nonzero linear observation is below the representable range, show it as unavailable with its logarithmic value rather than turning it into an exact zero. Display precision should communicate what the calculation can resolve.

## What the controls can and cannot prove

Compare the same actual field with and without the source in the diagnostic equation. The source-omission residual should equal the positive growth source. This tests the need for that term. Setting r to zero in the exact-profile parameters is a different operation: the length and speed parameterization degenerates and the displayed family is no longer the same test. D equal to zero is likewise outside this profile's domain.

Vary U through the negative of the relative speed. The laboratory front stops, then moves left if U becomes more negative, although relative propagation remains rightward. Track several relative levels and compare their common speed. These experiments develop intuition for reference frames without claiming that the exact special speed is the minimal front speed or the selected speed of arbitrary initial data. A research project studying speed selection must specify initial tails and an appropriate solution problem separately.

## Exercises with worked answers

### Exercise 1 — Locate the half level

Is the concentration at X equal to K divided by two? Where is the half level?

**Answer:** The concentration at X is K divided by four. The half level is X plus log of square root of two minus one, divided by lambda. This shift is negative, so the half level lies to the left.

### Exercise 2 — Check the worked example

In the worked case, what would the local time derivative be if only growth were counted?

**Answer:** Three quarters, which misses the diffusion contribution of one twelfth. The actual derivative is five sixths. Even an exact reaction law alone does not account for spatial redistribution.

### Exercise 3 — Stop the laboratory front

What U makes the laboratory front stationary, and does reaction stop then?

**Answer:** Choose U equal to minus v. The time derivative vanishes because the whole profile is stationary in that frame, but reaction and diffusive transport remain active and are balanced by advection.

### Exercise 4 — Interpret the global integral

Can a finite plotted interval provide the entire-line mass of this front?

**Answer:** No. The positive left plateau gives divergent entire-line mass. The finite integral is the mass inside the specified window and must be labeled accordingly.

### Exercise 5 — Diagnose a false zero

Concentration rounds to K but a stable calculation reports positive growth. Which observation should be discarded?

**Answer:** Neither needs to be discarded. They describe different resolved quantities. Rounded concentration has lost the small deficit, while stable complementary factors still resolve it. Show the distinction rather than forcing the source to zero.

### Exercise 6 — Interpret the special speed

Does verifying this analytic wave prove that every localized initial state travels at v?

**Answer:** No. It proves that this particular initial profile evolves by translation with that relative speed. Speed selection for other initial states is a separate problem and depends on additional assumptions.

## Sources and scope

[Ablowitz and Zeppetella, Explicit solutions of Fisher's equation for a special wave speed (1979)](https://link.springer.com/article/10.1007/BF02462380) identifies an explicit solution at a special dimensionless speed. The source review used the publisher's abstract, not the subscription full text. The dimensional profile and substitutions in this lesson are derived directly above and checked separately; they are not a claim to have reviewed an inaccessible full-paper proof.
