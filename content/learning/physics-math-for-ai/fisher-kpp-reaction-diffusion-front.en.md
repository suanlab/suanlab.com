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

## Research extension: upper and lower comparisons for spreading

The special wave above travels slightly faster than twice the diffusion-growth velocity scale. To understand why this is compatible with spreading theory, pose a **different initial-value problem**: a finite-mass positive Gaussian initial concentration with peak at most K, on the whole line. Let cNL be its bounded smooth nonlinear KPP solution, and let ell solve the linear growth equation with the same initial data. Comparison bounds cNL by K and ell. Indeed, the nonlinear reaction is no larger than linear growth; the difference cNL minus ell satisfies a linear differential inequality with zero initial value.

$$
\ell_t+U\ell_x=D\ell_{xx}+r\ell,\qquad rc(1-c/K)\le rc\ (c\ge0),\qquad0\le c_{\rm NL}(x,t)\le\min\{K,\ell(x,t)\}.
$$

The linear solution is the Gaussian transport field multiplied by exp(rt). It is an upper comparison, not the nonlinear concentration. Its peak eventually grows without saturation, whereas the nonlinear field stays below K. The initial peak condition in the display ensures the initial state lies between zero and K. The mass M belongs to that finite initial Gaussian; the linear mass grows as M exp(rt), rather than staying conserved.

$$
\ell(x,t)=\frac{Me^{rt}}{A\sqrt{2\pi v(t)}}\exp\!\left[-\frac{(x-\mu_0-Ut)^2}{2v(t)}\right],\quad v(t)=v_0+2Dt,\quad0<\frac{M}{A\sqrt{2\pi v_0}}\le K.
$$

Evaluate the logarithm along a laboratory ray of speed s-lab. The normalization contributes only logarithmic growth in time, while Gaussian spreading and reaction give the displayed exponential rate. The reference c-star is a fixed positive concentration making the logarithm dimensionless. Strictly outside the cone centered on U with half-speed twice sqrt(Dr), the upper comparison tends to zero, hence the nonlinear field does too. A positive exponent inside the cone says only that an upper bound grows; it does not yet prove that any nonlinear population reaches that location.

$$
x=\mu_0+s_{\rm lab}t:\quad\lim_{t\to\infty}\frac1t\log\frac{\ell(x,t)}{c_*}=r-\frac{(s_{\rm lab}-U)^2}{4D},\qquad |s_{\rm lab}-U|>2\sqrt{Dr}\ \Longrightarrow\ c_{\rm NL}(\mu_0+s_{\rm lab}t,t)\to0.
$$

An exponential leading-edge trial gives another view of the same speed scale. Insert it into the linear equation and cancel its positive amplitude. Minimizing D beta plus r over beta gives the critical decay exponent and velocity. This dispersion relation alone neither constructs a complete nonlinear wave nor selects the speed of arbitrary data. The two roots for a faster trial speed must not both be labeled the selected tail.

$$
\ell\propto e^{-\beta[x-(U+v)t]},\quad\beta>0:\qquad v(\beta)=D\beta+\frac r\beta,\quad\beta_* =\sqrt{r/D},\quad\min_{\beta>0}v(\beta)=v_*=2\sqrt{Dr}.
$$

The exact logistic-square wave has a leading exponential with twice its inverse-length parameter, since squaring the logistic factor doubles the tail exponent. Substitution into the dispersion relation recovers its special speed, with a shallower tail than the critical one. Its initial field has an infinite left plateau and an exponential right tail; it is not the finite Gaussian used for the upper comparison. Applying the Gaussian cone conclusion to that different initial field would create a false contradiction.

$$
\beta_{\rm AZ}=2\sqrt{\frac r{6D}}<\beta_*,\qquad v(\beta_{\rm AZ})=5\sqrt{\frac{Dr}{6}}>2\sqrt{Dr}.
$$

For a lower comparison, use a small compact bump moving at any **relative** speed s strictly inside the critical interval. Here s is relative to U, whereas s-lab in the earlier ray formula was a laboratory speed. Normalize concentration as u = c/K. Inside the bump a cosine enforces zero values at two edges and an exponential tilt compensates for drift in the moving coordinate. The zero extension is continuous and nonnegative but is not globally twice differentiable.

$$
y=x-\mu_0-(U+s)t,\quad |s|<v_*:\qquad V(y)=\begin{cases}\varepsilon e^{-sy/(2D)}\cos(\pi y/(2L)),& |y|<L,\\0,& |y|\ge L.\end{cases}
$$

Differentiate the interior bump directly. Its diffusion and relative-drift cost is a scalar eigenvalue consisting of a finite-width leakage term and a drift term. Choose the half-width large enough that this cost is below the low-density growth rate. The strict margin is what permits a small population to persist while moving; the linear upper bound by itself supplied no such lower information.

$$
-DV''-sV'=\Lambda_LV,\qquad\Lambda_L=D\left(\frac{\pi}{2L}\right)^2+\frac{s^2}{4D}<r\quad(|y|<L).
$$

The logistic residual is nonpositive if the bump height stays below the displayed margin. At the left edge its first derivative jumps upward from zero to a positive value; at the right it jumps upward from a negative value to zero. Thus its distributional second derivative has positive point masses and minus D times that derivative is nonpositive. The zero extension is a weak subsolution, not a classical C2 solution at the edges. The parabolic comparison principle applies to this lower comparison.

$$
-sV'-DV''-rV(1-V)=(\Lambda_L-r+rV)V\le0\quad\text{if }0\le V\le\frac{r-\Lambda_L}{r}.
$$

The positive Gaussian initial state has a positive minimum on every fixed compact interval. Choose epsilon small enough both to stay below that initial state and to satisfy the reaction margin. The exponential bound accounts for the largest tilt anywhere in the interval. Comparison then gives a strictly positive value along the bump-center ray for every nonnegative time. This proves nonvanishing on each strictly interior relative ray, separately from decay outside the cone. It does not by itself prove convergence to K, a measurable threshold near K, or the logarithmic front-position correction.

$$
m_0=\min_{|y|\le L}\frac{c_0(\mu_0+y)}K>0,\quad0<\varepsilon\le\frac12e^{-|s|L/(2D)}\min\!\left\{m_0,\frac{r-\Lambda_L}{r}\right\}\quad\Longrightarrow\quad\frac{c_{\rm NL}(\mu_0+(U+s)t,t)}K\ge\varepsilon>0.
$$

Rearranging the width condition shows its physical cost. As relative speed approaches the critical value, the required minimum width diverges and a safe amplitude can become extremely small. Near the limit, a mathematically positive lower bound may therefore be hard to observe experimentally or numerically. At or outside the critical speed no finite width satisfies this particular strict construction. That statement concerns this comparison method, not every possible traveling profile or an arithmetic clipping rule.

$$
\Lambda_L<r\quad\Longleftrightarrow\quad L>\frac{\pi D}{\sqrt{4Dr-s^2}}\quad(|s|<2\sqrt{Dr}),\qquad L_{\min}\longrightarrow+\infty\quad(|s|\uparrow2\sqrt{Dr}).
$$

For the table take D = 1 m²/s, r = 1/s and u0(x) = 0.2 exp[-x²/(2 m²)], centered at zero. Columns are relative speed in m/s, chosen half-width L in m, Lambda-L/r, positive epsilon, and the minimum initial u over that interval. The width is pi divided by sqrt(2[1-s²/4]) in these SI scales. The small amplitudes are shown in scientific notation to preserve their distinction from zero. The last figure panel shows V/epsilon and the initial minimum divided by epsilon on the support, not a nonlinear solution trajectory. No nonlinear trajectory is numerically solved in this supplement.

| relative speed | L | Lambda/r | epsilon | initial minimum |
| --- | --- | --- | --- | --- |
| 0.00000000 | 2.22144147 | 0.50000000 | 8.48049725e-03 | 1.69609945e-02 |
| 0.50000000 | 2.29429488 | 0.53125000 | 4.05399153e-03 | 1.43884077e-02 |
| 1.00000000 | 2.56509966 | 0.62500000 | 1.03329448e-03 | 7.45175245e-03 |
| 1.80000000 | 5.09633625 | 0.90500000 | 2.33417040e-09 | 4.58291912e-07 |

```python
from math import exp, pi, sqrt
# D=1 m^2/s, r=1/s, u0(x)=0.2 exp[-x^2/(2 m^2)].
for speed in (0.0, 0.5, 1.0, 1.8):
    gap = 1-speed*speed/4
    halfwidth = pi/sqrt(2*gap)
    eigenvalue = (pi/(2*halfwidth))**2+speed*speed/4
    minimum_initial = 0.2*exp(-halfwidth*halfwidth/2)
    epsilon = 0.5*exp(-speed*halfwidth/2)*min(minimum_initial, 1-eigenvalue)
    print(f"{speed:.8f} {halfwidth:.8f} {eigenvalue:.8f} {epsilon:.8e} {minimum_initial:.8e}")
```

The [Roquejoffre–Ryzhik KPP lecture notes](https://math.stanford.edu/~ryzhik/toulouse-lect.pdf) give comparison and positive-subsolution background. The constant-coefficient Gaussian upper comparison and compact moving bump here are derived explicitly. They are not a proof of the full periodic-media theorem in those notes. The current lab continues to display the original exact logistic-square family.

![Gaussian upper spreading cone and a separate compact moving lower comparison](/assets/learning/continuum-kpp-spreading-comparison-bump.en.svg)

### Research questions with explanations

**Does a growing linear upper bound prove nonlinear invasion?** **Answer:** No. It supplies no positive lower estimate. The compact moving bump gives a separate nonvanishing comparison for strictly interior relative speeds.

**Does the bump-center lower bound prove convergence to carrying concentration?** **Answer:** No. It guarantees only a positive epsilon on that ray. Saturation and precise front-location asymptotics require further arguments.

**Does the faster exact wave contradict the Gaussian cone bound?** **Answer:** No. The initial data are different: the exact wave has a left plateau and a shallower exponential right tail. The finite-Gaussian comparison cannot be transferred to it.

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
