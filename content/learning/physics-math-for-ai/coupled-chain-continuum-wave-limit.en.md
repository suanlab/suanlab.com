## From two masses to a field

A pair of coupled oscillators has two coordinates. A string appears to have a displacement at every position. The wave equation connects these descriptions: increase the number of moving elements while keeping the physical material and its initial shape fixed. The limit describes a field rather than a longer list of unrelated oscillators.

Our question is operational: when does a discrete mechanical model approximate the same continuum experiment? An animation that merely draws more points cannot answer it. We must specify masses, coupling, spatial wavelength, initial velocities and energy, then compare actual dynamics. This lesson uses a periodic chain with several independently initialized modes. It extends the earlier single-mode diffusion/wave comparison with a physical refinement experiment.

## Physical assumptions and SI units

Consider small transverse displacements $q_j(t)$ at equilibrium positions $x_j=ja$ on a uniform periodic string of length $L=Na$. The index wraps: $q_N=q_0$. A periodic computational domain models a repeating pattern; it is not a free or fixed endpoint.

Let line density $\mu$ have units kg/m and tension $\tau$ units N. Then

$$
m=\mu a,\qquad \kappa=\frac{\tau}{a},\qquad
m\ddot q_j=\kappa(q_{j+1}-2q_j+q_{j-1}).
$$

Here $q$ is in m, $m$ in kg and $\kappa$ in N/m. Constant tension and small transverse slopes justify linearizing the string force. The model excludes bending stiffness, damping, tension variation and geometric nonlinearities. The mathematical linear system still produces numbers at large slopes; those numbers need not describe a real stretched string.

## Derive the neighbor force

For a gently sloped segment, the transverse component of tension on the right is approximately $\tau(q_{j+1}-q_j)/a$. The left segment contributes $-\tau(q_j-q_{j-1})/a$. Their sum yields the second difference above.

A positive displacement at one particle while its neighbors remain still produces a negative force. This sign check matters: the reverse sign would produce an unstable anti-spring instead of oscillation. The longitudinal coordinate $x_j$ labels equilibrium position; the transverse displacement does not move particles along the horizontal plotting axis.

## Refine the material, not only the picture

Hold $L,\mu,\tau$ fixed and double $N$. The spacing halves, particle mass halves and coupling stiffness doubles:

$$
a\mapsto a/2,\quad m\mapsto m/2,\quad
\kappa\mapsto2\kappa,\quad
\frac{\kappa a^2}{m}=\frac{\tau}{\mu}=c^2.
$$

For $L=8$ m, $\mu=1$ kg/m and $\tau=1$ N, $N=16$ gives $a=0.5$ m, $m=0.5$ kg and $\kappa=2$ N/m. With $N=32$, these become $0.25$ m, $0.25$ kg and $4$ N/m. Both strings contain 8 kg and have continuum speed $c=1$ m/s.

Keeping $m$ and $\kappa$ fixed while reducing $a$ changes line density and tension. It is a different material sequence and cannot establish convergence for this experiment.

## Taylor expansion and its condition

For a smooth continuum displacement $u(x,t)$,

$$
u(x+a)-2u(x)+u(x-a)
=a^2u_{xx}+\frac{a^4}{12}u_{xxxx}+\cdots.
$$

Substitution and division by $\mu a$ give

$$
u_{tt}=c^2u_{xx}+\frac{c^2a^2}{12}u_{xxxx}+\cdots,
\qquad c=\sqrt{\tau/\mu}.
$$

The leading wave equation drops the higher spatial derivatives. For a component with spatial wavenumber $\xi$, the relevant dimensionless condition is $|\xi a|\ll1$. A small spacing measured in meters is insufficient if the initial shape oscillates on an equally small scale. Smoothness and a physical length scale are part of the approximation.

## Discrete and continuum spectra

Insert $q_j(t)=\exp(i\xi ja-i\omega_at)$ into the discrete equation. Using the two neighboring phases gives

$$
\omega_a^2=\frac{4c^2}{a^2}\sin^2\!\left(\frac{\xi a}{2}\right),
\qquad \omega=c|\xi|.
$$

For the nonnegative resolved modes used here,

$$
\frac{\omega_a}{\omega}
=\frac{\sin(\xi a/2)}{\xi a/2}
=1-\frac{(\xi a)^2}{24}+O((\xi a)^4).
$$

The lattice oscillates more slowly than the continuum component at the same physical wavenumber. Near the grid limit the difference is substantial. This does not make either exact solution numerically unstable; it means the lattice and continuum equations describe different dispersion. General wave packets and group velocity receive separate treatment later.

## Independent initial displacement and velocity

Use a finite real Fourier expansion with $\xi_n=2\pi n/L$:

$$
f(x)=\sum_n[A_n\cos(\xi_nx)+B_n\sin(\xi_nx)],
\qquad
g(x)=\sum_n[C_n\cos(\xi_nx)+D_n\sin(\xi_nx)].
$$

Position coefficients have units m and velocity coefficients m/s. A displaced string and a string receiving an initial kick need not have the same spatial profile. The chain receives samples $q_j(0)=f(x_j)$ and $\dot q_j(0)=g(x_j)$; the continuum receives those same physical functions.

Only distinct integer modes strictly below $N/2$ are admitted in this experiment. This avoids an unresolved mode being silently aliased and avoids the special Nyquist sine coordinate. The zero mode has only cosine coefficients, because $\sin0=0$.

## Exact time evolution, including the zero mode

For any nonzero angular frequency $w$, a mode coordinate with initial value $A$ and initial velocity $C$ is

$$
z(t)=A\cos(wt)+C\frac{\sin(wt)}w,
\qquad
\dot z(t)=-Aw\sin(wt)+C\cos(wt).
$$

Use $w=\omega_a$ for the chain and $w=\omega$ for the continuum. The implementation evaluates $\sin(wt)/w=t\operatorname{sinc}(wt)$, with $\operatorname{sinc}(0)=1$. At $w=0$ the result is $A+Ct$, not a singular calculation or a stationary point.

A nonzero constant initial velocity translates the whole string transversely. There is kinetic energy and momentum but no strain energy. Removing this motion to make a graph look centered would change the initial-value problem.

## What the dots and smooth curve mean

The chain has $N$ actual coordinates. A smooth trigonometric interpolant through them is a display aid, not additional moving mass. The continuum field exists between sample positions; evaluating it at the particle coordinates allows a matched comparison.

At time $t$, one useful measure is

$$
\varepsilon_N(t)=\left[\frac1N\sum_{j=0}^{N-1}
|q_j(t)-u(x_j,t)|^2\right]^{1/2}.
$$

Its units are m. A small error at one time can result from accidental phase alignment. Inspect several times and the frequency ratio as well. Increasing a drawing's pixel resolution does not change $N$ or the mechanical frequencies.

## Physical energy and periodic links

Each periodic link is counted once:

$$
E_a=\sum_j\left[\frac12m\dot q_j^2+
\frac12\kappa(q_{j+1}-q_j)^2\right].
$$

The continuum energy is

$$
E=\int_0^L\left[\frac12\mu u_t^2+
\frac12\tau u_x^2\right]dx.
$$

Both are conserved by their own exact dynamics. Their initial values need not be equal at finite spacing: a second difference and a derivative have different responses to the same Fourier mode. In the long-wave refinement limit their strain energies converge. Equal conservation errors alone do not prove equal physics.

For a nonzero resolved real mode, its chain energy is

$$
E_{a,n}=\frac{\mu L}{4}
[C_n^2+D_n^2+\omega_{a,n}^2(A_n^2+B_n^2)].
$$

The zero mode instead has energy $\mu LC_0^2/2$. Orthogonality makes these modal energies sum to the physical link-and-mass energy.

## Local energy and signed transfer

Assign half of each adjacent link's energy to a particle:

$$
E_j^*=\frac12m\dot q_j^2+
\frac14\kappa(q_j-q_{j-1})^2+
\frac14\kappa(q_{j+1}-q_j)^2.
$$

The flow through the link from $j$ to $j+1$, positive toward increasing index, is

$$
J_{j+1/2}=-\frac\kappa2(q_{j+1}-q_j)
(\dot q_{j+1}+\dot q_j),
\qquad \dot E_j^*=J_{j-1/2}-J_{j+1/2}.
$$

Differentiate the actual assigned energy using the force equation to check this identity. The flux has units W. A negative value is physical energy moving in the opposite direction; it is not a failed conservation check. Summing over the periodic chain cancels all internal flows.

## Worked refinement experiment

Take $L=8$, $\mu=\tau=1$ and a displacement component $0.05\cos(2\pi x/L)$ with zero initial velocity. For $N=16$, $\xi a=\pi/8$ and the frequency ratio is $\operatorname{sinc}(\pi/16)\approx0.99359$. At $N=32$ it is about $0.99839$.

The leading frequency deficit decreases by about four when spacing halves. With several fixed resolved modes and a fixed observation time, displacement error eventually shows the same second-order trend. This is an asymptotic statement, not a promise that every individual-time error decreases monotonically. Larger $t$ can accumulate larger phase differences. Keep initial functions, physical units and observation time fixed when making the comparison.

## Guided comparisons and model limits

First predict whether doubling $N$ should change $c$. Compare mass and stiffness before looking at trajectories. Next add a velocity mode absent from the initial displacement; check that the initial velocities are reconstructed rather than inferred from a phase chosen for convenience.

Then compare a low mode with a mode near the resolution limit. Both can be resolved lattice motions while only the former is a good continuum approximation. Finally add a constant velocity and inspect momentum and the affine mean motion. The supported calculator uses $N=8,16,32,64,128$, lengths4–16 m, positive density and tension0.25–4 in SI units, finite coefficients between−1 and1, and times0–64 s. These bounds describe the teaching experiment, not a universal restriction on wave theory.

## Verify with a different representation

An exact spectral implementation should not be checked only against another call to its own mode formula. Assemble the periodic second-difference matrix from neighbor forces, form the first-order position/velocity state matrix and evaluate its matrix exponential independently. A separately refined ODE solution provides a further numerical comparison. Compare actual coordinates, forces, local energies and link fluxes, not only a final energy sum.

For the continuum comparison, keep the same physical Fourier coefficients and change only the dispersion relation. Conservation validates each evolution separately, while refinement tests their relationship. In particular, the discrete and continuum initial strain energies differ at finite spacing. Replacing one with the other to obtain an identical displayed total would conceal the very modeling error the experiment measures.

## Exercises with explanations

### Exercise 1: Refining the same string

For $L=8$, $\mu=2$, $\tau=8$ and $N=16$, find $m,\kappa,c$, then double $N$.

**Solution:** $a=0.5$, $m=1$ kg, $\kappa=16$ N/m and $c=2$ m/s. Doubling $N$ gives $a=0.25$, $m=0.5$ and $\kappa=32$; speed stays2. The teaching calculator's tension bound excludes this particular numerical material, but the derivation remains valid.

### Exercise 2: A rigid transverse drift

If every particle starts at0.1 m with velocity0.05 m/s, what happens at12 s?

**Solution:** Every displacement is0.7 m. Differences and forces vanish. Energy is entirely kinetic, $\mu L(0.05)^2/2$; a zero frequency does not mean zero velocity.

### Exercise 3: A wavelength on four spacings

For $\lambda=4a$, compute $\omega_a/\omega$.

**Solution:** $\xi a=2\pi a/\lambda=\pi/2$. The ratio is $\sin(\pi/4)/(\pi/4)\approx0.90032$. Resolution alone is weaker than the long-wave assumption.

### Exercise 4: One link or two?

Why does the total energy use each link once although local assigned energies include two links?

**Solution:** Each particle receives half of each adjacent link. Summing over particles counts each physical link twice with weight one half. Counting two full links per particle would double the strain energy.

### Exercise 5: Small error at a late time

Does a very small $\varepsilon_N(t)$ at one late time prove a good continuum approximation?

**Solution:** No. Different frequencies can temporarily regain a similar phase. Compare multiple times, frequency ratios and a refinement sequence with unchanged physical initial data.

### Exercise 6: Initial kick without displacement

For one mode with $A=B=D=0$ and $C\ne0$, derive displacement and initial energy.

**Solution:** The cosine spatial component has amplitude $C\sin(\omega_at)/\omega_a$ and velocity amplitude $C\cos(\omega_at)$. Initial energy is $\mu LC^2/4$. For the zero mode the limiting displacement is $Ct$ and the normalization factor becomes one half.

## Connection to AI models and further reading

A learned simulator can fit a trajectory on one grid while encoding the wrong material scaling. Evaluate it on changed $N$ with fixed physical fields, and compare conservation, local transport and continuum error separately. Spectral neural operators also require a distinction between a physical wavenumber and an array index. These examples motivate verification questions; they do not establish generalization or a theorem for a trained network.

[MIT8.03SC lecture9](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/e4278fad9bdd90bc05f163fd7f670f65_MIT8_03SCF16_Lec9.pdf) derives a chain-to-wave limit. The multi-mode initial-value, SI energy, local flux and refinement calculation here follow directly from the stated mechanical model. The next lesson uses localized initial data and an actual endpoint, rather than a periodic repeating domain.
