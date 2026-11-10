## A fixed target differs from a uniform guarantee

On ℓ2, Pₙ retains the first N coordinates. For each fixed x, its square-summable tail tends to zero, so Pₙx→x strongly. Operator norm instead optimizes over every unit input before taking a limit.

$$
\|(I-P_N)x\|_2^2=\sum_{k>N}|x_k|^2,\qquad \|I-P_N\|=1.
$$

Choose eₙ₊₁ for each N: it is removed entirely, giving error one. The witness changes with N, so this does not contradict convergence for each fixed vector. Interchanging supremum and limit needs additional assumptions.

## Compute one exact tail

For xₖ=1/k, the Basel sum gives squared norm π²/6 and

$$
\|x-P_Nx\|_2^2=\frac{\pi^2}{6}-\sum_{k=1}^{N}\frac1{k^2}.
$$

Integral comparison bounds this between 1/(N+1) and 1/N. Blue decreases for this fixed target while orange operator error stays one. At N=4, witness e5 has error one and e2 has zero. A successful selected witness does not determine the supremum.

## Compact approximation and unstable inversion

For compact T=diag(1/k), the norm tail decays:

$$
\|T-T_N\|=1/(N+1),\qquad \|T_N^{-1}\|=N.
$$

The inverse is taken on the first-N-coordinate space. Every finite truncation is invertible, but growing inverse norms amplify small high-index noise. The original operator has no bounded inverse onto all ℓ2. Resolution and inverse stability are different properties.

## Galerkin also requires stability

For a bounded coercive bilinear form on a real Hilbert space, with continuity constant M and coercivity α>0, conforming Galerkin satisfies Céa's bound:

$$
\|u-u_N\|\le\frac M\alpha\inf_{v_N\in V_N}\|u-v_N\|.
$$

The lab compares convergence modes; it does not implement that PDE method. Small discrete residuals alone do not establish continuous accuracy. Verify consistency, approximation density, boundary conditions and resolution-independent stability constants.

## Exercises with explanations

1. Error of e17 at N=16? **One**, because it is still omitted.
2. Does convergence for 1/k provide one N for every unit input? **No.** Later basis vectors remain counterexamples.
3. Does a small forward tail ensure a stable inverse? **No.** Here the tail shrinks while inverse norm grows as N.

Further reading: [Imperial Finite Element Course, Convergence](https://finite-element.github.io/L5_convergence.html).
