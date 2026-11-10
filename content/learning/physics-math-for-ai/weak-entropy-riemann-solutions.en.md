## A jump requires an integral interpretation

A shock has no ordinary spatial derivative at its discontinuity. Conservation laws therefore need a weak, distributional interpretation that remains meaningful for piecewise smooth fields. For Burgers' equation the conserved flux is f(u)=u²/2. The Riemann initial data are constant uL to the left of zero and uR to the right.

A candidate discontinuity moving at speed s satisfies the weak conservation law precisely when its delta-function coefficient vanishes:

$$
s[u]=[f(u)],\qquad
s=\frac{u_L+u_R}{2},\qquad [u]=u_R-u_L.
$$

The speed formula extends continuously to equal states, where there is no actual jump. A zero Rankine–Hugoniot defect checks conservation across the interface; it does not by itself select the physically admissible evolution.

## Compression and expansion have different entropy behavior

If uL>uR, faster left characteristics meet slower right characteristics, forming a compressive shock. Its speed lies strictly between the two states. If uL<uR, an expansion jump with the same algebraic speed still satisfies the weak law, but characteristics move away from it rather than into it.

For the convex quadratic entropy eta=u²/2 and entropy flux q=u³/3, the jump coefficient is

$$
[q]-s[\eta]=\frac{(u_R-u_L)^3}{12}.
$$

The entropy inequality requires a nonpositive coefficient. Compression satisfies it; expansion violates it. The lab's candidate-jump mode exposes this counterexample rather than silently treating every conservative discontinuity as an entropy solution.

## A rarefaction fills the expanding gap

For uL<uR the entropy solution is a continuous fan at positive time:

$$
u(x,t)=\begin{cases}
u_L&x\le u_Lt,\\
x/t&u_Lt<x<u_Rt,\\
u_R&x\ge u_Rt.
\end{cases}
$$

With states zero and one at t=1, the value at x=1/4 is one quarter. The candidate expansion jump instead returns zero at that point. Both conserve weak flux, but only the fan satisfies the entropy selection. At time zero the plot returns the stated step initial data.

## Open observation intervals need boundary flux

The readout integrates mass over [-2,2] and separately integrates flux through both endpoints in time. Their balance is M(t)-M(0)+Fout-Fin=0. A wave leaving this window can change its mass without violating conservation. Boundary events are integrated piecewise exactly for these shock and fan formulas, rather than approximated by an unrelated generic decay curve.

Jumps are drawn as separate horizontal segments with endpoint conventions. A connecting diagonal would depict a false smooth transition and imply nonexistent intermediate states at fixed time.

## Exercise: a conservative wrong answer

Compare uL=0, uR=1 in entropy and candidate modes. Both Rankine–Hugoniot defects vanish, while the candidate's entropy coefficient is positive 1/12. Next use states two and one at t=2: the shock has passed the observation window, whose mass changes from six to eight. Verify that accumulated inflow minus outflow equals two and explain why unchanged window mass is not the correct conservation test.

Further reading: [Clawpack Riemann-book Burgers chapter](https://www.clawpack.org/riemann_book/html/Burgers.html). The entropy and exact window-flux experiments are independently derived.
