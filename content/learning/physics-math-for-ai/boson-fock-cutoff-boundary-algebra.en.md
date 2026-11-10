## 1. Multiple particles in one mode make the space infinite

A fermion mode in the previous chapter had occupancy0 or1. Bosons can occupy the same one-particle mode repeatedly. Photon number in an optical mode and excitation number in an oscillator are familiar examples. Even one mode needs all occupations0,1,2,… . Having one mode does not imply a two-dimensional state space.

Assume that the selected one-particle modes are orthonormal. Write their number states as |n〉and combine multiple modes with an ordinary tensor product. Operators at different boson modes commute; we do not import the fermionic parity string. Similar occupation diagrams can conceal different statistics and algebras.

Our route is **infinite ladder → finite projection → boundary commutator → coherent series → truncation error and observables**. Operator identities are used on finite Fock superpositions or states with the required moments. The infinite-dimensional a and a† are unbounded operators, so we do not assume that every product is defined on every Hilbert-space vector. Numerical examples use one or two modes with maximum occupancy3; this is not a limit on the theory or curriculum.

## 2. Square-root factors come from normalization

The one-mode vacuum satisfies a|0〉=0 and has unit norm. The canonical commutation relation, CCR, is [a,a†]=I. Unlike the fermionic anticommutator, it takes the difference of two products. Normalizing the eigenstates of the number operator n̂=a†a gives these ladder factors.

$$
|n\rangle=\frac{(a^\dagger)^n}{\sqrt{n!}}|0\rangle,\qquad
a|n\rangle=\sqrt n\,|n-1\rangle,\qquad
a^\dagger|n\rangle=\sqrt{n+1}\,|n+1\rangle.
$$

The squared norm of a|n〉is〈n|a†a|n〉=n. If the lower state has unit norm, the coefficient must therefore be sqrt(n). Using aa†=n̂+I gives the creation factor sqrt(n+1). Two creations do not vanish: a†a†|0〉=sqrt(2)|2〉. A matrix that merely changes an occupancy bit would not reproduce these normalizations or observables.

Including every n makes the operator matrix infinite. To use a finite array, we must choose a retained subspace. Removing an array's final direction is also a physical approximation choice. We will calculate the resulting boundary term instead of treating it as an invisible implementation detail.

## 3. Cutoff means maximum occupancy, with dimension one larger

Let C be the largest retained occupation and project onto0 through C. Thus C=3 gives dimension4, while C=0 still gives the one-dimensional vacuum space. Confusing cutoff with dimension puts the last ladder row and column in the wrong place.

$$
\Pi_C=\sum_{n=0}^{C}|n\rangle\langle n|,\qquad
a_C=\Pi_Ca\Pi_C,\qquad a_C^\dagger=\Pi_Ca^\dagger\Pi_C.
$$

Creating at |C〉would give sqrt(C+1)|C+1〉in the infinite space. Projection discards that direction, so a_C†|C〉=0. This is the boundary of a selected representation, not a new law forbidding a boson from exceeding occupation C. Its zero result has a different origin from fermionic exclusion.

| Input at C=3 | Output of a_C | Output of a_C† | Eigenvalue of [a_C,a_C†] |
| --- | --- | --- | --- |
| \|0〉 | 0 | \|1〉 | 1 |
| \|1〉 | \|0〉 | √2\|2〉 | 1 |
| \|2〉 | √2\|1〉 | √3\|3〉 | 1 |
| \|3〉 | √3\|2〉 | 0 | −3 |

Extrapolating the earlier rows onto the final row gives the wrong commutator. The create-then-annihilate path disappears at the boundary. Follow operator order and identify which intermediate path leaves the retained space.

## 4. Finite matrices cannot exactly preserve the infinite CCR

Acting on the basis gives diagonal entries0,1,…,C for a_C†a_C and1,2,…,C,0 for a_Ca_C†. Subtracting leaves a boundary projector.

$$
[a_C,a_C^\dagger]=I_C-(C+1)|C\rangle\langle C|,\qquad
\langle[a_C,a_C^\dagger]\rangle=1-(C+1)p_{\rm top},\qquad
p_{\rm top}=\langle C|\rho|C\rangle.
$$

This discrepancy is not floating-point error. For any finite matrices A and B, Tr(AB−BA)=0, whereas the identity has trace equal to a positive dimension. Hence [A,B]=I is impossible in a finite-dimensional space. Raising the boundary can improve an approximation, but it cannot remove this algebraic obstruction.

At C=3, the top state has commutator expectation−3. The state (|0〉+|3〉)/sqrt(2) has top probability1/2 and expectation−1. At C=0, a_C=0 and I_C=|0〉〈0|, so the commutator vanishes. Adjusting a reported value to1 would misrepresent the operators actually used.

## 5. Insert the actual commutator into uncertainty bounds

Define dimensionless quadratures X=(a+a†)/sqrt(2) and P=(a−a†)/(i sqrt(2)). The original infinite mode has [X,P]=iI. For an oscillator of mass m and frequency Omega, physical units are restored by x=sqrt(hbar/(m Omega))X and p=sqrt(m hbar Omega)P.

$$
X_C=\frac{a_C+a_C^\dagger}{\sqrt2},\qquad
P_C=\frac{a_C-a_C^\dagger}{i\sqrt2},\qquad
[X_C,P_C]=i\bigl(I_C-(C+1)|C\rangle\langle C|\bigr).
$$

Variance concerns a particular state and particular operators. With K the symmetrized covariance, the Robertson–Schrödinger inequality takes the following form. Its right-hand side uses the **expectation** of the finite commutator.

$$
K=\frac12\langle X_CP_C+P_CX_C\rangle-\langle X_C\rangle\langle P_C\rangle,\qquad
\operatorname{Var}(X_C)\operatorname{Var}(P_C)-K^2
\ge\frac14\left|1-(C+1)p_{\rm top}\right|^2.
$$

At C=0, both operators vanish, giving zero variances and a zero bound. This does not create a state violating the uncertainty principle for the original position and momentum; the operators changed. For the C=3 top state, each variance is3/2, K=0 and the bound is9/4, saturating the finite inequality. Do not impose the infinite-mode constant1/4 on every finite input.

## 6. Projecting a product differs from multiplying projected operators

Embed a finite state into the infinite space by extending its coefficients with zeros, and consider measuring the **original X**. Its mean equals that of X_C, but its second moment differs. A path temporarily reaching |C+1〉and returning exists in X² and is absent from X_C².

$$
\Pi_CX^2\Pi_C-X_C^2
=\Pi_CP^2\Pi_C-P_C^2
=\frac{C+1}{2}|C\rangle\langle C|.
$$

For that embedded state, original quadrature variances therefore equal finite variances plus(C+1)p_top/2. The finite vacuum at C=0 has zero finite variances, yet original X and P each give1/2. The C=3 top state changes from finite variance3/2 to original variance7/2. State explicitly whether the comparison changes a state, an operator or both.

The same distinction affects an oscillator Hamiltonian. Projecting the original H differs at the boundary from squaring previously truncated quadratures.

$$
\frac{H_C}{\hbar\Omega}=\hat n_C+\frac12I_C,\qquad
\frac{X_C^2+P_C^2}{2}
=\hat n_C+\frac12I_C-\frac{C+1}{2}|C\rangle\langle C|.
$$

For C=3, the top energy is7/2 under the first definition and3/2 under the second. The error comes from transferring an identity derived with the infinite CCR into finite matrices. To represent an oscillator with onsite and offset terms, specify the projected model H_C=hbar Omega(n̂_C+I_C/2). The operator definition precedes any smooth trajectory in a picture.

## 7. Two-mode hopping has bosonic enhancement

The two-mode dimension is(C₀+1)(C₁+1), with index n₀(C₁+1)+n₁. Use an ordinary tensor product, without a fermionic occupied-permutation sign when exchanging modes. The square-root creation and annihilation factors both enter a hopping amplitude.

$$
a_0^\dagger a_1|n_0,n_1\rangle
=\sqrt{(n_0+1)n_1}\,|n_0+1,n_1-1\rangle
\quad\text{when the target lies within the cutoffs}.
$$

Apply H=a₀†a₁+a₁†a₀ to |2,2〉. With both cutoffs3, two paths survive. Lowering the second cutoff to2 removes only the |1,3〉direction. The mean energy stays zero, but variance detects the change.

| Maximum occupations(C₀,C₁) | Nonzero Hamiltonian output | Mean E | Var(H) |
| --- | --- | --- | --- |
| (3,3) | √6\|3,1〉+√6\|1,3〉 | 0 | 12 |
| (3,2) | √6\|3,1〉 | 0 | 6 |

The finite experiment includes onsite epsilon_i n_i, self-density U_i n_i(n_i−1)/2, cross-density Vn₀n₁, complex hopping and drives f_i a_i†+f_i* a_i. Coefficients carry energy units. Without drives this model conserves total number, while drives can break that conservation. A vacuum having sharp N=0 at an instant still does not establish an operator conservation law.

## 8. A coherent state is an annihilator eigenstate

Solve a|alpha〉=alpha|alpha〉in the infinite space with normalization. Alpha is a dimensionless complex number. Writing number-state coefficients d_n gives a recurrence through the ladder action. Fix the arbitrary global phase by taking d₀ positive.

$$
\sqrt{n+1}\,d_{n+1}=\alpha d_n,\qquad
|\alpha\rangle=e^{-\mu/2}\sum_{n=0}^\infty\frac{\alpha^n}{\sqrt{n!}}|n\rangle,\qquad \mu=|\alpha|^2.
$$

The coefficients are d_n=d₀ alphaⁿ/sqrt(n!). Their squared norm sums to |d₀|² exp(mu), giving d₀=exp(−mu/2). Alpha=0 is the vacuum. Nonzero alpha generally needs infinitely many occupation directions. Its complex phase sets the direction of the X and P means.

A coherent state superposes different particle numbers. We do not automatically impose the preceding fermionic parity assumption on a boson model. Preparation, observation and a phase reference belong to the specified physical problem. Here we define and compare the stated infinite series and its operator means.

## 9. Derive Poisson statistics and minimum uncertainty

Number-measurement probabilities are squared amplitude magnitudes. Shifting the series index once or twice gives the moments of n and n(n−1), from which number variance follows.

$$
p_n=e^{-\mu}\frac{\mu^n}{n!},\qquad
\langle\hat n\rangle=\mu,\quad
\langle\hat n(\hat n-1)\rangle=\mu^2,\quad
\operatorname{Var}(\hat n)=\mu.
$$

Likewise, substitute〈a〉=alpha,〈a²〉=alpha²,〈a†a〉=mu and〈aa†〉=mu+1 into the quadrature expressions. The center moves without changing vacuum fluctuations.

$$
\langle X\rangle=\sqrt2\operatorname{Re}\alpha,\qquad
\langle P\rangle=\sqrt2\operatorname{Im}\alpha,\qquad
\operatorname{Var}(X)=\operatorname{Var}(P)=\frac12,\qquad K=0.
$$

Being an annihilator eigenstate does not mean being an energy or number eigenstate. For H=hbar Omega(n̂+1/2) and nonzero alpha, energy variance is(hbar Omega)²mu. Distinguish an eigenstate condition for the non-Hermitian operator a from a residual test for a Hermitian H.

## 10. Separate the infinite state, raw projection and renormalized state

Call the retained coefficients without their exponential prefactor z_C, and their squared norm Z_C. This Z_C is a finite-series norm, rather than an assumed thermodynamic partition function. Reusing a symbol does not transfer its physical interpretation.

$$
z_C=\sum_{n=0}^{C}\frac{\alpha^n}{\sqrt{n!}}|n\rangle,\qquad
Z_C(\mu)=\sum_{n=0}^{C}\frac{\mu^n}{n!},\qquad
\Pi_C|\alpha\rangle=e^{-\mu/2}z_C,\qquad
|\psi_C\rangle=\frac{z_C}{\sqrt{Z_C}}.
$$

| Object | Squared norm | Meaning |
| --- | --- | --- |
| Infinite \|α〉 | 1 | Original coherent state |
| Unnormalized Π_C\|α〉 | S=e^(−μ) Z_C | Retained component |
| Renormalized ψ_C | 1 | Conditional finite state |

Renormalizing does not make discarded probability zero. If an ideal projector measurement{Π_C,I−Π_C} were performed and its retained outcome selected, S would be its probability and psi_C the conditional state. We can also calculate these vectors as an approximation comparison without asserting that a measurement took place.

In the infinite space, the displacement D(alpha)=exp(alpha a†−alpha* a) takes the vacuum to a coherent state. Computing a finite D_C=exp(alpha a_C†−alpha* a_C) is a different operation from projecting the infinite series. At C=1 and alpha=1 it gives cos(1)|0〉+sin(1)|1〉, whereas projection and renormalization give(|0〉+|1〉)/sqrt(2). The changed finite CCR prevents blindly transferring the original normal-ordering identity.

## 11. Tail probability and distances have different formulas

Let S be retained probability and epsilon the discarded Poisson weight. Projection splits the original vector into orthogonal components, and the renormalized state's overlap with it is positive sqrt(S). Three error quantities can then be distinguished.

$$
S=e^{-\mu}Z_C,\qquad \varepsilon=1-S,\qquad
\bigl\||\alpha\rangle-\Pi_C|\alpha\rangle\bigr\|^2=\varepsilon,\qquad
|\langle\alpha|\psi_C\rangle|^2=S,\qquad
\bigl\||\alpha\rangle-|\psi_C\rangle\bigr\|^2=2(1-\sqrt S).
$$

The final two states are normalized, and their fidelity **squared** is S. Do not label the squared distance2(1−sqrt(S)) as the discarded probability epsilon. They approach one another for small epsilon but have different exact definitions. Conditional probabilities inside the cutoff are muⁿ/(n!Z_C), different again from the original exp(−mu)muⁿ/n!.

The tail can be written using a lower incomplete gamma function. Here gamma is a special function, not a one-body matrix. Reversing lower and upper conventions changes the answer.

$$
\Pr(n>C)=\frac{\gamma(C+1,\mu)}{C!}
=1-e^{-\mu}Z_C(\mu),\qquad
\gamma(s,x)=\int_0^x t^{s-1}e^{-t}\,dt.
$$

Two independent coherent modes with rectangular cutoffs have S=S₀S₁ and total tail1−S₀S₁. Adding the separate tails double-counts the event in which both modes fall outside. At mu=10⁻¹² and C=3, the tail is approximately mu⁴/24=4.17×10⁻⁵⁰ and remains positive. A conventional decimal calculation of1−CDF may appear zero without establishing that the probability is absent. At mu=16,C=3, retained probability is only about9.31×10⁻⁵, so this small finite example is a poor approximation to the original state.

## 12. A finite annihilator has no identical nonzero coherent eigenstate

Applying finite lowering C+1 times gives zero. Thus a_C is nilpotent and all its eigenvalues are zero. An exact eigenstate at nonzero alpha would contradict a_C^(C+1)psi=alpha^(C+1)psi. A small residual must not be confused with an exact eigenstate.

Below the boundary, truncated coherent coefficients satisfy the original recurrence. Only the final coefficient lacks a following direction and leaves a residual. Let p_top now denote the top probability in the renormalized state.

$$
p_{\rm top}=\frac{\mu^C}{C!Z_C},\qquad
\|(a_C-\alpha)|\psi_C\rangle\|^2=\mu p_{\rm top},\qquad
\langle a_C\rangle=\alpha(1-p_{\rm top}),\qquad
\langle\hat n_C\rangle=\mu(1-p_{\rm top}).
$$

At C=0, psi_C is the vacuum and p_top=1. Finite annihilator and number means vanish, while the eigenstate residual squared is mu. At alpha=0 the vacuum matches the original state at every cutoff, with zero tail and residual. Separating these cases shows that renormalization does not automatically recover physical information.

## 13. Check three comparisons by hand at alpha=1,C=3

Exact coefficients are[1,1,sqrt(2)/2,sqrt(6)/6], with Z₃=8/3. Conditional top probability is1/16, mean number and mean a_C are15/16, and actual commutator expectation is3/4. The annihilator eigenstate residual squared is1/16. These are calculated finite values, rather than copied infinite coherent-state means.

$$
\langle\hat n_C^2\rangle=\frac{27}{16},\qquad
\operatorname{Var}(\hat n_C)=\frac{207}{256},\qquad
\operatorname{Var}(X_C)=\frac{39}{128},\qquad
\operatorname{Var}(P_C)=\frac9{16},\quad K_C=0.
$$

Embedding this state into the infinite space and observing original X/P adds the boundary correction1/8 to both variances. That is another comparison, distinct from returning to the original coherent state. Read the **state and operators together** in the table.

| State and observables | Mean number | Var(X) | Var(P) | 〈[X,P]〉/i |
| --- | --- | --- | --- | --- |
| Original infinite coherent state, original X/P | 1 | 1/2 | 1/2 | 1 |
| Renormalized ψ_C, finite X_C/P_C | 15/16 | 39/128 | 9/16 | 3/4 |
| Embedded ψ_C, original X/P | 15/16 | 55/128 | 11/16 | 1 |

The second row's variance product is351/2048, exceeding its finite bound9/64. The third row's product is605/2048, exceeding the original1/4. A finite product below1/4 is not a violation of the uncertainty principle. Nor should every reduced X variance be attributed to the same cause.

## 14. A finite ground state does not certify infinite-space stability

The cutoff matrix for H=−n̂ is diagonal, with minimum−C. Increasing the cutoff lowers that minimum indefinitely. The infinite model is not bounded below and has no normalizable lowest-energy state. Solving one finite array exactly therefore does not establish stability of the infinite physical model.

For a bounded-below H with an appropriate domain, a variational subspace can instead give a Rayleigh–Ritz upper bound. Which finite matrix was used remains crucial: was it H_C=Π_C H Π_C, or another model constructed from truncated X/P? Cutoff convergence also requires checking the original operator, space and moment assumptions.

The experiment enumerates energies and classifies ground states only for diagonal finite H. It does not calculate arbitrary off-diagonal ground states, infinite systems, time evolution or optimization. Comparing the original coherent state's normally ordered mean energy with a finite-state energy does not establish that either state is a ground state.

## 15. A small tail does not bound every observable error

The trace distance between renormalized psi_C and the original pure coherent state is sqrt(epsilon). For a bounded observable A, its operator norm gives an expectation-error bound. The two states are compared using the **same A**; changing A by truncation is a separate effect.

$$
\left|\langle A\rangle_\alpha-\langle A\rangle_{\psi_C}\right|
\le2\|A\|\sqrt\varepsilon\qquad\text{for bounded }A.
$$

For an unbounded observable such as number, this cannot simply be treated as a guarantee with a finite constant. Consider an original state sqrt(1−epsilon)|0〉+sqrt(epsilon)|M〉with M>C. The tail is epsilon, but mean-number error after truncation is epsilon M and can be arbitrarily large. A small tail of a general state does not certify energy or higher moments.

Within the specified coherent family, knowledge of the distribution lets us calculate the mean-number difference as mu p_top. This motivates recording norm loss, number moments, energy, quadratures and operator-projection error separately in AI experiments. Scores from different physical models, state families, observable sets or cutoff rules are not immediately comparable.

## 16. Eight problems separating boundaries and states

**1. Ladder and trace.** At C=2, find the diagonals of a_C†a_C, a_Ca_C† and their commutator. Compare traces.

**Explanation.** They are(0,1,2), (1,2,0) and(1,1,−2). The commutator has trace0 and the identity has trace3. This algebraic distinction, rather than roundoff, prevents preserving the infinite CCR in a finite matrix.

**2. Top-state uncertainty.** For |3〉at C=3, calculate finite variances and the modified bound. What changes when original X/P are used?

**Explanation.** Both finite variances are3/2 with zero covariance, giving product9/4. Commutator expectation−3 gives the same bound. Each original variance adds2 and becomes7/2. These are different observables, not two reports of the same measurement.

**3. A missing hopping path.** Compare the action, mean and variance on |2,2〉with cutoffs(3,2) and(3,3).

**Explanation.** Both retain sqrt(6)|3,1〉. Only a second cutoff3 retains sqrt(6)|1,3〉as well. Outputs are orthogonal to the input, so both means vanish. Their squared norms give variances6 and12. The mean alone can miss boundary effects.

**4. Infinite coherence and finite vacuum.** At alpha=1,C=0, find the tail, renormalized state and annihilator eigenstate residual.

**Explanation.** Z₀=1, retained probability is exp(−1) and the tail is1−exp(−1). The conditional state is the vacuum but is not the original coherent state. With a_C=0, the squared norm of(a_C−1)|0〉is1. Obtaining a normalized vacuum does not erase discarded probability.

**5. Three values at alpha=1,C=3.** Calculate top probability, mean number and residual squared, along with fidelity squared and raw projection error.

**Explanation.** Z₃=8/3 gives p_top=1/16, mean15/16 and residual1/16. Retained S=(8/3)exp(−1) is fidelity squared, and the unnormalized squared error is1−S. The normalized squared distance must be evaluated separately as2(1−sqrt(S)).

**6. Same vacuum, different Hamiltonians.** At C=0, compare projected oscillator H_C/(hbar Omega) with(X_C²+P_C²)/2.

**Explanation.** The first is n̂_C+1/2=1/2, while the second vanishes. Two expressions equal in the infinite space differ when products of truncated operators are used. Define the Hamiltonian before comparing energies.

**7. Two-mode tail.** Given retained probabilities S₀ and S₁, calculate total tail. Why is it different from summing mode tails?

**Explanation.** For an independent product coherent input and rectangular cutoffs, joint retention is S₀S₁ and tail is1−S₀S₁. Equivalently it is epsilon0+epsilon1−epsilon0 epsilon1. The simple sum counts simultaneous exclusion twice. Do not automatically use this independent product for a correlated preparation.

**8. Small norm error, large number error.** Let M>C for the state sqrt(1−epsilon)|0〉+sqrt(epsilon)|M〉. What does its cutoff establish, and what does it not?

**Explanation.** Tail and unnormalized squared distance are epsilon. The renormalized state is the vacuum, while original mean number is epsilon M. Increasing M can enlarge number error at a fixed small epsilon. A bounded-observable norm bound cannot be applied unchanged to unbounded number.

## 17. Accurate arithmetic does not erase differences in definition

The experiment distinguishes raw-state and coherent preparation. Raw inputs specify nonzero complex amplitudes directly; coherent inputs generate coefficients from rational complex alpha. At maximum occupancies0…3, square roots of2,3 and6 in the ladder and hopping can be retained as exact algebraic coefficients. Do not generalize this small-cutoff number field to one fixed representation of every factorial in the infinite series.

Infinite exponential factors, tails and normalized coordinates are compared through certified intervals. Distinguish a displayed decimal zero from an exact zero, and check definitions of original probability, conditional probability, top occupation and residual. The per-mode mu≤16 range is the calculator's verified scope, not a theoretical restriction on alpha. Changing a cutoff control is also different from evolving under a Hamiltonian in physical time.

First predict the boundary rule for vacuum, top and intermediate occupations. Next change alpha's magnitude and phase to compare original and finite means. Finally fix a state and change only the observable to check the section6 correction. Explain what was truncated even when the pictures look similar.

## 18. Identify the approximation before moving to another representation

We can now separate the changes introduced when an infinite boson space becomes a finite array. Creation factors, boundary commutators, products of observables, conditional states and discarded probability are distinct pieces of the calculation. One low energy or a small loss cannot replace all of these conditions. MPS will address a different type of representation compression, while Schmidt truncation quantifies lost bipartition weights.

For further study, consult the [FAU/Max Planck exercises on oscillators, coherence and quadratures](https://mpl.mpg.de/fileadmin/user_upload/Exercise_1.pdf), [Stuttgart's coherent-series exercises](https://www.itp3.uni-stuttgart.de/downloads/Adv._Quantum_Theory_Engl_ws11.12/ex9.pdf) and [NIST DLMF's incomplete-gamma identities](https://dlmf.nist.gov/8.4). Extending the model to actual materials, optical modes or large systems requires stating domains, convergence, observables and resource assumptions again.
