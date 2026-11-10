# Does an aggregated state close its own dynamics? Markov lumping and memory

This manuscript accompanies a numerical foundation under review. It is not yet registered in Interactive Book and does not itself provide a browser lab.

## 1. Different internal states behind the same display

Suppose a dashboard labels every pending job A and every completed job B. Pending jobs include nearly finished jobs and jobs with substantial work remaining. Equal current fractions in A need not imply equal fractions moving to B next: the internal composition matters.

This is a dynamical question about coarse graining. Earlier macro-observation units grouped microscopic states to study probabilities and information loss. Here we ask whether the grouped variables also determine their next update. Compressing an output differs from predicting future outputs using the compressed state alone.

## 2. Distinguish a probability law from one observation

Specify a discrete Markov chain on states 0, 1 and 2. Columns of T index the starting state and rows the destination. The probability vector is a column with total mass one.

$$
p_{k+1}=Tp_k,\qquad T_{ij}=\Pr(X_{k+1}=i\mid X_k=j),\qquad \sum_iT_{ij}=1.
$$

The index k counts updates. No clock in seconds or continuous-time generator has been specified. Remaining outgoing probability is a self-transition. We propagate the exact distribution of a prepared initial law rather than randomly sampling a trajectory. Preparations concentrated at a single microscopic state are allowed.

## 3. Aggregation and a chosen lift

Group A={0,1} and B={2}. The first macro component is the sum of the first two micro probabilities. To represent macro mass A microscopically again, choose an internal fraction alpha.

$$
C=\begin{pmatrix}1&1&0\\0&0&1\end{pmatrix},\quad m=Cp,\quad R=\begin{pmatrix}\alpha&0\\1-\alpha&0\\0&1\end{pmatrix},\quad CR=I_2,\quad 0\leq\alpha\leq1.
$$

Aggregation discards composition. A lift chooses one composition rather than recovering the discarded truth. Actual p can differ from Rm. Alpha belongs to the specified lifting rule; it is not automatically the true conditional probability at every subsequent time. Boundary lifts are allowed, but they do not guarantee decay of hidden dynamics.

## 4. Closure for every initial law

Exact prediction from the macro state requires starting states within one block to have equal probabilities of reaching every destination block. For these two blocks, compare exits from states 0 and 1 into state 2.

$$
CT=AC\quad\Longleftrightarrow\quad T_{20}=T_{21},\qquad A=CTR.
$$

When this holds, the next macro law is always Am. The statement concerns every initial law rather than one stationary preparation. We call this property strong lumpability. Necessity follows by comparing initial point masses at states 0 and 1, which have the same macro state. Sufficiency follows because the first two columns of CT coincide. Terminology varies across aggregation literature, so the algebraic condition and initialization scope define our usage.

## 5. Stationary agreement can hide different futures

Every row and column of the following matrix sums to one, so the uniform microscopic law is stationary. Nevertheless, exits to B differ between states 0 and 1.

$$
T=\begin{pmatrix}1/2&1/4&1/4\\1/4&0&3/4\\1/4&3/4&0\end{pmatrix}.
$$

Use alpha=1/2 in this comparison. The stationary macro law is (2/3,1/3), which A also preserves.

| Initial micro law | Initial macro law | Actual next B | Next B predicted by A |
| --- | --- | --- | --- |
| (1,0,0) | (1,0) | 1/4 | 1/2 |
| (0,1,0) | (1,0) | 3/4 | 1/2 |
| (1/3,1/3,1/3) | (2/3,1/3) | 1/3 | 1/3 |

Validation using only stationary data can see the success of the final row while missing the first two. Those preparations present the same macro input and different correct next outputs. A deterministic predictor receiving only current m cannot fit both exactly. Better parameter estimation alone cannot recover information removed by the observation map.

## 6. Separate discarded information exactly

Split the lifted component from the remaining hidden component. The latter is a signed deviation, not a probability law.

$$
H=I_3-RC,\qquad h_k=Hp_k,\qquad p_k=Rm_k+h_k,\qquad Ch_k=0.
$$

Increasing the probability of state 0 while decreasing state 1 by the same amount leaves the macro sum unchanged. This is the hidden direction. H is idempotent, but we do not claim that this chosen lift gives a Euclidean orthogonal projection. Negative entries in the hidden deviation are expected. Clipping them to zero destroys the decomposition.

## 7. Propagate resolved and hidden parts together

Insert the decomposition into the micro update, then apply the aggregation and hidden projection.

$$
m_{k+1}=Am_k+Bh_k,\qquad h_{k+1}=Dm_k+Eh_k,\qquad B=CT,\quad D=HTR,\quad E=HT.
$$

The hidden variable is still present, but its equation explains how previous macro states create new hidden components. Even if the initial hidden component vanishes, the term Dm can produce one immediately. Agreement with a lift at initialization therefore does not justify dropping memory at later times.

## 8. Eliminating a variable leaves its history

Repeated substitution eliminates the hidden variable. This is an exact rewriting of the microscopic linear update.

$$
m_{k+1}=Am_k+F_k+\sum_{j=0}^{k-1}K_{k-1-j}m_j,\qquad F_k=BE^kh_0,\quad K_\ell=BE^\ell D.
$$

The first term uses the current macro state, F retains the hidden initial preparation, and the sum retains past macro states. At the initial update the sum is empty. We derive this discrete identity directly. Section 2 of the [Gouasmi–Parish–Duraisamy research paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5627381/) presents elimination of hidden variables in continuous linear systems. Exact memory equations do not by themselves establish computational savings or short memory.

## 9. Memory kernels are not transition probabilities

Kernel entries can be signed and their columns sum to zero. Positive and negative history contributions combine with the current term to give the correct law. Do not interpret each contribution separately as a probability.

The hidden space here is one-dimensional, so kernel dependence on lag reduces to powers of one scalar. Write a with subscripts i,j for the probability of moving from state i to state j.

$$
\lambda=1-a_{01}-a_{10}-(1-\alpha)a_{02}-\alpha a_{12},\qquad K_\ell=\lambda^\ell K_0.
$$

Positive scalar powers preserve sign; negative powers can alternate. Magnitudes near one imply slow decay. At zero eigenvalue, kernels after the first vanish, but the first kernel need not vanish. Conversely, a persistent hidden mode may be invisible to the resolved variables through B. Hidden decay and macro approximation error are different diagnostics.

## 10. Compare three trajectories

The reference propagates the true micro law before aggregating it. Compare this with repeated application of A and with a finite-memory approximation retaining the complete initial forcing and only the most recent L history terms.

$$
\widetilde m_{k+1}^{(L)}=A\widetilde m_k^{(L)}+F_k+\sum_{j=\max(0,k-L)}^{k-1}K_{k-1-j}\widetilde m_j^{(L)}.
$$

At depth zero, initial forcing is still retained, so this differs from repeating A alone. Retaining every lag needed for the displayed trajectory reproduces the exact result. Short memory does not universally preserve nonnegative probabilities, and increasing depth by one does not establish monotonically decreasing global error.

| Method | Hidden initial term | Memory sum | Established property |
| --- | --- | --- | --- |
| Micro propagation then aggregation | Included microscopically | Included microscopically | Exact probability law |
| Repeating A | Discarded | Discarded | Probability law, potentially different from the true macro process |
| Finite memory L | Retained | Most recent L terms | Generally a signed approximation; accuracy and positivity need checking |

## 11. Negative probabilities expose approximation failure

Take probability one for 0 to 2, one quarter for 1 to 0, and one half each for 2 to 0 and 2 to 1. Other off-diagonal probabilities vanish. With alpha zero, initial state 0 and zero memory depth, the first macro result is correct but the second approximation is (5/4,−1/4). The true second macro law is (1,0).

The microscopic chain has not produced an invalid law. A truncated approximation has left probability space. Clipping the negative component or renormalizing defines a different algorithm and should not be presented as the original approximation. Its half-L1 difference from the true law is one quarter, but this is not total variation between two probability laws.


$$
\varepsilon_L(k)=\frac12\sum_{b\in\{A,B\}}\left|m_b(k)-\widetilde m_b^{(L)}(k)\right|.
$$

Evaluating omitted kernels on the exact past trajectory gives a local omission term, not a global rollout-error bound. Earlier approximation errors feed back into later updates. A bound requires additional control of stability and kernel magnitudes.

## 12. Exact arithmetic still has display limits

Inputs, matrix powers, kernels, probabilities and residuals remain exact rational values. Integer magnitudes are at most one million; the number of updates and memory depth range from zero to 32. Initial nonnegative weights are normalized once. Outgoing transition weights determine each self-transition from the remainder. Probability laws are not renormalized after individual updates.

An exact coefficient can be smaller than ordinary floating-point display range. For example, the lag-31 kernel for eigenvalue10⁻¹² can remain nonzero while its Number projection is unavailable. Inspect the sign and rational numerator and denominator rather than substituting zero. Mass-balance residuals and full-memory reconstruction residuals must be exactly zero.

Records run from the initial step through the requested endpoint. Each also contains a separately identified next-update reconstruction diagnostic, including the final record. Do not mistake that diagnostic for an extra point added to the displayed trajectory. Interpolated curves would not establish exact intermediate-time probabilities either.

## 13. AI state representations and effective descriptions

A small latent representation need not be a sufficient predictive state. If two preparations map to the same representation and have different futures, decide which history, additional feature or inferred probability state to retain. Arbitrary random noise does not automatically restore missing preparation information and memory.

Changing alpha alters the instantaneous matrix and the split between initial forcing and memory, but not the true aggregated trajectory. This distinguishes a lifting choice from a physical observation change. Our model uses one fixed partition of a finite discrete chain. Continuous time, fast-variable elimination, Markov limits with time-scale assumptions and initial layers, spatial blocking, renormalization, effective couplings and universality require separate derivations and further study. The finite checks here do not complete that larger scope.

## 14. Exercises with explanations

**1. Distinguishing hidden preparations.** Do initial states 0 and 1 share a macro state, and what are their next B probabilities in the first table? Solution: both have probability one in A, but their next B probabilities are one quarter and three quarters. Current macro state alone cannot distinguish the correct futures.

**2. The stationary trap.** Does matching the uniform stationary micro law and its macro stationary law prove strong lumpability? Solution: no. The all-initial-law criterion compares the two exits into B. They differ here, even though stationary behavior matches.

**3. Is zero initial hidden state enough?** Compare second-step predictions for initial law (1/2,1/2,0) and alpha one half. Solution: both first macro laws are (1/2,1/2). The true second law is (13/16,3/16), while repeated A gives (3/4,1/4), with total variation1/16. The resolved-to-hidden term has created a new hidden component.

**4. Lifting invariance.** What must remain unchanged when alpha changes for a fixed micro preparation? Solution: aggregation, transition and initial micro law are unchanged, so the exact macro trajectory is unchanged. Instantaneous A, forcing and kernels can change. Check their complete reconstruction rather than any term alone.

**5. Interpreting negativity.** How should a truncated second result (5/4,−1/4) be reported? Solution: call it a signed approximation rather than a probability law. Its half-L1 error from (1,0) is one quarter. Do not silently clip or renormalize it.

**6. Memory versus numerical zero.** Can zero eigenvalue and eigenvalue10⁻¹² be treated identically? Solution: no. Zero gives algebraically zero kernels after lag zero. A small positive eigenvalue gives nonzero higher-lag terms even below display range. Retain exact sign and rational evidence, and specify the approximation and its error criterion explicitly.
