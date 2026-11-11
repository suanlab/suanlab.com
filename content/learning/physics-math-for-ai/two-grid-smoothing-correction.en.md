## Why smooth errors survive on a fine grid

Iterative linear solvers respond to the spatial shape of error, not only its magnitude. Neighbor-based relaxation can quickly damp oscillatory components while barely changing slowly varying ones. Multigrid represents that smooth fine-grid error more cheaply on a coarser grid and corrects it there.

The exact solution here is zero, and the initial error is e(x)=sin(mπx). Thus the true error is known without using another PDE solver as an assumed reference. Blue shows error after actual cycles, orange shows the initial error, and the table reports weighted L2 error and residual separately.

## Smoothing and the residual equation

The Dirichlet Poisson matrix has diagonal 2/h² and neighboring entries -1/h². Weighted Jacobi applies

$$
e\leftarrow e-\omega D^{-1}Ae,\qquad r=-Ae.
$$

A low-frequency sine has a small eigenvalue and receives only a small relative correction. Oscillatory components can be damped more strongly when the weight is chosen appropriately. The largest allowed weight is not automatically the best smoother. The experiment uses the selected number of sweeps both before and after coarse correction.

## Assemble the actual coarse operator

P linearly interpolates coarse nodes, and R=Pᵀ/2 performs full weighting. The coarse matrix is assembled from the current fine operator and those transfer matrices:

$$
A_H=RA_hP,\qquad A_H e_H=Rr,\qquad
e\leftarrow e+Pe_H.
$$

A small pivoted direct solve computes the coarse correction. Post-smoothing follows, and the cycle is actually repeated. Transfer dimensions, homogeneous boundary treatment and the sign of the residual all matter. Reversing the residual sign would add the represented error rather than remove it.

## Coarse correction is an energy projection

For symmetric positive-definite A and full-column-rank P, the scalar factor in R=Pᵀ/2 cancels from the exact coarse correction. Its error operator is C=I−P(PᵀAP)⁻¹PᵀA. Direct multiplication gives C²=C, CᵀA=AC, and PᵀAC=0. Therefore

$$
\|Ce\|_A^2=\|e\|_A^2-\|e-Ce\|_A^2.
$$

The correction removes the closest coarse-space component in the A-energy norm, not generally in Euclidean distance. This proves nonexpansion for exact coarse correction but not uniform two-grid convergence without a suitable smoothing and approximation argument.

Weighted Jacobi's high-frequency band θ∈[π/2,π] has worst factor max(|1−ω|,|1−2ω|). Balancing the two endpoint errors gives the minimax choice ω=2/3 and worst factor 1/3. The constant concerns this uniform 1D Poisson band; it is not an optimal weight for every operator.

![Weighted Jacobi factors across spatial frequencies, highlighting the high-frequency band and the minimax weight.](/assets/learning/scientific-smoothing-en.svg)

| ω | Worst factor on the stated high band |
|---|---:|
| 1/2 | 1/2 |
| 2/3 | 1/3 |
| 1 | 1 |

```python
from fractions import Fraction as F
A=[[F(2),F(-1),F(0)],[F(-1),F(2),F(-1)],[F(0),F(-1),F(2)]]
p=[F(1,2),F(1),F(1,2)];e=[F(1),F(-1),F(2)]
mv=lambda v:[sum(a*b for a,b in zip(row,v)) for row in A]
dot=lambda a,b:sum(x*y for x,y in zip(a,b))
coefficient=dot(p,mv(e))/dot(p,mv(p))
removed=[coefficient*x for x in p]
corrected=[x-y for x,y in zip(e,removed)]
assert dot(p,mv(corrected))==0
assert dot(e,mv(e))==dot(corrected,mv(corrected))+dot(removed,mv(removed))
for weight in (F(1,2),F(2,3),F(1)):
    print(weight,max(abs(1-weight),abs(1-2*weight)))
```

The rational projection below verifies orthogonality and Pythagoras on three fine unknowns. The live solver assembles its actual Galerkin coarse matrix and performs both smoothing passes. A neural pooling layer supplies an analogous diagram only after its operator, norm and correction equation have been specified.

## Worked example and exercise

Compare m=1 smooth error with m=7 oscillatory error on the same fine grid. A Jacobi sine-mode multiplier is 1-ω(1-cos(mπh)). For small mπh it is close to one, explaining slow progress. At suitable higher frequencies its absolute value is smaller. Coarse correction then addresses much of the smooth component left by relaxation.

Exercise: assume the fine error is exactly Pz and perform a coarse correction without smoothing. Solution: r=-APz, so e_H=(RAP)^{-1}R(-APz)=-z and e+Pe_H=0. This proves exact removal of representable coarse error in exact arithmetic. It does not prove removal of every fine-grid component after one correction.

## Multiscale computation and AI

Pooling and upsampling in a mesh network may resemble multigrid visually without solving the same residual equation. Identify the operator and norm being corrected. Physical homogenization is also a separate task: effective coefficients can require scale separation and cell problems. This experiment computes a genuine two-level algebraic correction, not an entire physical coarse-graining theory.

For broader verification, inspect the Galerkin identity, coarse-space removal, reductions across several meshes and actual work. One small residual on one mesh cannot prove mesh-independent convergence or linear computational complexity. Extension to variable coefficients and anisotropy may require different smoothers and transfer operators.

Reference: [MIT numerical PDE course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
