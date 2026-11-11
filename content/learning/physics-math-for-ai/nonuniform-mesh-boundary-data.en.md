## Changing a mesh differs from changing a boundary

A nonuniform mesh concentrates computation locations in selected regions. Here x_i=(i/N)^r determines the mesh, and each element uses its actual length. Replacing all lengths with a single average h would assemble a different operator. The upper diagram displays those actual node locations.

The base function is u=x³+ax with source -6x. A selected bias changes the source to -6x+δ. Boundary modes are two Dirichlet values, left Dirichlet with right Neumann, and pure Neumann. Even with identical source data, changing boundary conditions changes the mathematical problem.

## Assemble elements and eliminate known values

For an element of length h_e, linear FEM uses

$$
K_e=\frac1{h_e}\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\qquad
K_{II}u_I=b_I-K_{IB}u_B.
$$

Eliminating a Dirichlet value requires updating neighboring right-hand sides, not merely replacing its row. Natural Neumann data use the outward derivative: g_L=-u′(0)=-a and g_R=u′(1)=3+a. Applying the same coordinate-direction sign at both endpoints would make the left boundary wrong.

Loads are integrated by two-point Gauss quadrature. The current linear source multiplied by a linear hat function is integrated exactly. In this one-dimensional constant-coefficient setting, exact weak loads give exact FEM nodal values. Nevertheless the linear interpolant inside an element differs from the continuous cubic. This special nodal fact is not a general FEM exactness claim.

## Existence and uniqueness for pure Neumann data

$$
\int_0^1 f(x)\,dx+g_L+g_R=0,\qquad K\mathbf1=0.
$$

The first equation is compatibility. The second expresses the freedom to add a constant without changing derivatives. For the selected source, the compatibility sum equals δ. If δ is nonzero, prescribed boundary fluxes and interior source cannot balance, so no solution exists. The experiment then omits a blue numerical solution. Orange remains the base reference function, not a claimed solution of the incompatible problem.

For compatible data, imposing u(0)=0 selects one representative from the constant family. Gauge fixing removes nonuniqueness; it does not repair incompatible data. Distinguish the original unconstrained stiffness nullity from the nonsingular system after selecting a gauge.

## A gauge cannot turn incompatible loads into a solution

For the symmetric Neumann stiffness matrix on a connected 1D mesh, K1=0 and its range is orthogonal to constants. Thus Σbᵢ=0 is necessary and, with nullity one, sufficient for solvability. Fixing one nodal value chooses a representative only after this condition holds. If an incompatible load is pinned anyway, the dropped equation absorbs the missing balance.

The live source perturbation has analytic compatibility sum δ. It now rejects every nonzero prescribed δ, even when rounded quadrature loads hide it or a generic tolerance would accept it. The independently assembled floating sum remains available in the calculation API; it does not define the existence criterion for these known data.

A uniform source correction δ has nodal load weights wᵢ=∫φᵢ. On a nonuniform mesh these are not uniform. For nodes (0,1/4,1), they are:

| Node | Constant-source load weight |
|---|---:|
| 0 | 1/8 |
| 1/4 | 1/2 |
| 1 | 3/8 |

```python
from fractions import Fraction as F
xs=[F(0),F(1,4),F(1)]
K=[[F(0) for _ in xs] for _ in xs];weights=[F(0)]*3
for j in range(2):
    h=xs[j+1]-xs[j]
    K[j][j]+=1/h;K[j+1][j+1]+=1/h
    K[j][j+1]-=1/h;K[j+1][j]-=1/h
    weights[j]+=h/2;weights[j+1]+=h/2
base=[x**3 for x in xs];delta=F(1,10**20)
rhs=[sum(a*b for a,b in zip(row,base))+delta*w for row,w in zip(K,weights)]
assert sum(rhs)==delta and weights==[F(1,8),F(1,2),F(3,8)]
a,b,c=K[1][1],K[1][2],K[2][2]
det=a*c-b*b
u=[F(0),(c*rhs[1]-b*rhs[2])/det,(a*rhs[2]-b*rhs[1])/det]
residual=[sum(x*y for x,y in zip(row,u))-r for row,r in zip(K,rhs)]
assert residual==[-delta,0,0]
print(weights,residual)
```

The exact pinned solve leaves residual −δ in its omitted equation. Subtracting δw removes a uniform source bias. In contrast, the Euclidean load projection b−δ1/3 is a different correction, despite also making the load sum zero. Existence, choice of gauge and choice of which physical data to revise are distinct decisions.

## Worked example and exercise

For a=1/2 and δ=0, the source integral is -3, and outward derivatives are -1/2 and 7/2. Their sum is zero, so pure Neumann solutions exist up to a constant. Choosing u(0)=0 selects the displayed base function.

Exercise: retain the boundary data but set δ=0.2. Solution: the source integral becomes -2.8 and the full compatibility sum becomes 0.2. Adding a constant changes neither derivatives nor fluxes, so a gauge cannot fix the mismatch. The physically incorrect source or boundary data must instead be revised.

## Boundaries and gauges in AI models

PINNs, mesh neural operators and pressure or potential estimators are sensitive to boundary signs and constant freedoms. A small observation loss can coexist with incompatible integral constraints. Check operator residuals, boundary weak residuals and existence conditions separately. Do not generalize this special one-dimensional nodal exactness to higher-dimensional or variable-coefficient problems.

Reference: [MIT numerical PDE course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
