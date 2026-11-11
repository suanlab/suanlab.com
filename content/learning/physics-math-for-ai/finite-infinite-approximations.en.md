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

## From coercivity to existence and stability

Let a be a bounded bilinear form on a real Hilbert space V and ℓ a continuous functional. Riesz gives bounded A and g with ⟨Au,v⟩=a(u,v),ℓ(v)=⟨g,v⟩. Coercivity a(u,u)≥α‖u‖² and Cauchy–Schwarz imply ‖Au‖≥α‖u‖. Thus A is injective with closed range: if Au_n is Cauchy, u_n is Cauchy, and completeness plus continuity preserve the limit.

If y is orthogonal to the range, a(u,y)=0 for every u. Taking u=y forces y=0. The closed range is therefore dense and equals V. There is a unique solution Au=g with ‖u‖≤‖ℓ‖/α. This proves Lax–Milgram in this setting without requiring symmetry. Restriction to a closed conforming subspace gives Galerkin existence too.

Galerkin orthogonality a(e,v_N)=0, e=u−u_N, yields for any v_N

$$
\alpha\|e\|^2\le a(e,e)=a(e,u-v_N)
\le M\|e\|\|u-v_N\|.
$$

Taking the infimum gives Céa's bound. For A=[[2,1],[1,2]], right side (1,0) and V_N=span(e₁), the exact solution is (2/3,−1/3), while Galerkin gives (1/2,0), different from Euclidean projection (2/3,0). Its Euclidean error √5/6≈.372678 exceeds the best Euclidean error 1/3 but satisfies the M/α=3 bound. In this symmetric positive example Galerkin is optimal in the energy norm.

## Why increasing inverse-problem rank can hurt

For T=diag(1/k) and true x_k=1/k, data are y_k=1/k². With noise ‖η‖₂≤δ, truncated inversion gives x̂_k=k(y_k+η_k) for k≤N and zero afterward. Tail bias and retained noise have orthogonal supports, so

$$
\sup_{\|\eta\|_2\le\delta}\|x-\widehat x_N\|_2^2
=\sum_{k>N}\frac1{k^2}+N^2\delta^2.
$$

Noise η=δe_N attains the bound. This adversarial noise changes with N: the table and figure are a worst-case envelope over an allowed noise set, not observations from one fixed noisy sequence.

![Truncation bias decreases while noise amplification Nδ increases, making worst-case total inversion error rise again. Here δ=.05 and the rank axis uses log2 spacing with doubling ticks.](/assets/learning/functional-inverse-rank-noise-en.svg)

| N, δ=.05 | Truncation bias | Maximum noise Nδ | Worst total error |
|---|---|---|---|
| 2 | .628438 | .100000 | .636344 |
| 4 | .470450 | .200000 | .511198 |
| 8 | .342800 | .400000 | .526794 |
| 16 | .246145 | .800000 | .837011 |
| 32 | .175405 | 1.600000 | 1.609586 |

The code checks fraction-valued Galerkin equations and inversion of the constructed worst-case noise. Ranks such as 32 belong to this separate code example; the screen's control stops at sixteen. Comparing ranks 1..64 gives minimum six for this envelope. For this particular target, N→∞ with Nδ→0 gives convergence as δ→0. Balancing squared tail≈1/N with noise²=N²δ² suggests N of order δ⁻²ᐟ³ and error of order δ¹ᐟ³. These are not universal rates for every inverse problem.

**Exercise:** Does zero discrete residual at each N imply better recovery? **Explanation:** No. Exactly fitting noisy data can amplify Nδ. Check residual, approximation, stability and noise together.

```python
from fractions import Fraction as F
from math import sqrt,pi

A=[[F(2),F(1)],[F(1),F(2)]]
u=[F(2,3),F(-1,3)]
galerkin=[F(1,2),F(0)]
euclidean=[u[0],F(0)]
def energy(p,q):return sum(p[i]*A[i][j]*q[j] for i in range(2) for j in range(2))
e=[u[i]-galerkin[i] for i in range(2)]
e_best=[u[i]-euclidean[i] for i in range(2)]
assert energy(e,[F(1),F(0)])==0
assert sum(x*x for x in e)==F(5,36)
assert energy(e,e)==F(1,6)<energy(e_best,e_best)==F(2,9)
assert sqrt(float(sum(x*x for x in e)))<=3*sqrt(float(sum(x*x for x in e_best)))
print('Galerkin Euclidean error, best Euclidean error:',sqrt(5)/6,1/3)

noise=F(1,20)
def errors(N):
    tail2=pi*pi/6-sum(1/(k*k) for k in range(1,N+1))
    # Different worst-case noise eta=delta*e_N is allowed for each N.
    data=[F(1,k*k)+(noise if k==N else 0) for k in range(1,N+1)]
    recovered=[k*y for k,y in enumerate(data,1)]
    noise2=sum((recovered[k-1]-F(1,k))**2 for k in range(1,N+1))
    assert noise2==(N*noise)**2
    return sqrt(tail2),float(N*noise),sqrt(tail2+float(noise2))
for N in [2,4,8,16,32]:
    print('rank, bias, maximum noise amplification, exact worst-case error:',
          N,*(round(x,6) for x in errors(N)))
best=min(range(1,65),key=lambda N:errors(N)[2])
print('best rank among 1..64 for this envelope:',best)
```


## Exercises with explanations

1. Error of e17 at N=16? **One**, because it is still omitted.
2. Does convergence for 1/k provide one N for every unit input? **No.** Later basis vectors remain counterexamples.
3. Does a small forward tail ensure a stable inverse? **No.** Here the tail shrinks while inverse norm grows as N.

Further reading: [Imperial Finite Element Course, Convergence](https://finite-element.github.io/L5_convergence.html).
