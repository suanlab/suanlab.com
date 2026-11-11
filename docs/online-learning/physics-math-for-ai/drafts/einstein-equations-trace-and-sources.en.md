## 1. Relate curvature to stress and energy

Use the spacetime signature $(-,+,+,+)$ in this chapter. With the curvature convention used here, the field equation is
$$
G_{\mu\nu}+\Lambda g_{\mu\nu}=\kappa T_{\mu\nu},\qquad
G_{\mu\nu}=R_{\mu\nu}-\tfrac12R g_{\mu\nu},
\qquad \kappa=\frac{8\pi G}{c^4}.
$$
The equation relates tensors at each spacetime point. The stress tensor includes energy density, flux and spatial stress. It is not a prescription to identify coordinate components in unrelated coordinate systems. [Tong, The Einstein Equations §4.1](https://davidtong.org/teaching/general-relativity/grhtml/S4)

## 2. Take the trace in four dimensions

Contract with $g^{\mu\nu}$ and use $g^{\mu\nu}g_{\mu\nu}=4$. The trace of the Einstein tensor is $-R$, so
$$
R=4\Lambda-\kappa T,\qquad
R_{\mu\nu}=\kappa(T_{\mu\nu}-\tfrac12Tg_{\mu\nu})+\Lambda g_{\mu\nu}.
$$
The second equation is the trace-reversed form. It is often useful when the matter tensor is simple. Changing dimension changes the trace coefficients.

## 3. Evaluate a perfect fluid in its rest frame

Use an orthonormal frame at one event, with energy density $\rho$ and isotropic pressure $p$ in the same units. The lower-index stress components are $(\rho,p,p,p)$ on the diagonal, so $T=-\rho+3p$. Therefore
$$
R_{\hat0\hat0}=\tfrac{\kappa}{2}(\rho+3p)-\Lambda,\qquad
R_{\hat{i}\hat{i}}=\tfrac{\kappa}{2}(\rho-p)+\Lambda
$$
for each spatial direction $i$, without summing over $i$. Pressure enters curvature along with energy density.

## 4. Compare dust, radiation and vacuum

For dust with $p=0$ and $\Lambda=0$, the scalar curvature is $\kappa\rho$. For radiation with $p=\rho/3$, the trace vanishes and so does $R$ at $\Lambda=0$. The time component of the Ricci tensor is nevertheless $\kappa\rho$ for this radiation. For vacuum at $\Lambda=0$, all Ricci components vanish. Full Riemann curvature can still be present, as the exterior of a spherical mass shows. Curvature data lost in contraction cannot be inferred from the trace.

## 5. Check consistency and compute examples

The contracted Bianchi identity gives $\nabla^\mu G_{\mu\nu}=0$. For constant $\Lambda$, metric compatibility then implies $\nabla^\mu T_{\mu\nu}=0$. This consistency condition constrains matter evolution. To predict a spacetime, one also needs a matter model and appropriate initial or boundary data.

```python
from math import isclose

def curvature(rho,p,lam):
    return 4*lam+rho-3*p, (rho+3*p)/2-lam, (rho-p)/2+lam

assert curvature(3,0,0)[0] == 3
assert curvature(3,1,0) == (0,3,1)
for rho,p,lam in [(3,1,0),(2,0,1),(0,0,0.5)]:
    scalar, time, space = curvature(rho,p,lam)
    assert isclose(scalar,-time+3*space)
print(curvature(3,1,0))
```

## 6. Inspect what each result determines

- Use the metric signature when taking the trace of a tensor.
- Change pressure at fixed density and compare $R$ and $R_{\hat0\hat0}$.
- Add a cosmological constant and verify the shifts in the two components and the scalar.
- State whether your calculation determines a scalar, the Ricci tensor at an event, or a spacetime solution.
