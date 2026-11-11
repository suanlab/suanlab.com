## A real loss with complex parameters

A complex parameter z=x+iy does not require a holomorphic learning objective. The real loss |z|²=x²+y² is nonholomorphic but smooth on R² and readily optimized.

Wirtinger operators are combinations of real derivatives:

$$
\partial_z=\frac12(\partial_x-i\partial_y),\qquad
\partial_{\bar z}=\frac12(\partial_x+i\partial_y).
$$

Treating z and its conjugate formally as independent is a computational rule derived from these definitions, not a doubling of the underlying real degrees of freedom.

## Recover the real gradient

For L=|z−c|²,

$$
\frac{\partial L}{\partial z}=\overline{z-c},\qquad
\frac{\partial L}{\partial\bar z}=z-c.
$$

Encoding the Euclidean real gradient as Lₓ+iLᵧ gives 2∂L/∂z̄. The factor two follows from the stated half-factor convention. Check a software library’s convention rather than transferring names alone.

## Why this gives descent

The real differential is

$$
dL=2\operatorname{Re}\left[\left(\frac{\partial L}{\partial\bar z}\right)^*dz\right].
$$

Choosing dz=−2η∂L/∂z̄ makes the first-order change nonpositive. Using the other Wirtinger derivative without accounting for conjugation can instead move away from the target.

## Solve the quadratic update exactly

The lab applies z_next=z−2η(z−c), so

$$
e_{\text{next}}=(1-2\eta)e,\qquad
L_{\text{next}}=(1-2\eta)^2L.
$$

For 0<η<1 the error contracts. At η=1 it changes sign without shrinking; above one, nonzero errors grow. At zero there is no update; at one half, one step reaches the minimizer. This range belongs to this explicit quadratic, not every complex objective.

At z=1+i,c=0,η=0.25, the next point is 0.5+0.5i and loss falls from two to one half. Shifting c changes the displacement used by the gradient.

## Read the implementation

Blue is six actual parameter updates; orange is the minimizer. This is a parameter-plane trajectory, not a loss-height surface. Loss values are separate readouts.

Independent real/imaginary finite differences can validate a complex autodiff convention. The engine here evaluates analytic expressions and does not claim to run a specific autodiff library. Nonsmooth magnitudes and branch operations need additional care.

## Follow both input paths through a real-linear layer

A useful extension is s=az+b·conj(z), with fixed complex a,b. This is real linear; b≠0 introduces a conjugate path. For residual r=s−c and real loss L=|r|², differentiate L=r·conj(r) using the declared half-factor convention:

$$
\frac{\partial L}{\partial\bar z}=\bar a r+b\bar r,\qquad
g=L_x+iL_y=2(\bar a r+b\bar r).
$$

The first term comes from differentiating conj(r); the second comes from differentiating r itself. Keeping only the first term silently treats the layer as holomorphic and loses the conjugate path.

More generally, if g_s encodes a real downstream loss gradient at s, the input gradient is

$$
g_z=\bar a g_s+b\bar g_s.
$$

To check it, pair a direction dz with the output change ds=a dz+b·conj(dz). Expanding Re(conj(g_s)ds) gives Re(conj(g_z)dz), the same real differential. This is the adjoint for the real inner product Re(conj(u)v), not an ordinary complex-linear transpose rule.

## Work one complete update

Choose a=1,b=1/2,c=1+i and z=1+i. In real coordinates,

$$
L(x,y)=(\tfrac32x-1)^2+(\tfrac12y-1)^2,\qquad
g=\tfrac32-\tfrac12 i.
$$

At η=1/4 the update z−ηg gives:

| Quantity | Before | After |
|---|---|---|
| z | 1+i | 5/8+9i/8 |
| r | (1−i)/2 | −1/16−7i/16 |
| L | 1/2 | 25/128 |

The minimizer is z*=2/3+2i; the target c is an output value, not the parameter minimizer. Omitting the conjugate path would give gradient 1−i instead of 3/2−i/2. The Hessian in (x,y) is diag(9/2,1/2), so convergence from every start for this update requires 0<η<4/9. This differs from the original identity-layer example’s 0<η<1.

## Reproduce real and imaginary derivative checks

The code tests three fixed layers and three parameter points against independent real/imaginary central differences. It also checks the exact reference update and minimizer. The lab above remains the original |z−c|² example; the extra layer is a separate runnable exercise.

```python
def loss(z, a, b, c):
    r = a*z + b*z.conjugate() - c
    return r.real*r.real + r.imag*r.imag

def gradient(z, a, b, c):
    r = a*z + b*z.conjugate() - c
    return 2*(a.conjugate()*r + b*r.conjugate())

cases = [(1+0j, 0.5+0j, 1+1j),
         (1+1j, 0.25-0.5j, 0.5-0.25j),
         (1+0j, 0j, 0j)]
h = 1e-5
for a, b, c in cases:
    for z in [1+1j, -0.5+0.25j, 0j]:
        dx = (loss(z+h, a, b, c) - loss(z-h, a, b, c))/(2*h)
        dy = (loss(z+1j*h, a, b, c) - loss(z-1j*h, a, b, c))/(2*h)
        assert abs(gradient(z, a, b, c) - complex(dx, dy)) < 1e-8

a, b, c = cases[0]
z = 1+1j
assert gradient(z, a, b, c) == 1.5-0.5j
updated = z - 0.25*gradient(z, a, b, c)
assert updated == 5/8 + 9j/8
assert abs(loss(updated, a, b, c) - 25/128) < 1e-14
assert abs(loss(2/3+2j, a, b, c)) < 1e-28
print(updated, loss(updated, a, b, c))
```

Framework connection: [PyTorch 2.15 Gradcheck mechanics](https://docs.pytorch.org/docs/2.15/notes/gradcheck.html) explicitly describes returning twice its conjugate Wirtinger derivative for real losses. Compare conventions with g=Lₓ+iLᵧ, rather than matching a derivative’s name alone. The code here uses analytic formulas and Python’s standard library; it does not reproduce framework training.

## Exercises and answers

1. η=1/2 **reaches c in one step**.
2. η=1 **preserves nonzero loss while oscillating**.
3. A real-valued learning objective **need not be holomorphic**.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
