## 1. Keep the tangent parallel

A geodesic has a tangent vector that is parallel transported along its own path. For an affine parameter $\lambda$ this condition reads
$$
\frac{d^2 x^i}{d\lambda^2}
 +\Gamma^i_{jk}\frac{dx^j}{d\lambda}\frac{dx^k}{d\lambda}=0.
$$
The second term compensates for changing coordinate bases. Arc length is an affine parameter for a regular Riemannian geodesic. Proper time is affine for timelike freely falling motion. A null geodesic needs another affine parameter because its proper time is zero. [Tong, Geodesics in Spacetime §1](https://davidtong.org/teaching/general-relativity/grhtml/S1)

## 2. Follow a straight line in polar coordinates

On the plane with $ds^2=dr^2+r^2d\theta^2$, the equations are
$$
\ddot r-r\dot\theta^2=0,\qquad
\ddot\theta+2\dot r\,\dot\theta/r=0.
$$
The second equation gives constant $r^2\dot\theta$. For the Cartesian line $(x,y)=(1,\lambda)$, the polar coordinates are $r=\sqrt{1+\lambda^2}$ and $\theta=\arctan\lambda$. They both change nonlinearly although the path is straight. At $\lambda=1$, $\ddot r=1/(2\sqrt2)$ and $r\dot\theta^2=1/(2\sqrt2)$. They cancel in the geodesic equation.

## 3. Derive a great circle from the sphere constraint

Let $P$ and $Q$ be unit vectors that specify points on a sphere of radius $R$. For distinct points that are not antipodal, set $\vartheta=\arccos(P\cdot Q)$ and $T=(Q-\cos\vartheta\,P)/\sin\vartheta$. The tangent $T$ has unit norm and is orthogonal to $P$. The curve
$$
X(s)=R\big(\cos(s/R)P+\sin(s/R)T\big)
$$
has unit speed. Its acceleration points toward the center, so its tangential acceleration is zero. It is a geodesic and reaches $RQ$ at $s=R\vartheta$.

## 4. Distinguish geodesic motion from global minimization

The shorter arc has length $R\vartheta$. The arc in the opposite direction has length $R(2\pi-\vartheta)$ and also obeys the geodesic equation. It continues past the distance at which the shorter route becomes available. At antipodal endpoints the short and long lengths are both $\pi R$, and infinitely many great semicircles join the endpoints. The formula for $T$ requires an initial direction when the endpoint alone cannot select one.

## 5. Verify the path and its length

```python
from math import cos, isclose, pi, sin, sqrt

R = 2
theta = 0.9
P = (1, 0)
Q = (cos(theta), sin(theta))

def position(angle):
    return R*cos(angle), R*sin(angle)

short, long = R*theta, R*(2*pi-theta)
for angle in [theta, theta-2*pi]:
    assert all(isclose(a,R*b,abs_tol=1e-12) for a,b in zip(position(angle),Q))
    assert isclose(sqrt(sum(v*v for v in position(angle))), R)
assert short < long
assert isclose(short+long, 2*pi*R)
print(short,long)
```

## 6. Inspect a geodesic result

- Check speed and tangential acceleration as well as the endpoint.
- Compare polar and Cartesian coordinates of the same straight line.
- Keep the two sphere endpoints fixed and compare the short and long arcs. Explain which property they share and which quantity differs.
- Move the endpoints toward antipodal separation and look for the loss of a unique initial direction.
