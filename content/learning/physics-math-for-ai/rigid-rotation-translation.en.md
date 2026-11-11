## A rigid pose combines orientation and position

SO(3) is the group of three-dimensional proper rotations: RᵀR=I and detR=1. Orthogonality preserves lengths and angles; determinant one excludes reflections. The set is a curved three-dimensional manifold inside nine matrix entries, so treating all entries as independent unconstrained parameters would lose the rotation conditions.

SE(3) adds translation. A pose acts on a point p by Rp+t and can be stored as a homogeneous matrix:

$$
T=\begin{pmatrix}R&t\\0&1\end{pmatrix},\qquad
T\begin{pmatrix}p\\1\end{pmatrix}
=\begin{pmatrix}Rp+t\\1\end{pmatrix}.
$$

A direction vector instead has homogeneous last coordinate zero and is unaffected by translation. The diagram's reference and transformed vectors begin at their respective frame origins; translation moves the orange origin as well as the endpoints.

## Derive composition rather than adding translations blindly

Apply T₂ and then T₁:

$$
R_1(R_2p+t_2)+t_1
=(R_1R_2)p+(R_1t_2+t_1).
$$

Thus (R₁,t₁)(R₂,t₂)=(R₁R₂,R₁t₂+t₁). The rightmost transform acts first. The second translation must be expressed through the first rotation; t₁+t₂ alone generally mixes frames incorrectly. The inverse is (Rᵀ,−Rᵀt), obtained by solving p'=Rp+t for p.

The lab uses A=(Rz(α),t) and B=(Rx(β),0). These axis choices make order effects easy to inspect; the next unit permits arbitrary rotation-vector generators. Matrix products and point actions here remain fully three dimensional.

## Work the quarter-turn example

Set α=β=π/2, t=(1,0,0), p=(1,0,0). For AB, Rx leaves p unchanged, Rz sends it to (0,1,0), and translation gives (1,1,0). For BA, A first gives (1,1,0), then Rx gives (1,0,1). Their discrepancy is √2. No change of numerical precision can make those different compositions equal.

The lower plot shows xy and xz projections of the selected transformed point, with different meanings for the second coordinate. Read full xyz, matrix rows, determinant and orthogonality error in the table. The upper drawing is an oblique projection and does not preserve visible angles or lengths.

## Change a twist's frame with the adjoint

SE(3) is a semidirect product: rotations act on translations before they are added. The same coupling appears when changing a velocity generator's frame. Conjugate its homogeneous matrix by T=(R,t):

$$
T\begin{pmatrix}\widehat\omega&v\\0&0\end{pmatrix}T^{-1}
=\begin{pmatrix}\widehat{R\omega}&Rv+t\times R\omega\\0&0\end{pmatrix}.
$$

With angular coordinates listed first, this is the adjoint map Adₜ(ω,v)=(Rω,t×Rω+Rv). Its translation term is a lever-arm velocity. Rotating both three-vectors without that term would be incorrect when the frame origins differ. For a differentiable pose, spatial and body twists satisfy Vₛ=AdₜVᵦ when their matrix definitions are ṪT⁻¹ and T⁻¹Ṫ respectively.

For R=I,t=(1,2,0),ω=(0,0,1),v=0, conjugation gives linear component (2,−1,0). This is not a translation of the original point cloud; it is the same infinitesimal rigid motion described relative to another origin. The exact homogeneous-matrix code checks the sign and multiplication order.

| Origin offset t | Angular component | Resulting linear component |
|---|---|---|
| (0,0,0) | (0,0,1) | (0,0,0) |
| (1,0,0) | (0,0,1) | (0,-1,0) |
| (1,2,0) | (0,0,1) | (2,-1,0) |

The corresponding Lie bracket is (ω₁×ω₂,ω₁×v₂−ω₂×v₁). Both this bracket and the adjoint are frame-aware tools for deriving pose updates; they do not authorize adding rotation vectors as if finite rotations commuted.

## Reproduce the calculation in Python

```python
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
Xi=[[0,-1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]
for t in [(0,0,0),(1,0,0),(1,2,0)]:
    T=[[int(i==j) for j in range(4)] for i in range(4)]
    inv=[r[:] for r in T]
    for i in range(3): T[i][3]=t[i]; inv[i][3]=-t[i]
    out=mm(mm(T,Xi),inv)
    linear=[out[i][3] for i in range(3)]
    assert linear == [t[1],-t[0],0]
    print(t,linear)
```

## Exercise and explained answer

Set both angles to zero and t=(1,2,−1). The reference point maps to (2,2,−1), and the two orders coincide because B is identity. Restore nonzero angles, change the y translation, and inspect how reversing order also rotates the translation. Finally verify that applying the computed inverse returns the reference point.

## Connection to embodied AI

Robot poses, camera extrinsics and spatial neural features require clear active/passive and frame conventions. A valid rotation matrix alone does not tell you whether a transform maps world to camera or camera to world. Label source and target frames, multiplication order and whether the object is a point or direction before composing predictions or comparing pose errors.

Further reading: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
