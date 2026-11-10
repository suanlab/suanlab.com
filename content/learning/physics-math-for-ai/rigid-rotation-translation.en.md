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

## Exercise and explained answer

Set both angles to zero and t=(1,2,−1). The reference point maps to (2,2,−1), and the two orders coincide because B is identity. Restore nonzero angles, change the y translation, and inspect how reversing order also rotates the translation. Finally verify that applying the computed inverse returns the reference point.

## Connection to embodied AI

Robot poses, camera extrinsics and spatial neural features require clear active/passive and frame conventions. A valid rotation matrix alone does not tell you whether a transform maps world to camera or camera to world. Label source and target frames, multiplication order and whether the object is a point or direction before composing predictions or comparing pose errors.

Further reading: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
