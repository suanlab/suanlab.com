## Surface orientation matters even at the same point

Pressing down on a book and pushing along its side illustrate different surface interactions. A single number at a point cannot describe the force direction on every possible cut. Stress maps a surface normal to force per unit area, called traction. Specify the point and the oriented current surface before decomposing this vector. This unit investigates that map and its force balance; a stress field alone does not automatically determine a material’s displacement.

## Prerequisites, signs and physical units

Review matrix-vector multiplication, dot products, moments and momentum conservation. Both stress and traction have pressure units, whereas an integrated force has units of newtons. Use tensile normal stress as positive. An outward normal does not require traction to point outward; compression can point inward. In a two-dimensional drawing, the segment length becomes a face area only after including a specified depth. Keep forces, moments and force densities in separate channels.

$$
[\sigma]=[\mathbf t]=\mathrm{Pa}=\mathrm{N/m^2},\qquad [\mathbf F]=\mathrm N,\quad [M_z]=\mathrm{N\,m},\quad [\mathbf f]=\mathrm{N/m^3}.
$$

## From a normal to traction

For a unit normal n, the Cauchy traction is sigma times n. Feeding an area vector into the same linear map instead produces force, so identify the input convention. Opposite normals refer to opposite sides of a cut and reverse the traction. Our implementation uses a symmetric in-plane stress tensor with zero out-of-plane components. It is a plane-stress teaching family, not a complete representation of arbitrary three-dimensional stress.

$$
\mathbf t(\mathbf n)=\boldsymbol\sigma\mathbf n,\qquad \|\mathbf n\|=1,\qquad \mathbf t(-\mathbf n)=-\mathbf t(\mathbf n).
$$

## Normal stress and signed shear

Project traction onto the normal to obtain scalar normal stress. The remaining vector lies along the surface. In two dimensions choose the tangent obtained by rotating n counterclockwise through a right angle; its dot product with traction defines a signed shear. Reversing this tangent changes the scalar sign but not the physical shear vector. Report the convention when comparing implementations rather than comparing only unsigned magnitudes.

$$
\sigma_n=\mathbf n^T\boldsymbol\sigma\mathbf n,\qquad \tau=\mathbf m^T\boldsymbol\sigma\mathbf n,\qquad \mathbf t=\sigma_n\mathbf n+\tau\mathbf m.
$$

## Stress symmetry and its assumptions

In a classical Cauchy continuum without couple stresses or body couples, angular-momentum balance leads to symmetric stress. Do not impose that conclusion indiscriminately on generalized continuum models. We select a symmetric two-dimensional field and independently integrate central moments of its face forces. Agreement checks this family and its orientation conventions. It does not validate generalized couple stresses, arbitrary rotational microstructure or a complete dynamical model.

$$
\boldsymbol\sigma=\begin{pmatrix}\sigma_{xx}&\tau_{xy}\\\tau_{xy}&\sigma_{yy}\end{pmatrix}=\boldsymbol\sigma^T.
$$

## Worked example: force on a tilted face

Choose the in-plane tensor with entries three, one, one and minus one pascal and a normal at forty-five degrees. The traction points along the positive x-axis even though the normal does not. Its normal and signed-shear components are two and minus two pascals under our tangent convention. Multiplying by a two-square-metre reference patch gives the actual force. This example separates surface orientation from force orientation and from their scalar projections.

$$
\boldsymbol\sigma=\begin{pmatrix}3&1\\1&-1\end{pmatrix}\mathrm{Pa},\quad \mathbf t=(2\sqrt2,0)\mathrm{Pa},\quad \sigma_n=2\mathrm{Pa},\quad\tau=-2\mathrm{Pa},\quad\mathbf F=(4\sqrt2,0)\mathrm N.
$$

## In-plane principal stresses and nonunique directions

When traction is parallel to the normal, shear vanishes. For a symmetric matrix these directions are its eigenvector axes, with eigenvalues giving the in-plane principal stresses. Equal eigenvalues leave no preferred in-plane direction; an arbitrary zero angle would hide this nonuniqueness. A plane-stress model also has an out-of-plane zero principal value, so its two in-plane extrema need not be the full three-dimensional extrema. Axes are equivalent under reversal.

$$
\sigma_{\pm}=\frac{\sigma_{xx}+\sigma_{yy}}2\pm\sqrt{\left(\frac{\sigma_{xx}-\sigma_{yy}}2\right)^2+\tau_{xy}^2}.
$$

A small principal stress can disappear if it is computed by subtracting two nearly equal rounded numbers. For a diagonal tensor with entries 1 and 10⁻²⁰ Pa, the two eigenvalues are already those entries; averaging and then subtracting can falsely replace the smaller one by zero. The calculation keeps diagonal values directly. For a general symmetric tensor it scales the components before finding the larger-magnitude root, then obtains the other root from the unprojected determinant divided by that root. This retains small stresses without confusing a genuinely singular tensor with a weakly stressed direction. The root used in the division is also retained before its final display rounding.

## In-plane compression and three-dimensional hydrostatic stress

The in-plane tensor minus p times the two-dimensional identity gives inward traction and zero shear for every in-plane normal. It illustrates the directional pattern of pressure on those cuts. Our plane-stress completion sets the out-of-plane normal stress to zero, however. A fully hydrostatic three-dimensional state would also require that component to be minus p. Keep this distinction when interpreting mean stress or all three principal values.

$$
\boldsymbol\sigma_{2D}=-pI_2\quad\Longrightarrow\quad\mathbf t=-p\mathbf n,\quad\tau=0;\qquad \text{3D hydrostatic: }\boldsymbol\sigma=-pI_3.
$$

## A spatially varying stress family

Let stress vary affinely with current position. Separate normal gradients from a transverse shear gradient so their contributions to resultant force remain visible. These parameters have pressure-per-length units, not force units. The selected field has constant divergence, while local face tractions still vary with position and orientation. Uniform stress and static force balance are different concepts: a nonuniform field can balance an appropriate body-force density.

$$
\boldsymbol\sigma(x,y)=\begin{pmatrix}a+g_x x&b+ky\\b+ky&d+g_y y\end{pmatrix},\qquad\nabla\!\cdot\boldsymbol\sigma=(g_x+k,g_y).
$$

## Integrate the actual rectangular boundary

Consider a rectangular prism with specified centre, half-widths and depth. Left and right face areas differ from top and bottom face areas when the widths differ. Integrate traction at actual current positions with outward normals, retaining vector signs. Moments in this experiment are measured about the rectangle centre. A uniform body-force density has zero integrated moment about that centre, even when its resultant force is nonzero.

$$
V=4h_xh_yB,\qquad\mathbf F_s=\int_{\partial V}\boldsymbol\sigma\mathbf n\,dA,\qquad M_z=\int_{\partial V}\bigl((\mathbf x-\mathbf c)\times\boldsymbol\sigma\mathbf n\bigr)_z\,dA.
$$

## Worked force balance with a body force

Choose gradients gx equal to one, k equal to two and gy equal to minus one half pascal per metre. With half-widths one and one half metre and depth two metres, volume is four cubic metres. The surface-force resultant is twelve and minus two newtons. A declared body-force density of minus three and one half newtons per cubic metre balances it. Omitting that force leaves a static imbalance; without mass density we do not infer an acceleration.

$$
\mathbf F_s=V\nabla\!\cdot\boldsymbol\sigma=(12,-2)\mathrm N,\qquad\mathbf F_b=V\mathbf f=(-12,2)\mathrm N,\qquad\mathbf F_s+\mathbf F_b=0.
$$

## Signs relative to the preceding momentum flux

The previous gas experiment used a positive pressure contribution in outward momentum flux, whereas pressure traction carries a negative sign. These conventions are consistent: an outward transport term and a surface force appear on opposite sides of the momentum balance. Subtracting Cauchy stress from the advective momentum-flux tensor produces the familiar positive pressure term. Our static stress experiment does not solve a separate velocity field or claim to evaluate that full advective transport.

$$
\text{momentum flux}=\rho\mathbf u\otimes\mathbf u-\boldsymbol\sigma;\qquad\boldsymbol\sigma=-pI\ \Longrightarrow\ \rho\mathbf u\otimes\mathbf u+pI.
$$

## Rotation and the same physical state

Under a consistent active rotation Q, transform both stress and normal, and traction must rotate by the same Q. Normal stress, consistently oriented signed shear and principal values remain unchanged. Changed matrix components do not necessarily mean a different physical loading. Rotating only the tensor while holding the normal fixed generally selects another observation. State this transformation contract when testing an AI model instead of requiring every component to remain numerically identical.

$$
\boldsymbol\sigma^{\prime}=Q\boldsymbol\sigma Q^T,\qquad\mathbf n^{\prime}=Q\mathbf n,\qquad\mathbf t^{\prime}=Q\mathbf t.
$$

## Zero area, small geometry and unavailable arithmetic

A zero-area patch has zero integrated force, while a mathematical traction query at a supplied normal remains a distinct quantity. A collapsed rectangle has no ordinary closed-surface observation in this implementation. Positive half-widths below ten to the minus eight metres are flagged outside the supported geometry, without widening or replacing them by zero. Unrepresentable nonzero products are unavailable. This does not recover every cancellation or provide arbitrary-precision mechanics.

## Verification, constitutive information and AI limits

Begin with the tilted-face calculation and normal reversal, then compare in-plane compression and pure shear. Inspect the affine field’s surface-force and moment integrals against independent volume observations, and retain the omitted-body-force control. A residual assembled from shared formulas is not an independent certificate. Static equilibrium alone does not identify displacements or material constants; compatibility, boundary data and constitutive information are also needed. The next unit measures deformation through actual mapped segments.

## Research extension: stress identification from directional observations

### Choose the unknown and the measured component

Keep the same symmetric in-plane Cauchy stress at one point. Suppose each supplementary observation records only normal stress, rather than the complete traction vector. The three unknown parameters below have units pascals. Here p is the in-plane mean stress, not positive compressive fluid pressure, and tau is the fixed-coordinate xy component, not the signed shear on every tilted face. The Frobenius norm is proportional to the Euclidean norm of these parameters. This declares the comparison metric before discussing conditioning.

$$
\sigma=\begin{pmatrix}p+d&\tau\\\tau&p-d\end{pmatrix},\qquad s=(p,d,\tau)^T,\qquad\|\sigma\|_F^2=2\|s\|_2^2.
$$

### Derive one scalar observation

Insert a unit normal into the quadratic normal projection. Double-angle identities give the displayed row of the forward map. Opposite normals represent the same scalar measurement, even though their full traction vectors reverse. The data refer to the same point and unchanged physical state; measurements after changing the material loading would be another inverse problem. We do not claim that cutting real samples repeatedly leaves their stresses unchanged or that these equations model an actual measuring instrument.

$$
n(\theta)=(\cos\theta,\sin\theta)^T:\qquad m(\theta)=n^T\sigma n=p+d\cos2\theta+\tau\sin2\theta,\quad m(\theta+\pi)=m(\theta).
$$

### Invert information rather than count readings

Stack the known directions into A. With independent equal-variance errors the least-squares estimate has the form below whenever all three parameter directions are observable. Three distinct axes modulo pi give three distinct points on the double-angle unit circle; a line meets a circle in at most two points, so those three rows are independent. Repeating the same axis does not create the missing directions. Extra repeated observations can reduce noise in an observed component but cannot restore rank by themselves.

$$
m=As+e,\quad A_{j,:}=(1,\cos2\theta_j,\sin2\theta_j),\qquad\widehat s=(A^TA)^{-1}A^Tm\quad(\operatorname{rank}A=3).
$$

### Full traction is a different observation

One complete traction vector supplies two components, but a symmetric two-dimensional stress still has one invisible direction. Add a multiple of the tangential dyad below and that traction does not change. Two nonparallel full-traction observations can determine the map, subject to the symmetry consistency relation between their cross projections. These facts do not contradict the three-scalar result, because the recorded data differ. A regularizer in an unobserved direction selects a stress; it does not demonstrate that the measurements identified it.

$$
t=\sigma n,\quad n^Tz=0,\quad\|n\|=\|z\|=1:\qquad (\sigma+\beta zz^T)n=t\quad\text{for every real }\beta.
$$

### Separate uniqueness from noise stability

Subtract the true forward relation from the least-squares formula. The pseudoinverse norm gives the deterministic error bound, while independent zero-mean equal-variance errors give the covariance. A full-rank A can have a small minimum singular value and large noise amplification. The bound assumes calibrated directions and additive stress errors; it omits orientation uncertainty, instrument response and model discrepancy. If errors are correlated, the covariance and an appropriate weighted estimator must be derived for that declared model instead.

$$
\mathbb E e=0,\quad\operatorname{Cov}(e)=\nu^2I:\quad\operatorname{Cov}(\widehat s)=\nu^2(A^TA)^{-1},\quad\|\widehat s-s\|_2\le\frac{\|e\|_2}{\sigma_{\min}(A)},\quad[\nu]=\mathrm{Pa}.
$$

### Spread directions to balance the information

For the three axes zero, sixty and one hundred twenty degrees, summing the double-angle columns makes the Gram matrix diagonal. The mean and the two deviatoric coordinates are uncorrelated in the resulting estimate under the stated noise model. This symmetric design attains condition number square root of two. The value is not one because each observation contains a full mean column while its two angular components share a fixed total squared length. Condition number alone is also not the size of an absolute stress uncertainty.

$$
(\theta_1,\theta_2,\theta_3)=(0,\pi/3,2\pi/3):\quad A^TA=\operatorname{diag}(3,3/2,3/2),\qquad\kappa_2(A)=\sqrt2.
$$

### Prove what is optimal under the declared metric

For any three rows, the mean column has squared norm three, so the largest Gram eigenvalue is at least three. Restricting to the other two parameter coordinates gives a two-dimensional block with trace three; its smaller Rayleigh value is at most three halves. The smallest eigenvalue of the full matrix is no larger. This proves the displayed lower bound, attained by the preceding axes. It concerns three scalar observations with this variable scaling, not every sensor cost, noise correlation or choice of experimental objective.

$$
\lambda_{\max}(A^TA)\ge3,\qquad\lambda_{\min}(A^TA)\le3/2\quad\Longrightarrow\quad\kappa_2(A)\ge\sqrt2\quad\text{for three full-rank scalar observations}.
$$

### Nearly repeated directions retain rank but lose precision

Choose axes zero, delta and twice delta. Subtracting the first row from the others and using trigonometric identities gives the determinant below. It remains positive for the declared interval, yet vanishes cubically as delta shrinks. Exact delta zero is a different rank-deficient input. A small nonzero determinant is not itself a complete condition-number calculation, and a rounded angle or determinant must not silently become an exact repeated axis. The next formula resolves the singular values directly.

$$
(\theta_1,\theta_2,\theta_3)=(0,\delta,2\delta),\quad0<\delta<\pi/2:\quad\det A=4\sin^2\delta\sin2\delta>0,\qquad\det A\sim8\delta^3\quad(\delta\to0).
$$

### Evaluate the small eigenvalue without cancellation

Rotate the deviatoric parameter coordinates to centre the three axes. The Gram matrix splits into a two-by-two block and a scalar. The block has trace T and determinant D; obtain its smaller eigenvalue from the product, rather than subtracting two close large numbers. Combine it with the third eigenvalue to find the minimum singular value. The table reports determinant, minimum singular value, condition number and worst directional parameter variance divided by nu squared. It is a finite numerical comparison, not a universal floating-point error bound near rank loss.

$$
c=\cos2\delta,\quad T=4+2c^2,\quad D=8\sin^4\delta:\quad\lambda_+=\tfrac12(T+\sqrt{T^2-4D}),\quad\lambda_-=D/\lambda_+,\quad\lambda_0=2\sin^22\delta,\quad\sigma_{\min}(A)=\sqrt{\min(\lambda_-,\lambda_0)}.
$$

### Work the familiar three-axis example

For zero, forty-five and ninety degrees the inverse can be written without a matrix solve. The original tensor [[3,1],[1,−1]] Pa produces normal observations (3,2,−1) Pa, yielding p=1, d=2 and tau=1 Pa. Independent measurement errors need not give independent recovered parameters: the displayed covariance includes a nonzero p–tau entry. An isotropic stress has no preferred principal axis even if its components are measured accurately; that physical axis nonuniqueness is separate from a rank-deficient observation matrix. The native traction and affine-force lab remains unchanged.

$$
\theta=(0,\pi/4,\pi/2):\quad p=\tfrac12(m_1+m_3),\quad d=\tfrac12(m_1-m_3),\quad\tau=m_2-p,\quad\operatorname{Cov}(\widehat s)/\nu^2=\begin{pmatrix}1/2&0&-1/2\\0&1/2&0\\-1/2&0&3/2\end{pmatrix}.
$$

| delta (degrees) | determinant A | minimum singular value | condition number | worst variance / nu squared |
| --- | --- | --- | --- | --- |
| 5.00000000 | 0.00527621 | 0.00881572 | 276.45298154 | 12867.20641698 |
| 15.00000000 | 0.13397460 | 0.08083776 | 28.99405294 | 153.02820107 |
| 30.00000000 | 0.86602540 | 0.33763747 | 6.20274201 | 8.77200187 |
| 60.00000000 | 2.59807621 | 1.22474487 | 1.41421356 | 0.66666667 |

```python
from math import sin, cos, pi, sqrt
for degrees in [5,15,30,60]:
    delta = degrees*pi/180
    c = cos(2*delta)
    trace = 4+2*c*c
    determinant = 8*sin(delta)**4
    large = (trace+sqrt(trace*trace-4*determinant))/2
    small = determinant/large
    third = 2*sin(2*delta)**2
    minimum, maximum = min(small,third), max(large,third)
    det_A = 4*sin(delta)**2*sin(2*delta)
    values = [float(degrees),det_A,sqrt(minimum),sqrt(maximum/minimum),1/minimum]
    print(" ".join(f"{x:.8f}" for x in values))
```

[Alberta’s Cauchy stress notes](https://engcourses-uofa.ca/stress/cauchy-stress-as-a-linear-map/) supply the traction-map and symmetry background. The scalar observation model, null direction, noise calculation and conditional design bound are derived explicitly here; they do not certify an instrument or determine a constitutive response.

![Directional stress measurements, clustered-axis conditioning and different observation scopes](/assets/learning/continuum-stress-directional-identification.en.svg)

### Research questions with explanations

**Do repeated measurements on one axis identify every stress component?** **Answer:** No. Repetition can reduce observed noise while leaving the same unobserved parameter directions.

**Is a small nonzero determinant the same as an unidentifiable tensor?** **Answer:** No. The clustered design can remain invertible but amplify noise strongly. Exact rank loss and poor conditioning are different.

**Does component identification always select a unique principal axis?** **Answer:** No. An isotropic in-plane stress has no preferred axis even when the measurement design identifies its components.

## Exercises with answers

### Exercise 1

Find traction, normal stress and signed shear in the tilted-face worked example.

**Answer:** They are (2 sqrt(2), 0) pascals, 2 pascals and −2 pascals. The two-square-metre patch carries (4 sqrt(2), 0) newtons.

### Exercise 2

Reverse n at the same stress. What changes?

**Answer:** Traction and the shear vector reverse. Normal stress is unchanged. Signed shear is unchanged when the tangent is reversed consistently too.

### Exercise 3

Is the plane-stress example sigma2D=−2I2 fully hydrostatic in three dimensions?

**Answer:** No. Its out-of-plane stress is zero. Fully hydrostatic compression would also require sigma_zz=−2 pascals.

### Exercise 4

With volume four cubic metres and stress divergence (3, −0.5) newtons per cubic metre, find surface force and a balancing body-force density.

**Answer:** Surface force is (12, −2) newtons; the balancing body-force density is (−3, 0.5) newtons per cubic metre.

### Exercise 5

Does zero resultant force under uniform stress mean every face force vanishes?

**Answer:** No. Nonzero forces on different faces can cancel. The sum of magnitudes differs from the vector resultant.

### Exercise 6

Do stress and static equilibrium uniquely determine actual displacement?

**Answer:** No. Compatibility, boundary conditions and constitutive information are also required.

## Sources and next connections

For stress and equilibrium background, consult MIT’s [Elements of Continuum Elasticity](https://www.ocw.mit.edu/courses/2-002-mechanics-and-materials-ii-spring-2004/f7b561fbe8762ff73da9236e6c0fff73_lec7.pdf) and Alberta’s [Cauchy Stress Tensor](https://engcourses-uofa.ca/stress/cauchy-stress-as-a-linear-map/). For strain definitions and rotations, see MIT’s [The Concept of Strain](https://ocw.mit.edu/courses/2-080j-structural-mechanics-fall-2013/71034688fba0ff26d25653a4bd236a4a_MIT2_080JF13_Lecture2.pdf) and Alberta’s [Strain Measures](https://engcourses-uofa.ca/displacement-and-strain/strain-measures/). The affine families, worked observations and arithmetic comparisons here are textbook examples, not certification of general material simulations.
