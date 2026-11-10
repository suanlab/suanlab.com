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
