## Turning a body differs from stretching it

Rotating a square sheet on a table can move its corners substantially while preserving side lengths and angles. Reading displacement itself as strain would mistake rigid motion for material stretching. We map actual points and segments, observe their lengths and area, and compare those observations with infinitesimal and finite strain measures. The experiment prescribes a deformation family; it does not solve a material response from stress or elastic constants.

## Separate reference and current positions

Write X for reference position and x for the mapped current position. Displacement is their difference and has length units. Its reference-coordinate gradient H is dimensionless, while the deformation gradient F equals identity plus H. Some sources use another symbol convention, including F for a displacement gradient. Verify definitions before transferring a formula. Throughout this experiment H and F denote different objects and retain these meanings.

$$
\mathbf x=\boldsymbol\chi(\mathbf X),\qquad\mathbf u=\boldsymbol\chi-\mathbf X,\qquad H=\nabla_X\mathbf u,\qquad F=\nabla_X\boldsymbol\chi=I+H.
$$

## Rotation, shear and stretch parameters

Use positive diagonal parameters a and b, a shear parameter g and a spatial rotation. When g vanishes, the diagonals give stretches along the reference axes. With nonzero shear, however, a reference y-vector has length ratio square root of g squared plus b squared. The triangular factor is not generally a symmetric stretch tensor, and R is not generally the polar-decomposition rotation. Positive diagonals keep this selected family orientation preserving and nonsingular.

$$
F=R(\theta)B,\qquad B=\begin{pmatrix}a&g\\0&b\end{pmatrix},\qquad a,b>0,\qquad\mathbf x=F\mathbf X+\mathbf c.
$$

## Observe length from two actual points

The difference between two mapped points is F times their reference separation. A common translation cancels. For a unit reference direction, the exact stretch ratio is the norm of its mapped vector, and engineering strain subtracts one. Measuring this distance from actual point coordinates provides a geometric check distinct from substituting into a strain-tensor formula. Rounded coordinates can lose small differences, so distinguish the observation from a stable family calculation.

$$
\Delta\mathbf x=F\Delta\mathbf X,\qquad\lambda(\mathbf n)=\|F\mathbf n\|,\qquad e_{\rm eng}(\mathbf n)=\lambda(\mathbf n)-1.
$$

## Green strain measures a change in inner products

Comparing squared lengths naturally introduces F transpose times F. Half its difference from identity defines Green–Lagrange strain in reference coordinates. The directional contraction n transpose E n is half a squared-length change, not engineering strain itself. Consequently two different strain measures can correctly report different numbers for the same deformation. Specify the measure before judging agreement, especially when matching data or training targets across simulation tools.

$$
E=\tfrac12(F^TF-I),\qquad\|F\mathbf n\|^2-1=2\mathbf n^TE\mathbf n.
$$

## The term omitted by infinitesimal strain

Insert identity plus H into the Green-strain definition. The result contains the symmetric part of H plus a quadratic term. Infinitesimal strain keeps only the first part. The approximation concerns displacement gradients and rotations, not merely the absolute displacement magnitude. A large rigid translation changes no lengths, while a substantial rotation can defeat this linearization. Also distinguish tensor shear components from engineering shear conventions when reading off-diagonal entries.

$$
\varepsilon=\tfrac12(H+H^T),\qquad E=\varepsilon+\tfrac12H^TH.
$$

## Stable Green components for this family

Exact spatial rotation cancels from F transpose F, allowing a direct expression through the triangular factor. Factoring a squared minus one helps retain stretches close to unity. A nonzero shear-square term below representation range remains unavailable. We keep this family evaluation separate from Green strain computed using the rounded displayed F. Its exact construction does not certify arbitrary numerical matrices or general deformation fields.

$$
E=\frac12\begin{pmatrix}(a-1)(a+1)&ag\\ag&(b-1)(b+1)+g^2\end{pmatrix}.
$$

## Worked example: apparent compression under pure rotation

Choose unit diagonal parameters, zero shear and a sixty-degree rotation. Actual side lengths and area remain unchanged, and Green strain is zero. Infinitesimal strain nevertheless has diagonal entries minus one half. Reading those entries as fifty-percent physical compression would be incorrect: the finite rotation lies outside the relevant linearization. Compare the actual rotated geometry before interpreting the two strain readouts or using either as a learning target.

$$
F=R(\theta):\quad E=0,\quad\varepsilon=(\cos\theta-1)I;\qquad\theta=\pi/3:\ \varepsilon=-\tfrac12I.
$$

## A small but nonzero rotation correction

At very small angles, the floating-point cosine can round to one, erasing its difference from one. The sine-squared identity retains a representable second-order correction. If halving the angle or squaring its sine falls below representation range, that observation becomes unavailable. An exact zero Green strain of the rigid-rotation family and a tiny nonzero infinitesimal-strain correction express different facts; collapsing both to numerical zero would conceal the approximation.

$$
\cos\theta-1=-2\sin^2(\theta/2).
$$

## Worked example: a missing component in simple shear

For unit diagonals, zero rotation and shear 0.4, infinitesimal strain has off-diagonal entries 0.2 and zero diagonal entries. Green strain additionally has a lower-right entry 0.08. A reference y-segment maps to the vector 0.4, one, so its length ratio is square root of 1.16. The triangular parameter b remains one, but the actual y-directed segment length does not. This makes the omitted quadratic contribution observable.

$$
F=\begin{pmatrix}1&0.4\\0&1\end{pmatrix},\quad\varepsilon=\begin{pmatrix}0&0.2\\0.2&0\end{pmatrix},\quad E=\begin{pmatrix}0&0.2\\0.2&0.08\end{pmatrix},\quad e_{\rm eng}(\mathbf e_y)=\sqrt{1.16}-1.
$$

## A skew increment is not a finite rotation

Consider identity plus omega times the planar skew generator. Its symmetric first-order strain vanishes, but the map is not orthogonal at finite omega. It stretches every direction by square root of one plus omega squared and has positive Green strain. At omega one half, the Green diagonal is one eighth. Compare this actual affine map with an exact rotation through angle omega; they are different transformations, not two evaluations of the same physical motion.

$$
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad F_{\rm control}=I+\omega J,\quad E_{\rm control}=\tfrac12\omega^2I,\quad\lambda_{\rm control}=\sqrt{1+\omega^2}.
$$

## Compare determinant and actual polygon area

The reference square has area four. Under this two-dimensional map its area becomes four times a times b; shear and spatial rotation do not change that determinant. Independently compare an area computed from the mapped corners. Connecting area ratio to volume ratio would require a specified depth convention. The drawing does not determine arbitrary three-dimensional compressibility, and zero or negative determinants are excluded from the selected family.

$$
\det F=ab>0,\qquad A_{\rm current}=4ab,\qquad A_{\rm ref}=4.
$$

## A superposed rigid rotation and objectivity

Superpose an orthogonal spatial rotation Q, giving deformation gradient QF. Reference Green strain and actual lengths are unchanged. Spatial vector and Cauchy-stress components instead transform with Q and Q sigma Q transpose. Invariance of a reference tensor does not require every displayed component to stay numerically identical. Nor does knowing E alone determine the current Cauchy stress without material and orientation information from a constitutive model.

$$
E^{\prime}=\tfrac12((QF)^T(QF)-I)=E,\qquad Q^TQ=I.
$$

## Small length changes and two numerical evaluations

Subtracting one from a rounded length ratio can erase a small extension. Given the directional Green value q, rationalizing the square-root difference can retain a representable engineering strain. Keep this family value alongside results from rounded coordinates and matrices. It does not recover all nearly cancelling directional information or distinctions already absent from input data. Report representation limits and distinguish unavailable observations from exact physical zeros.

$$
q=\mathbf n^TE\mathbf n,\qquad e_{\rm eng}=\frac{2q}{\sqrt{1+2q}+1}.
$$

An unavailable component need not make every directional observation unavailable. With unit diagonal parameters and an extremely small nonzero shear, E22 can lie below the display range while the cross term still produces a representable extension along a diagonal direction. We therefore contract the complete polynomial before rounding its components. Likewise, when b=1.5, an unrepresentably small shear square must not erase the finite E22=0.625 contribution from the transverse stretch. The displayed tensor and the independently evaluated direction can consequently have different availability states; this is not a change of strain definition.

## Experiment sequence and AI validation

Begin with translation and exact rotation, checking actual lengths and area. Increase the angle to expose the difference between infinitesimal and Green strain, then examine simple shear and axial extension. Compare the skew-increment control and tiny-angle corrections. For an AI-predicted displacement field, also check the chosen strain definition, mapped-point distances, determinant, boundary conditions and material relations. A small shared-formula residual does not certify an entire deformation field or its physical stress.

## Exercises with answers

### Exercise 1

For an exact sixty-degree rotation, find diagonal Green and infinitesimal strain components.

**Answer:** Green strain is zero and the infinitesimal diagonal is −0.5. Actual lengths are preserved, so this is not physical fifty-percent compression.

### Exercise 2

For a=b=1 and g=0.4, find E22 and engineering strain along the reference y direction.

**Answer:** E22 is 0.08; engineering strain is sqrt(1.16)−1, approximately 0.07703. Parameter b is not the actual y-directed stretch in this sheared map.

### Exercise 3

For an axial stretch of 1.2, are engineering and Green strain equal?

**Answer:** No. Engineering strain is 0.2; Green strain is (1.2 squared minus one)/2, or 0.22.

### Exercise 4

Is I+0.5J an exact rigid rotation?

**Answer:** No. Its stretch ratio is sqrt(1.25) and Green diagonal is 0.125. Zero symmetric first-order strain does not make it an exact finite rotation.

### Exercise 5

Superpose a spatial rotation Q. What happens to reference E and area?

**Answer:** Both remain unchanged. Spatial vectors and stresses still transform according to their respective rotation laws.

### Exercise 6

If cos(theta) rounds to one, is the infinitesimal rotation correction physically zero?

**Answer:** Not necessarily. The sine-squared identity can retain a representable nonzero correction; an unrepresentable one remains unavailable.

## Sources and next connections

For stress and equilibrium background, consult MIT’s [Elements of Continuum Elasticity](https://www.ocw.mit.edu/courses/2-002-mechanics-and-materials-ii-spring-2004/f7b561fbe8762ff73da9236e6c0fff73_lec7.pdf) and Alberta’s [Cauchy Stress Tensor](https://engcourses-uofa.ca/stress/cauchy-stress-as-a-linear-map/). For strain definitions and rotations, see MIT’s [The Concept of Strain](https://ocw.mit.edu/courses/2-080j-structural-mechanics-fall-2013/71034688fba0ff26d25653a4bd236a4a_MIT2_080JF13_Lecture2.pdf) and Alberta’s [Strain Measures](https://engcourses-uofa.ca/displacement-and-strain/strain-measures/). The affine families, worked observations and arithmetic comparisons here are textbook examples, not certification of general material simulations.
