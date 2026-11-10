## Dimensionless does not mean physically determined

Dimensional analysis can rule out an equation that adds incompatible quantities. It can also reduce a physically specified relation to fewer dimensionless inputs. It cannot, by itself, choose the empirical function relating those inputs. Several distinct physical laws, or deliberately constructed mathematical laws, can satisfy the same dimensional requirements.

This lesson will distinguish three claims: a candidate monomial has no dimensions, a collection of groups is independent and complete for a declared variable list, and a proposed relationship between groups describes a physical experiment. The first two can be proved algebraically here. The third needs additional theoretical or observational evidence.

## Declare the variables and their meanings

Consider force magnitude F, fluid density rho, speed V, characteristic diameter D and dynamic viscosity mu. Assume they are positive when forming the displayed ratios. Their M,L,T exponent columns are

$$
\begin{array}{c|rrrrr}
 &F&\rho&V&D&\mu\\\hline
M&1&1&0&0&1\\
L&1&-3&1&1&-1\\
T&-2&0&-1&0&-1
\end{array}
$$

Dynamic viscosity has units kg/(m*s); it is not kinematic viscosity, whose units are square meters per second. Replacing one by the other without changing the variable list changes the dimensional problem. A characteristic diameter also needs a geometry definition; a shape or roughness parameter might be an additional relevant input.

We use force magnitude rather than an arbitrarily signed coordinate component. Positivity makes ratios with speed, density and diameter well defined and supports the logarithmic interpretation of monomials. Zero-speed limits require separate analysis rather than division by a zero reference.

## Turn dimensions into a linear map

Write a candidate monomial as F raised to eF times rho raised to erho, V raised to eV, D raised to eD and mu raised to emu. Multiplication adds dimension exponents, so its dimension vector is the matrix above times the exponent vector e.

A zero vector means that this monomial is dimensionless. It does not mean that its numerical value must be one, that it has no physical meaning, or that it is a complete description of an experiment. A Reynolds number and a normalized force can both be dimensionless while representing different quantities.

For example, F/(V squared*D squared) has dimensions of density. Its exponent vector is (1,0,-2,-2,0), and its defect is (1,-3,0). The missing density factor is visible algebraically. Dividing by rho removes that defect, but still does not tell us which values the resulting group should take in nature.

## Derive the complete monomial nullspace

The dimension matrix has rank three. One way to verify that is to select the density, speed and diameter columns: their three-by-three determinant is nonzero. There are five exponent coordinates, so the nullspace has dimension two.

Set a=eF and b=emu. The mass equation gives erho=-a-b. The time equation gives eV=-2a-b. The length equation then gives eD=-2a-b. Thus every dimensionless monomial has the form

$$
e=(a,-a-b,-2a-b,-2a-b,b).
$$

This argument is algebraic and works for real exponents on positive variables. The interface's bounded integer controls are a way to explore it, not the mathematical limit of the full nullspace. A successful finite scan would not replace this general derivation.

## Choose independent groups and explain completeness

Choose basis vectors with (a,b)=(1,0) and (0,1). Their groups are

$$
\Pi_F=\frac{F}{\rho V^2D^2},\qquad
\Pi_\mu=\frac{\mu}{\rho VD}=\frac{1}{\mathrm{Re}}.
$$

Every dimensionless monomial is PiF raised to a times Pimu raised to b. The two basis vectors are independent because the force exponent selects a and the viscosity exponent selects b. They span the entire monomial nullspace for the stated list.

Different choices are possible. Re can replace its reciprocal, and powers or invertible combinations can give another basis. However, merely listing two dimensionless quantities is insufficient. PiF and PiF squared are dependent and cannot capture changes in Pimu independently. Dimensionlessness, independence and completeness are separate checks.

Our PiF is a normalized force group, not automatically the conventional drag coefficient. Numerical constants depend on the area and one-half convention used in defining that coefficient. State those conventions before identifying the quantities.

## Explain what Buckingham reduction assumes

Suppose the force is a single-valued function of the other listed variables, the relation is dimensionally covariant, and the variable list includes the relevant dimensional information. It can then be expressed using groups in a form such as

$$
\Pi_F=\Phi(\mathrm{Re}).
$$

This reduces the inputs but leaves Phi unspecified. If geometry, compressibility, roughness or another dimensional scale matters, additional variables and groups may be required. Dimensional algebra cannot detect every omitted piece of physics from an incomplete list.

The theorem should therefore be applied after defining the physical question, rather than using a convenient matrix as evidence that the chosen variables must be sufficient. Experimental comparison or a derivation from governing equations supplies information that dimensions alone do not contain.

## Compare two constructed admissible laws

Consider two deliberately constructed functions:

$$
\Phi_1(R)=1/R,\qquad \Phi_2(R)=1+1/R,\qquad R>0.
$$

Both respect the dimensions. In physical variables the first gives F=mu*V*D, while the second gives F=rho*V squared*D squared+mu*V*D. These are examples of dimensional admissibility, not claimed empirical drag laws.

At Re=2 the normalized predictions are one half and three halves. A normalized-force query of one half agrees with the first and disagrees with the second. That comparison tests the chosen two hypotheses at a query; it does not establish that the first describes a real experiment or that no other admissible function exists.

The comparison also illustrates why a correct unit check cannot determine a missing coefficient or functional dependence. Physics and data still have work to do after the dimensional reduction is complete.

## Connect the groups to learning and experimental design

A learning model can use dimensionless groups as features to respect a known scaling structure. This may combine observations taken with different units or physical scales more coherently than treating every raw number as unrelated. The benefit depends on the physical variable list and modeling assumptions being appropriate.

It does not remove the need for data across relevant Reynolds numbers or additional shape parameters. A model trained on a narrow group range can still extrapolate poorly. A normalization also transforms measurement uncertainty: uncertain density or speed in a denominator changes the uncertainty of the normalized force. Dimensional consistency is not a statistical error analysis.

When reporting results, separate the algebraic proof of invariance, the completeness of the chosen physical variables, and the empirical accuracy of a learned function. Those claims require different evidence and can fail independently.

## Exercises with worked solutions

1. **Exercise:** What is the dimension defect of F/(V squared*D squared)? **Solution:** It is M*L to the minus three, the dimension of density. Dividing by rho removes it.
2. **Exercise:** Why are there two independent groups? **Solution:** The matrix has five columns and rank three, leaving nullity two; the displayed two independent basis vectors span it.
3. **Exercise:** Are PiF and PiF squared a complete independent pair? **Solution:** No. They encode only one independent exponent direction and omit the viscosity group.
4. **Exercise:** Write the monomial for a=2,b=-1. **Solution:** The exponents are (2,-1,-3,-3,-1), giving PiF squared divided by Pimu.
5. **Exercise:** At Re=2, which toy function matches normalized force 0.5? **Solution:** Phi1 does; Phi2 predicts 1.5. This is a hypothesis comparison, not empirical validation.
6. **Exercise:** Does dimensional covariance determine Phi? **Solution:** No. It restricts the representation while leaving the dimensionless function to physics or data.

## Sources and further study

The [MIT Buckingham lecture](https://ocw.mit.edu/courses/2-25-advanced-fluid-mechanics-fall-2013/c0a4521f55e9191d557c167e99e97469_MIT2_25F13_The_Buckingham.pdf) provides background on independent groups, physical variable selection and experimental comparison. The matrix, nullspace proof and two constructed functions above are direct derivations for this lesson. 
