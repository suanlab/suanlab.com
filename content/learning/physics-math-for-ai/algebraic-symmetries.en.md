## Symmetries are composable transformations

Rotations and reflections preserving a square form a group under composition. We write D4 for the square symmetry group with eight elements. A quarter-turn R and reflection S in the horizontal axis generate every element. Some texts call this group D8, so check the order as well as the name. The lab acts exactly on integer-coordinate points.

$$
R(x,y)=(-y,x),\qquad S(x,y)=(x,-y),\quad R^4=S^2=I.
$$

## Apply the rightmost transformation first

In R to the r times S to the f, apply S first and then R. These operations generally do not commute. Acting on (1,0), SR gives (0,−1), while RS gives (0,1). Treating these as identical would also corrupt an equivariance condition. The controls select r from zero to three and f from zero or one.

$$
SR=R^{-1}S.
$$

## Orbits and stabilizers

The orbit contains every position reached under the group. The stabilizer is the subgroup fixing the original point. A corner (1,1) has four orbit points and two stabilizing transformations. The axis point (1,0) also has orbit size four and stabilizer size two. The center is fixed by all eight transformations. Arrows show how the selected group element permutes the orbit, rather than a time trajectory.

$$
|\operatorname{Orb}(v)|\,|\operatorname{Stab}(v)|=|D_4|=8.
$$

## Action laws and learning

An action requires the identity to fix points and group multiplication to agree with successive application. This is stronger than providing an arbitrary list of transforms. For rotation augmentation or symmetry-aware layers, specify actions on both inputs and outputs. This integer-coordinate example does not simulate pixel interpolation, boundary handling or continuous-angle errors. Whether a real task possesses the proposed symmetry is a separate modeling question.

$$
e\cdot v=v,\qquad (gh)\cdot v=g\cdot(h\cdot v).
$$

## Exercises with explanations

1. What is the center orbit size? **One.**
2. How many symmetries fix a corner? **Two: the identity and reflection in its diagonal.**
3. Do the arrows measure velocity? **No: they encode a selected group action.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).
