## Differentiate an executed graph

Consider F(x,y)=sin(xy)+x². Its graph contains two inputs, a product a=xy, b=sin(a), another product c=x*x and the sum F=b+c. The same input x participates through more than one route. A correct derivative must include every route, including both arguments of the square multiplication.

The lab creates a primitive tape with values, parent indices and local partial derivatives. The graph labels switch among primal values, forward tangents and reverse adjoints. These labels come from actual propagation rather than substituting a closed-form gradient into each node. The analytic gradient is retained separately as a verification reference.

## Forward mode pushes a direction

Choose an input direction v=(vx,vy). Propagate its tangent through the chain rule:

$$
\dot a=yv_x+xv_y,\quad
\dot b=\cos(a)\dot a,\quad
\dot c=2xv_x,\quad
\dot F=\dot b+\dot c=Jv.
$$

One forward sweep computes the Jacobian action for this direction. For a general function with many outputs, it produces their directional changes together. Constructing a full Jacobian would require multiple independent directions, so the number of inputs and desired derivative products matters when selecting a mode.

## Reverse mode pulls a cotangent

Seed the scalar output adjoint with w. Traversing the tape backward adds local contributions to each parent. In particular, x receives the route through a and both parent entries of c=x*x:

$$
\bar x=w\left(y\cos(xy)+2x\right),\qquad
\bar y=w x\cos(xy).
$$

Overwriting a shared parent's adjoint would lose a path. The drawing collapses the two identical x-to-square edges for readability, while the table lists both local parent contributions and the reverse implementation accumulates both.

For a scalar output, one reverse sweep obtains all input partials multiplied by w. Retaining or recomputing forward states creates a memory tradeoff, developed in the checkpointing unit. AD applies derivative rules to the executed program; it is neither symbolic simplification of every expression nor subtraction of nearby function values.

## Pairing gives an independent consistency check

The scalar identity w(Jv)=(wJ)v compares forward and reverse products. At (x,y)=(1,0), the gradient is (2,1). With v=(1,1) and w=1 both sides equal three. Changing v to (0,1) changes JVP to one while leaving primal and VJP unchanged. Setting w=0 zeros every reverse adjoint while preserving the primal and forward tangent.

The lower plot compares F along the selected input direction with its first-order tangent. A tangent agrees locally, not necessarily over a finite step. Curvature explains their separation away from zero.

## Exercise: remove a path on paper

Pretend the square node sends only one x contribution. At (1,0) the resulting x derivative would be one rather than two, and JVP/VJP consistency against the correct tape would fail. List the contributions separately, then verify the true directional derivative with a moderate centered difference. Agreement near a smooth point provides evidence, while the next unit explains why finite-difference checks alone are insufficient at a kink.

Further reading: [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html). This tape and graph are implemented directly for the stated function.
