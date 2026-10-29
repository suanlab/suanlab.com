## An expression includes its domain

Simplifying a formula can preserve values on allowed inputs without preserving an unrestricted domain. This is essential when implementing mathematical identities: an optimization that cancels a denominator may accidentally assign outputs at inputs where the original function was undefined.

A polynomial is a finite sum of nonnegative integer powers with coefficients. On real inputs it is defined everywhere. A rational function is a quotient of polynomials, defined where its denominator is nonzero. Its definition includes that exclusion.

## A removable point worked exactly

Factor the numerator:

$$
f(x)=\frac{x^2-1}{x-1}=\frac{(x-1)(x+1)}{x-1}=x+1,\qquad x\ne1.
$$

The cancellation uses x−1≠0. At x=0 the value is 1; at x=2 the value is 3. At x=1 the original expression is 0/0 and has no value. The polynomial g(x)=x+1 is defined there, so f and g are not identical functions with their full domains.

The graph of f is the line with point (1,2) removed. A separately defined extension assigning f(1)=2 makes a new function. It can be a useful extension, but the change must be explicit.

## A limit is a statement about nearby allowed inputs

As x approaches one through allowed inputs, x+1 approaches two. Thus the limit exists and equals two, despite the missing value. No argument evaluates the original denominator at zero to obtain that limit.

Continuity at a point requires the function value to exist and match the limit. A hole can therefore prevent continuity while leaving the nearby limit well behaved. Formal epsilon-delta conditions are developed in the later analysis material; this example separates the underlying objects first.

## Contrast a nonremovable pole

For h(x)=1/(x−c), approaching c from the right produces unbounded positive values and from the left unbounded negative values. At c itself the function is undefined. Infinity is not supplied as an ordinary real value there.

At c=0, h(0.1)=10 and h(−0.1)=−10. No single finite two-sided limit connects them. A line drawn from one branch across the pole would depict values the function does not have.

A nonzero numerator over a vanishing denominator differs from the cancellable zero-over-zero example. The symbol “division by zero” identifies an invalid substitution, not the entire limiting behavior; analyze the actual numerator, denominator and approach.

## Read the laboratory sampling honestly

The two branches are separate paths. For the hole, samples stop 0.02 from x=1 and an open circle marks (1,2). For a pole, sampling stops 0.08 from the excluded input. No output values are clamped to fake a finite asymptote, and no segment is drawn across the gap.

These gaps are drawing choices, not extra domain exclusions of the mathematical function. Values closer to the excluded point still belong to its domain when defined. The probe readout evaluates the requested input separately from the drawing grid.

The polynomial mode x²−c has no denominator restriction. Parameter c changes the pole location or polynomial offset; it does not change the fixed cancellation example. The displayed formula makes the current case explicit.

## Algebraic identities and AI implementations

Symbolic simplification, compiler rewrites and loss transformations must preserve required domains and boundary cases. Dividing by a norm assumes nonzero norm. A normalized vector at zero needs a documented rule or a statement that it is undefined, rather than quietly borrowing an identity from nonzero vectors.

A smooth-looking finite plot does not prove a function is smooth everywhere. Automatic differentiation at a point evaluates the implemented graph and its conventions; rewriting a graph can change numerical behavior or expose a previously excluded input. Validate the mathematical domain and finite-precision behavior separately.

## Exercises and reference

1. Simplify (x²−4)/(x−2) and state the original domain. **Answer:** x+2 for x≠2.
2. Does assigning the missing value change the original function? **Answer:** yes, it defines an extension.
3. Explain why a finite sampled pole cannot demonstrate a finite function value at the excluded point.
4. Find a normalization operation in AI code and identify its zero-denominator condition before applying algebraic cancellation.

For polynomial and rational functions see [OpenStax Algebra and Trigonometry 2e](https://openstax.org/details/books/algebra-and-trigonometry-2e). The excluded-point and split-path rules above are explicitly specified for this lab.
