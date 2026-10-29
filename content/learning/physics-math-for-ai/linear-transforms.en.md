## Changing a space, not just a point

How can one language describe stretching, rotating, and shearing vectors? **A matrix records where the basis vectors go.** Knowing its action on a basis lets us transform every vector.

This lesson uses column vectors with three components and 3×3 matrices. Coordinates are dimensionless. Distinguish a mathematical transformation from a change in camera viewpoint.

## What you will gain

The goal is to connect **how an output is calculated with why the geometry changes**, rather than memorize a multiplication rule. Begin with two basis vectors in a plane, then extend the same reasoning to the 3×3 experiment.

- Use rows to calculate output components and columns to read transformed basis vectors.
- Distinguish rotation, scaling, shear and reflection, and explain why equal volume need not mean equal shape.
- Interpret information loss and a neural-network linear layer using small numerical examples.

## Read the symbols and shapes first

| Symbol or term | How to read it | Concrete meaning |
| --- | --- | --- |
| $\mathbf x$ | Input vector | Three components written as a 3×1 column |
| $A$ | Transformation matrix | Nine numbers arranged in three rows and three columns |
| $a_{ij}$ | Entry in row i, column j | Contribution of input component j to output component i |
| $A\mathbf x$ | A times x | Output from applying the same rule to every input |
| $\mathbf e_1,\mathbf e_2,\mathbf e_3$ | Standard basis | `(1,0,0)`, `(0,1,0)`, `(0,0,1)` |
| $\det(A)$ | Determinant | Signed volume scale of a square matrix |

The superscript T denotes transpose. Here, read `(1,2,3)ᵀ` as three components placed vertically in a column. Transposing a matrix exchanges its rows and columns.

Distinguish matrix dimensions from the number of independent directions its outputs can span. The latter connects to information loss below. A 3×3 matrix does not necessarily preserve information in three directions.

For column vectors, `(3×3)·(3×1)=(3×1)`. The matrix's column count must match the vector's row count. The experiment's fields are entered by row, while the mathematical convention is $\mathbf y=A\mathbf x$ with column vectors.

## A basis supplies the building blocks

In a plane, take $\mathbf e_1=(1,0)$ and $\mathbf e_2=(0,1)$. The vector `(1,2)` uses one copy of the first building block and two of the second:

$$
\mathbf x=1\mathbf e_1+2\mathbf e_2.
$$

A linear transformation preserves this assembly rule. When the blocks become $A\mathbf e_1$ and $A\mathbf e_2$, the coefficients remain 1 and 2. **Knowing where the basis goes therefore determines where every other vector goes.** An arbitrary nonlinear transformation need not have this property.

## Reading the columns

With basis vectors $\mathbf e_1,\mathbf e_2,\mathbf e_3$,

$$
A\mathbf x=x_1A\mathbf e_1+x_2A\mathbf e_2+x_3A\mathbf e_3.
$$

The first, second, and third columns are the transformed basis vectors. Linearity means

$$
A(\mathbf u+\mathbf v)=A\mathbf u+A\mathbf v,\qquad A(c\mathbf u)=cA\mathbf u.
$$

When you change the matrix, check that the basis arrows, unit cube, and point all follow the same transformation. Gray shows the original shape; color shows the transformed shape.

## Work through a calculation

A matrix that doubles x, halves y, and preserves z is

$$
A=\begin{pmatrix}2&0&0\\0&0.5&0\\0&0&1\end{pmatrix}.
$$

For $\mathbf x=(1,2,3)^T$,

$$
A\mathbf x=(2,1,3)^T.
$$

Each output component is the dot product of a row with the input. Change one matrix entry and predict which output component will change.

## Calculate by row and check by column

Multiply each row of the example above by matching components of `(1,2,3)` and add:

| Output component | Row dotted with input | Result |
| --- | --- | --- |
| First | `2×1 + 0×2 + 0×3` | 2 |
| Second | `0×1 + 0.5×2 + 0×3` | 1 |
| Third | `0×1 + 0×2 + 1×3` | 3 |

Reading by column gives `1×(2,0,0) + 2×(0,0.5,0) + 3×(0,0,1)=(2,1,3)`. Row calculations and a weighted sum of columns are **two views of the same multiplication**. Editing a row changes one output component's rule. Editing a column changes how one input component contributes to all outputs.

### Build a shear by hand

In the plane z=0, consider

$$
A=\begin{pmatrix}1&1&0\\0&1&0\\0&0&1\end{pmatrix},\qquad
x'=x+y,\quad y'=y.
$$

The first basis vector `(1,0)` stays fixed, and the second `(0,1)` tilts to `(1,1)`. Assemble the input `(1,2)` from one transformed first basis vector and two transformed second basis vectors: the result is `(3,2)`. Points with y=0 stay fixed, while points with larger y move farther along x. This differs from a translation, which moves every point by the same vector.

Use the concept diagram's “change the basis → assemble with the same coefficients” steps, then select **Shear** in the 3D experiment to compare the result.

## Six transformations

- **Identity:** The basis and every point stay unchanged.
- **Axis scaling:** Length changes by different factors along different axes.
- **90° rotation about z:** `(1,0,0)` maps to `(0,1,0)` while preserving length.
- **Shear:** With $x'=x+y,\ y'=y$, `(1,2,0)` becomes `(3,2,0)`.
- **Reflection:** One axis reverses. Distinguish volume magnitude from orientation.
- **Collapse:** Setting one axis to zero can map different inputs to the same output.

Compare the original and transformed bases using presets, then edit the matrix entries yourself.

## Determinant is a signed volume scale

The volume magnitude of the transformed unit cube is $|\det(A)|$. The determinant's sign records orientation reversal.

$$
\det\begin{pmatrix}-1&0&0\\0&1&0\\0&0&1\end{pmatrix}=-1.
$$

This reflection has volume scale 1, **not negative volume**. The axis-scaling matrix above also has determinant $2\times0.5\times1=1$, yet it does not preserve every vector's length. Equal volume does not mean equal shape.

## Does the order matter?

Let S be the reference scaling matrix and R a 90° rotation around z. Starting from `(1,0,0)`, **scale then rotate** gives `(2,0,0) → (0,2,0)`. **Rotate then scale** gives `(0,1,0) → (0,0.5,0)`.

$$
RS\mathbf e_1=(0,2,0)^T,\qquad SR\mathbf e_1=(0,0.5,0)^T.
$$

In matrix products, the rightmost transformation acts first. In general, $RS\ne SR$. Record operation order when analyzing multiple neural-network layers as well. If an elementwise nonlinear activation lies between the layers, the entire computation cannot simply be combined into a product of two matrices.

## When no inverse exists

The matrix

$$
A=\begin{pmatrix}1&0&0\\0&0&0\\0&0&1\end{pmatrix}
$$

erases every point's y component. Both `(1,2,3)` and `(1,-2,3)` become `(1,0,3)`. The original y cannot be recovered from the output, so there is no inverse. Select the collapse preset and change the input y.

## Connecting to linear layers in AI

Matrix-vector multiplication is a core operation in a neural-network linear layer. Adding a bias gives $A\mathbf x+\mathbf b$, which is generally an affine map rather than a linear map. Nonlinear activation is another separate operation.

The 3D scene is a small example for understanding these relationships. A high-dimensional tensor in a real model is not literally this three-dimensional space.

### A small AI example: two features, two outputs

Use these invented, fixed weights and input for illustration:

$$
W=\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad
\mathbf x=\begin{pmatrix}2\\1\end{pmatrix},\quad
W\mathbf x=\begin{pmatrix}3\\1\end{pmatrix}.
$$

The first output reads the sum of the features; the second reads their difference. Adding bias $\mathbf b=(1,-1)^T$ gives `(4,0)`. Applying ReLU, which replaces negative entries with zero, leaves this particular output at `(4,0)`, but the activation is a separate nonlinear rule. These numbers illustrate the computation of a layer, not a claim about training performance.

For n inputs and m outputs, W has shape m×n. Rows specify output rules and columns specify input contributions, even in high dimensions. However, **a 3D cube's volume interpretation and a 3×3 determinant cannot simply be applied to a rectangular matrix.**

## Read information loss as a research question

In the collapse preset, one transformed basis vector becomes zero. Only two independent output directions remain; their number is the rank. The deleted input y cannot be recovered from output alone. This is the numerical reason different inputs to a singular matrix can share an output.

Having an inverse is also different from being numerically stable. In two dimensions, `diag(1,0.01)` shrinks the second component by 100, and its inverse multiplies that component by 100. An output error of 0.001 in that component becomes a reconstruction error of 0.1. Singular values and condition numbers will later describe this sensitivity more precisely.

In research, distinguish discarding information, retaining task-relevant information and sensitivity to numerical error. A reduction in dimension alone does not establish improved performance.

## Common misconceptions

| Thought | Counterexample or check |
| --- | --- |
| Matrix rows are transformed basis vectors | Under the column-vector convention, columns are transformed basis vectors |
| Determinant 1 means rotation | Shear and anisotropic scaling can also have determinant 1 |
| Determinant −1 means negative volume | Volume scale is absolute value 1; the sign indicates orientation reversal |
| Every square matrix has an inverse | Deleting a direction makes the transformation noninvertible |

## Challenge and check

Construct a determinant-1 matrix that sends `(1,2,0)` to `(3,2,0)`. Must it be a rotation?

**Explanation:** The shear $\begin{pmatrix}1&1&0\\0&1&0\\0&0&1\end{pmatrix}$ works. Determinant 1 alone does not imply rotation.

Learning reference: [MIT — Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
