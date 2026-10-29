## Compatibility does not establish meaning

Broadcasting aligns arrays for elementwise operations. Under NumPy rules, compare trailing dimensions: sizes must match or one must be one. Missing leading axes behave as size one.

$$
(2,3)\quad\text{and}\quad(3,)\longrightarrow(2,3).
$$

A valid output shape does not guarantee the intended semantic axis was used.

## Channel versus row values

For A=[[1,2,3],[4,5,6]] and B=(1,2,3), addition uses the same channel bias in each row:

$$
Y_{ij}=A_{ij}+B_j.
$$

The result is [[2,4,6],[5,7,9]]. A (2,1) array instead provides one value per row. A shape-(2,) array is incompatible: its trailing two conflicts with three.

## A special outer-product layout

Shapes (2,1) and (1,3) expand in different directions. With values (1,4)ᵀ and (1,2,3),

$$
Y_{ij}=A_{i0}B_{0j}=\begin{pmatrix}1&2&3\\4&8&12\end{pmatrix}_{ij}.
$$

This elementwise broadcast layout happens to implement an outer product. Arbitrary elementwise multiplication is not automatically an outer or matrix product. Matrix multiplication includes an explicit summed index.

## Failure is not a zero tensor

$$
(2,3)\quad\text{and}\quad(2,)\quad\text{are incompatible}.
$$

The lab shows compatibility zero and undefined output shape/values rather than inventing zeros. The standalone shape helper also handles empty axes: (0,3) with (1,3) gives (0,3). Blindly taking a maximum would be wrong in that case.

## Memory and implementation scope

Conceptual repetition need not create physical copies; strides and views depend on implementation. The result itself may still require substantial memory. This lab computes small explicit outputs, not a running NumPy session. Reproduce the examples separately with NumPy to check the convention.

## AI connections and exercises

Per-example weighting, channel bias and attention masks use broadcasting. Equal batch and feature lengths can conceal an incorrectly aligned axis. Insert singleton axes explicitly and test with small asymmetric values.

1. (5,1) with (1,6) gives **(5,6)**.
2. (2,3) with (2,) is **incompatible**.
3. Broadcasting alone **does not perform contraction**; summation is separate.

Further reading: [NumPy broadcasting documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html).The small arrays, reconstructions and cost examples here are independently constructed.
