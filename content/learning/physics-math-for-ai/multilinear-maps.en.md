## Linear in each slot

A bilinear map F:U×V→W is linear in either argument while the other is fixed:

$$
F(\alpha u+\beta u',v)=\alpha F(u,v)+\beta F(u',v).
$$

It need not be linear in the concatenated pair. Ordinary multiplication is linear in each variable but quadruples when both are doubled.

## Store coefficients in bases

Record the kth output component of F(eᵢ,fⱼ) as Xᵢⱼₖ. Expanding both inputs gives

$$
F(u,v)_k=\sum_{i,j}X_{ijk}u_iv_j.
$$

The coefficient array represents the map relative to chosen bases. In the lab Xᵢⱼₖ=1+4i+2j+k, so basis inputs e₀,e₁ select output (3,4).

## Scaling both inputs

$$
F(2u,v)=2F(u,v),\qquad F(2u,2v)=4F(u,v).
$$

The reference outputs are (6,8) and (12,16). Any zero slot makes the output zero. An affine bias surviving a zero slot means the whole expression is not strictly multilinear in that slot.

## Linearization on a tensor product

A bilinear map corresponds to a linear map on U⊗V, evaluated at u⊗v. Its coordinates uᵢvⱼ explicitly encode interactions:

$$
\dim(U\otimes V)=\dim U\,\dim V.
$$

This differs from the sum of dimensions in concatenation. A full universal-property proof belongs to the later abstract development; coordinate expansion supplies the current calculation.

## Read and check the example

Blue is F(u,v); orange is F(2u,v). The table also displays simultaneous scaling. For both inputs (1,1), the output is (16,20). The additivity residual independently compares F(u+v,e₀) with F(u,e₀)+F(v,e₀), rather than copying one result twice.

## AI interpretation and exercises

Bilinear pooling, multimodal interactions and attention scores contain such structure. Adding softmax or a nonlinear activation generally destroys multilinearity of the entire block. Identify the exact subfunction and input slot under discussion.

1. Doubling all three slots of a trilinear map gives **eight times the output**.
2. A zero first slot gives **zero output**.
3. Concatenation and a tensor product have **different dimensions and interaction semantics**.

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
