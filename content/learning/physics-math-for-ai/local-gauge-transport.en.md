## Local frames can change without changing the base geometry

A global symmetry applies one transformation throughout the domain. A local gauge change can choose a different feature frame at each location. The base triangle in this example stays fixed. Its three feature vectors hᵢ have two components, and each frame change gᵢ belongs to SO(2). We use an explicit component convention hᵢ'=gᵢhᵢ; alternative passive-frame conventions use inverses and must be handled consistently.

Changing coordinates independently makes direct addition of neighboring components ambiguous. A transport link Uᵢⱼ maps components at node j into the component space at node i. Its transformation law follows from demanding consistent transport:

$$
U_{ij}'=g_iU_{ij}g_j^{-1},\qquad
U_{ij}'h_j'=g_i(U_{ij}h_j).
$$

The inverse at the input endpoint cancels its local frame change, leaving only the output-point change. This cancellation is the basic reason transported messages can be gauge covariant.

## Work a concrete message

Let h₁=(0.5,1), U₀₁=Rπ/2. Then m₀=U₀₁h₁=(−1,0.5). Change only the relevant output frame by g₀=Rπ/2 while g₁=I. Correctly changing the link produces m₀'=g₀m₀=(−0.5,−1). If the link is incorrectly held fixed, the message remains (−1,0.5), giving error √2.5≈1.581.

The orange segments show changed local feature components at a common display scale of one quarter. They are not physical displacement vectors moving triangle vertices. Individual feature components and link angles can change while suitable scalar norms or loop traces remain fixed.

## A closed loop transforms by conjugation

Links follow the directed cycle 0 from 1, 1 from 2, and 2 from 0. Their product L=U₀₁U₁₂U₂₀ acts at node zero. All intermediate local frame factors cancel:

$$
L'=g_0Lg_0^{-1},\qquad \operatorname{tr}(L')=\operatorname{tr}(L).
$$

Here SO(2) is Abelian, so the loop matrix itself is unchanged. In a non-Abelian group the general conclusion is conjugacy, not equality of all matrix entries. With original link angles (π/2,π/2,0), L=Rπ=−I and trace is −2.

## Exercise and explained answer

Start from identity links by setting the reference link angle to zero. Choose local angles (π/2,π,−π/2). The transformed links have nonzero component angles, but the loop is still identity with trace two. This is a pure change of frames: nonzero link coordinates alone do not demonstrate curvature. Verify both the loop trace and message-covariance discrepancy.

## Connection to geometry and AI

On curved surfaces, neighboring tangent features require compatible local frames and transport. Gauge-equivariant constructions organize this dependence systematically. A discrete connection can carry genuine holonomy, while a gauge change changes its representation. This finite three-node SO(2) calculation introduces those laws; it does not implement a full non-Abelian gauge field, continuum curvature tensor or the entire architecture of a gauge-equivariant network.

Further reading: [Cohen et al., Gauge Equivariant Convolutional Networks](https://proceedings.mlr.press/v97/cohen19d.html).
