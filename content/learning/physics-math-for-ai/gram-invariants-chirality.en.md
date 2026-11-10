## An invariant feature can deliberately discard information

For ordered vectors v₀,v₁,v₂ in R³, the Gram matrix stores all pairwise dot products. Applying an orthogonal transformation g preserves them because gᵀg=I. Both rotations and reflections are orthogonal. Therefore Gram features cannot generally distinguish full-rank mirror configurations when only proper rotations should be considered equivalent.

The lab uses v₀=ex, v₁=ey and v₂=(0.3,0.4,h). It applies a z rotation, a y rotation and an optional reflection of the z coordinate. All operations and dot products are computed in three dimensions. The upper image is only an oblique projection, while the lower plot compares the six independent Gram entries.

$$
V=[v_0\ v_1\ v_2],\quad G=V^TV,\qquad
(gV)^T(gV)=G.
$$

## What the Gram matrix does determine

If two ordered vector configurations have equal Gram matrices, the map sending one set of generators to the other preserves all linear-combination norms. It is therefore a well-defined isometry between their spans. Extending orthonormal bases of the complementary spaces gives an orthogonal transformation in R³. Thus the full Gram matrix determines the O(3) orbit of this ordered configuration, not just individual lengths.

Using only the three lengths would discard pairwise angles as well. Arbitrary learned invariant features need not be complete even for O(3); the completeness argument here uses the entire Gram matrix and the stated ordered-vector action.

## Signed volume distinguishes handedness

The triple product χ=detV measures oriented volume. Under the transform,

$$
\chi'=\det(g)\chi,\qquad \det G=(\det V)^2=\chi^2.
$$

For h=1, the original volume is one. Proper rotations preserve it, while the reflected case has volume −1 and exactly the same Gram matrix. For full-rank triples, Gram plus the volume sign distinguishes the two SO(3) orientation classes inside an O(3) orbit.

At h=0, the vectors span a plane and volume vanishes. There is no nonzero-volume handedness sign to report, so the interface leaves that sign undefined. It does not manufacture positive orientation at degeneracy.

## Numerical factorization matters

The determinant computed by expanding nearly canceling Gram entries may show a tiny negative floating-point residue when the exact determinant is zero. The readout separates that expanded result from the stable identity detG=χ². This does not make an exact Gram matrix indefinite or justify interpreting roundoff as negative squared volume. The raw discrepancy remains visible rather than being silently clamped. The transformed-volume readout likewise uses the known orthogonal determinant law and exposes the expanded floating-point determinant separately, avoiding an invented orientation at near-degeneracy.

## Exercise and explained answer

Compare proper and reflected transforms at h=1. All six Gram entries agree, but volume changes from +1 to −1. Next set h=0 and explain why rank drops to two, the stable Gram determinant becomes zero and orientation sign disappears. Vary the rotations without changing h and check the same invariants.

## Research connection

Invariant molecular or point-cloud features should match the task's equivalence class. Enforcing reflection invariance can erase chirality relevant to chemistry or geometry. Conversely, attaching orientation information when reflection should be irrelevant can create unnecessary distinctions. Specify O(3) versus SO(3), ordering or permutation treatment, translation centering and degeneracy assumptions before claiming that a feature captures the desired shape completely.

Further reading: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
