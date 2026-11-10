## Separate a universal claim from a feasible example

An adversarial attack searches for one allowed input that the model misclassifies. A safety certificate proves a statement about every allowed input. Failure to find an example is not a proof that none exists, and failure of a sufficient certificate is not proof that an example exists.

This laboratory makes these distinctions visible in three fixed two-dimensional networks with one hidden ReLU layer. Their size permits an exact global calculation that we can compare with an attack and a conservative bound. This is an educational exact calculation for these networks, not a general deep-network verification system.

## Define the network and closed threat box

Each score has the form

$$
f(u)=b+\sum_j a_j\max(0,w_j^\top u-h_j).
$$

The true label is+1, and score zero predicts+1. Perturbations lie in a closed infinity-norm ball, equivalently a box centered at x with coordinate radius epsilon. The target is assumed unchanged throughout that set. The controls choose a fixed network, box center, radius, attack start, number of iterations and step size.

All supported inputs and activation intersections have exact binary coordinates. Fixed line normals use components0 or plus/minus1, and thresholds lie on a quarter grid. Pairwise intersections therefore introduce only power-of-two denominators in this stated domain. This arithmetic property would not automatically extend to arbitrary network weights.

## Prove where the exact minimum can occur

Each hidden unit changes its affine formula on an activation line. These lines partition the input box into convex polygonal cells. Within a cell the complete network score is affine and continuous up to its boundary.

A linear function on a nonempty compact polygon attains a minimum at a vertex, including constant functions for which many other points also minimize it. Candidate vertices come from box corners, activation-line intersections with box sides and intersections between activation lines. Enumerating all feasible candidates therefore yields the exact global minimum M for this toy problem.

Parallel or identical activation lines are handled without dividing by a zero determinant. At radius zero the set is one point. Green candidate markers show minimizing vertices, not the full set of minimizers: a constant network may attain the same minimum across the entire box.

## Derive the independent interval lower bound

For each hidden preactivation, its center value plus or minus epsilon times the L1 norm of its weight gives a valid range. Applying ReLU clips the range below at zero. At the output, a positive coefficient uses the hidden lower endpoint for a lower bound; a negative coefficient uses the upper endpoint.

This independent interval construction encloses all actual hidden outputs but may also admit mutually incompatible combinations. Its resulting lower margin bound L is therefore conservative:

$$
L\le M.
$$

A positive L certifies a strictly positive score throughout the box. A negative L means only that this sufficient procedure did not certify strict positivity. It does not assert that the network actually attains L or even that its true minimum is negative.

## Define the projected attack and its direction

The attack minimizes the score because the true label is+1. At each step it subtracts a step size times the sign of the input gradient, then projects each coordinate back into the allowed box. At a ReLU kink the implementation chooses derivative zero.

The initial point is included among visited points. The code retains the smallest score seen so far, not just the last score, because a large step can overshoot or cycle. Every visited input remains feasible. Thus the best visited score U is an upper bound on the global minimum:

$$
L\le M\le U.
$$

An attack with U below zero supplies an actual error witness. Increasing steps or changing a start can improve U but does not, by itself, prove convergence to M.

## Work the flat-gradient failure

The first network is

$$
f(u)=1-3\operatorname{ReLU}(u_0-1/4)+3\operatorname{ReLU}(u_0-3/4).
$$

It does not depend on the second coordinate. At center(0,0) both hidden units are inactive, the score is1 and the chosen gradient is zero. A center-start attack remains there for every iteration.

Nevertheless, for u_0 at least3/4 the score equals−1/2. A radius-one box around the center contains such inputs, so its exact minimum is−1/2. The failed attack reports best score1 while a genuine error exists elsewhere. The arrangement calculation detects the failure of robustness without relying on this attack to discover it.

## Change the start without changing the optimization problem

The second guide keeps the network and threat box fixed but starts the attack at the lower-right corner(1,−1). That initial point already has score−1/2, so an error witness is found immediately.

The exact minimum and interval lower bound are unchanged. They depend on the model and threat set, not on attack start, step count or step size. This is an important experimental invariant: changing the search procedure should not silently change the set over which the robustness statement is made.

A witness demonstrates failure at one feasible point. Its validity does not establish that a particular attack always finds every error, nor that an attack failure elsewhere establishes safety.

## Work the cancellation example

The third guide uses

$$
f(u)=1+\operatorname{ReLU}(u_0)-\operatorname{ReLU}(u_0)=1.
$$

For a radius-two box centered at zero, each hidden activation individually lies between0 and2. Independent output intervals may choose0 for the positive copy and2 for the negative copy, producing lower bound−1. Those two choices cannot occur simultaneously because the activations are the same function of the same input.

The true minimum and maximum are both1, and the classifier is robust throughout the box. The negative sufficient bound has failed because of lost dependency, not because of a real adversarial example. This example complements the failed attack: neither search failure nor certificate failure can be reversed into an unsupported universal conclusion.

## Interpret boundary ties and exact arithmetic

The panel reports both strict positivity and actual correctness under the declared positive tie convention. If M equals zero, the classifier is still correct everywhere for the fixed+1 target, but the strict positive-margin property fails. A zero lower bound is not a strict certificate.

The arrangement values are exact for the supported binary-grid geometry. The independent test reference clips every activation-pattern polygon, rather than repeating the production pairwise-intersection enumeration. A separate exploratory rational-arithmetic reference also checked these finite examples. A dense grid would supply additional feasible points but could miss a thin region, so it is not the proof of a global minimum.

## Read the geometry and bound diagrams

The geometry diagram shows the input box, dashed activation boundaries, candidate vertices and projected attack visits. Orange paths are search trajectories; they are not paths followed by a physical system or evidence of a universal optimization guarantee.

The comparison panel displays L,M and U and separates sufficient certification, exact toy robustness and a found error witness. The final graph plots the best visited score against attack iteration, together with the fixed exact minimum and interval lower bound. Its best-so-far curve cannot increase, but it may remain strictly above the exact minimum forever at a flat start.

## Exercises and explained solutions

1. A feasible attack finds score−0.2. What follows? Solution: an actual error exists for the+1 target, and the global minimum is at most−0.2.
2. An attack only visits positive scores. What follows? Solution: only those visits were positive; unvisited regions may contain errors.
3. The interval lower bound is−1 while the exact minimum is1. Is that contradictory? Solution: no. A lower bound need not be attainable.
4. Why retain the initial point? Solution: it is feasible and may already be the best witness, as in the corner-start guide.
5. Why are minimizing green vertices not the whole minimizer set? Solution: an affine cell can be constant, so its edges or interior can also attain the minimum.
6. Does increasing the iteration count change M? Solution: no. M depends on the fixed network and box, whereas U depends on the search.
7. At M=0, is there a negative-score error? Solution: no under the fixed positive target and positive tie convention, though strict positivity still fails.

## Connect to the wider research problem

[Madry and colleagues](https://arxiv.org/abs/1706.06083) study adversarial robustness through robust optimization. [Wong and Kolter](https://arxiv.org/abs/1711.00851) study provable bounds through a convex outer approximation. The simple interval bound and vertex enumeration here are direct teaching constructions, not reproductions of those complete methods. Their purpose is to make the direction and limitations of attack and certificate evidence explicit before larger models are considered.
