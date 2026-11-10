## Replace exactly one observation

How much does learning change when one label is replaced in a dataset of fixed size? Use an intercept-only predictor w with labels in [-1,1]. Only the first label changes; the remainder alternate +0.5,-0.5. The experiment separates sensitivity to this actual neighbor from a guarantee over all allowed datasets.

The objective is mean((w-y)²)/2+lambda*w²/2, with exact solution w=mean(y)/(1+lambda). Even lambda zero gives curvature one in this special intercept problem. That uniqueness should not be generalized to an unregularized overparameterized model.

## Uniform prediction and loss bounds

Replacing a bounded label changes its sample mean by at most 2/N, hence

$$
|w_S-w_{S'}|\le\frac{2}{N(1+\lambda)}.
$$

Both weights and the test label lie in [-1,1]. The absolute weight derivative of half-squared loss is at most two, while the loss range is [0,2]. Therefore

$$
\beta_{uniform}\le\min\left(2,\frac{4}{N(1+\lambda)}\right).
$$

This is an upper bound quantified over every allowed replace-one dataset and test label, not the exact value for the currently selected neighbor. Sharper analysis is possible; this transparent bound is deliberately easy to verify.

## Exact supremum for the selected neighbor

For the two solutions w and v, test-loss difference is affine in y:

$$
\ell(v,y)-\ell(w,y)=(v-w)\left(\frac{v+w}{2}-y\right).
$$

Its maximum absolute value on [-1,1] is attained at an endpoint. Blue and orange show the actual loss functions. The signed query change and endpoint supremum are reported separately. One small query change cannot replace a full test-interval supremum or a uniform-over-datasets statement.

## Worked example and exercise

For N=2, lambda=1, datasets [-1,.5] and [1,.5] give w=-.125 and v=.375. At y=-1 the loss change is .5625; at y=1 it is -.4375. The exact neighboring supremum is .5625, while the uniform bound 4/(2·2)=1 is more conservative.

Exercise: let N=1, lambda=0, replace -1 by +1, and test at y=1. Solution: the original loss is two and the new loss is zero, attaining absolute change two. Increasing N or lambda reduces the prediction bound, but a smaller individual change is not a guarantee of improved test performance everywhere.

## Stability and generalization

For an algorithm that is uniformly beta-stable and IID training samples, a replace-one argument controls the magnitude of the expected generalization gap by beta. This is not a pointwise claim that every observed gap on the displayed deterministic dataset is bounded by beta. High-probability statements need additional conditions and terms.

In AI research, distinguish observed sensitivity, algorithm-wide uniform properties, expected generalization and confidence guarantees. This experiment replaces a label only; it does not simultaneously perturb features, architecture or the entire data-generating process.

Reference: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
