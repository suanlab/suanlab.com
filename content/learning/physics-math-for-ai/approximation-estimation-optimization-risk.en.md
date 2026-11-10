## Define the objectives behind the three error names

Simply naming approximation, estimation and optimization error can suggest that every displayed term must be positive. First specify the oracle and objective being compared. Here x is uniform on {-1,0,1}, the true regression function is f(x)=x+c x², and labels add independent centered Rademacher noise ±sigma. Loss is half the squared error.

The hypothesis has one coefficient multiplying either a constant feature or x. Actual GD runs on the sample quadratic, while the exact sample ERM and population class oracle are solved separately. Blue is population risk and orange is empirical risk: they average over different objects.

## An exact telescoping identity

Bayes risk is sigma²/2. Let w_H minimize population risk within the class, and w_S minimize empirical risk. For the current iterate,

$$
R(w_k)-R_{Bayes}=[R(w_H)-R_{Bayes}]
+[R(w_S)-R(w_H)]+[R(w_k)-R(w_S)].
$$

The first term is approximation error and the second is this sample's estimation effect. The final population optimization effect can be negative. In contrast, the same-objective empirical gap Rhat(w_k)-Rhat(w_S) is nonnegative. Replacing the last population term with that empirical gap would generally invalidate the identity.

## Actual quadratic optimization

For feature phi, sample curvature is A=mean(phi²) and the linear term is B=mean(phi y):

$$
w_{k+1}=w_k-\eta(Aw_k-B),\qquad w_S=B/A\quad(A>0).
$$

When all sampled features are zero, A=B=0 and the empirical objective is flat. The declared tie rule chooses w_S=0, and GD stays at zero. Having informative features in the population does not mean a particular sample contains them. No hidden ridge is added to change this degenerate case.

## Worked example and exercise

For c=0, sigma=1, N=2 and seed nine, the actual observations are (-1,-2),(1,2). Linear-feature ERM is w=2. Population risk is (w-1)²/3+1/2, so the class oracle is one and the estimation effect is 1/3. Starting at zero with eta=1/2, one GD step reaches w=1. Its population optimization effect is -1/3, even though the empirical optimization gap remains 1/2.

Exercise: compute constant-class approximation error at c=1. Solution: true values are [0,0,2], with mean 2/3. Half their mean squared residual is 4/9. For the linear feature, the oracle coefficient is one and approximation error is 1/3. More optimization iterations cannot remove a representation restriction.

## Interpret early stopping carefully

The example proves that early stopping can outperform sample ERM in population risk, not that it always does. Report the optimized objective, evaluation law, hypothesis class, initialization and step settings together. Identifying the three effects in real research requires an oracle or independent estimation and experimental design. This finite world provides transparent definitions and checks, not a universal recipe for choosing stopping time.

Reference: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
