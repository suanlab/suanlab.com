## Ask which conditions are meant to hold together

Fairness criteria condition on different events. A positive-decision rate averages over the whole group, a true-positive rate conditions on the actual label and positive predictive value conditions on the decision. Equalities of these different probabilities need not be compatible.

This lesson assumes the target distributions are known, separating compatibility from the selection-identification problem in the preceding unit. The task is algebraic: for a specified pair of binary joint distributions, which equalities hold, and which assumptions make a broader impossibility argument valid? A failed equality at one configured point is not by itself a proof that no other point can satisfy a collection of constraints.

We use G0 and G1, base rates pi0 and pi1, true-positive parameters t0 and t1 and false-positive parameters f0 and f1. All controls are integer percentages. The distributional masses are expected quantities, not estimated finite-sample counts.

## Construct the joint distribution from conditional rates

In one group with base rate pi, true-positive rate t and false-positive rate f, the four probability masses are

$$
TP=\pi t,\quad FP=(1-\pi)f,\quad
FN=\pi(1-t),\quad TN=(1-\pi)(1-f).
$$

The lab scales these probabilities by10000 so every cell is an exact integer for the chosen controls. The four cells sum to10000 and have the declared label marginal.

At pi0 or1, one actual label class is absent. A conditional-rate parameter for that absent class can still be used to construct the same joint distribution, but it is not an observable conditional rate. The lab displays the actual denominator and marks the resulting rate undefined. The main compatibility derivations below assume both base rates are interior.

## Name each equality separately

Decision-rate parity, often called demographic parity, requires matching $\Pr(\widehat Y=1\mid G)$. Equal opportunity requires matching TPR. Equalized odds requires matching both TPR and FPR. Positive predictive parity requires matching PPV; negative predictive parity requires matching NPV.

[Hardt, Price and Srebro](https://arxiv.org/pdf/1610.02413) define equalized odds and equal opportunity through their conditioning events. [Chouldechova](https://arxiv.org/pdf/1703.00056) studies relations between predictive parity, error rates and base rates. The exact elementary identities used in this experiment are derived below with explicit boundary assumptions.

Undefined is a third state in the displayed comparisons. When a positive decision never occurs, PPV is undefined and cannot be counted as a satisfied parity condition. Each defined equality uses integer cross-products; its truth does not depend on whether two rounded numbers look identical.

## Equalized odds and decision-rate parity

Suppose equalized odds holds with common t and f. The group positive-decision rate is

$$
s_g=\pi_g t+(1-\pi_g)f=f+(t-f)\pi_g.
$$

Subtract the two rates:

$$
s_1-s_0=(t-f)(\pi_1-\pi_0).
$$

If the base rates differ, adding decision-rate parity forces t=f. The decision then has the same positive probability regardless of the true label within each group. It can still be randomized and need not be a constant decision, but its label-conditioned distinction has disappeared.

This conclusion concerns these particular simultaneous constraints. It does not say that one is universally preferable, nor that every classifier with a small demographic gap must be uninformative: the common-error-rate assumption is part of the derivation.

## Derive the PPV relation and its strictness conditions

Bayes conditioning on a positive decision gives

$$
PPV(\pi)=\frac{t\pi}{t\pi+f(1-\pi)}
=\frac{t\pi}{f+(t-f)\pi}.
$$

For shared t>0 and f>0, differentiate:

$$
\frac{d\,PPV}{d\pi}=\frac{tf}{[f+(t-f)\pi]^2}>0.
$$

Therefore distinct interior base rates imply distinct PPVs under common positive t and f. Equalized odds and positive predictive parity cannot both hold in this nondegenerate region.

The words “positive t and f” are essential. The derivative is not strictly positive on every boundary, and an expression with a zero conditioning denominator has no PPV value. A useful theorem states those conditions rather than hiding them behind a generic phrase such as “fairness is impossible.”

## Work the unequal-base guide

Set base rates(.2,.6), with t=.8 and f=.2 in both groups. The expected confusion tables are

$$
G_0:(1600,1600,400,6400),\qquad
G_1:(4800,800,1200,3200).
$$

Both groups have TPR = 0.8 and FPR = 0.2, so equalized odds holds. But PPV is1600/3200=1/2 in G0 and4800/5600=6/7 in G1. Positive-decision rates are.32 and.56. NPV is16/17 versus8/11.

Accuracy happens to be.8 in both groups here because t=1-f=.8. That symmetry is not a general implication of equalized odds. For example shared t=.8,f=.1 gives accuracies.88 and.84 at these same base rates. The conditioning events explain why an aggregate metric can change even with equal error rates.

## Equal base rates permit compatibility

The second guide sets both bases to.5 with shared t=.8 and f=.2. Both tables are(4000,1000,1000,4000). Decision rates = 0.5, PPV = 0.8,NPV = 0.8 and the two error rates all match.

This is an imperfect classifier: both kinds of error remain. It nevertheless satisfies the displayed parity equalities at this configuration. Equal base rates permit compatibility; they do not force it if the group-specific decision parameters are chosen differently.

The statement is about known distributions. Whether a trained model can attain a chosen pair of conditional rates with its available features is a separate feasibility problem not solved by moving these controls.

## Zero FPR gives a genuine imperfect exception

The third guide keeps bases(.2,.6) and t=.8 but sets $f=0$ in both groups. The tables become

$$
G_0:(1600,0,400,8000),\qquad
G_1:(4800,0,1200,4000).
$$

PPV is1 in both groups and equalized odds still holds, while20% of actual positives are missed. Thus equalized odds plus PPV parity does not generally require perfect prediction. The positive-f assumption in the strict derivative argument has failed.

Decision rates are.16 and.48, so demographic parity does not hold. NPV is20/21 versus10/13. The exception permits a specific pair of conditions, not every condition simultaneously.

## Inspect the remaining boundaries

At $t=0$, $f>0$, all positive decisions are false positives and $PPV=0$ can match across groups. At $t=f=0$ no positive decisions exist, so PPV is undefined. At $t=f=1$ every decision is positive, giving PPV equal to the base rate and undefined NPV.

A perfect decision has $t=1$, $f=0$. It gives both predictive values1 but its positive-decision rate is the group base rate; unequal bases therefore still prevent demographic parity. A perfectly reversed decision has $t=0$, $f=1$. It gives both predictive values0 with equalized odds, again without equal decision rates at unequal bases.

These cases show why a theorem about several criteria must specify the criteria and exclusions exactly. “Only perfect prediction” may describe a different theorem involving calibrated probability scores and two class-conditional score means; it is not a safe substitute for the binary PPV calculation.

## Read exact equalities and hypothetical curves

The first diagram displays the joint cells and exact predictive-value fractions. The second distinguishes defined equalities from the hypotheses of the nondegenerate conflict argument. A missing hypothesis does not automatically certify compatibility at the current point.

The final graph fixes each group's t/f parameters and varies a hypothetical base rate. Blue is the G0 rate pair and orange is the G1 pair. Dots use the actual group bases. If the pairs are shared, the two curves coincide even though the dots can have different heights. Open endpoints indicate undefined positive-decision conditionals; an entirely absent curve means no base rate produces a positive decision under that configured pair.

The graph illustrates exact ratio formulas using floating coordinates. It is not a search over all learned models or all threshold policies. Equality badges use the underlying integer cells, while decimal readouts and logarithm-free curves are rounded for display.

## Why binary PPV does not mean score calibration

PPV parity compares $\Pr(Y=1\mid\widehat Y=1,G)$ across groups; it need not equal1. A probability score R calibrated within groups instead requires E[Y|R,G]=R. This includes every occupied score level, not only the positive decision.

If the imperfect zero-FPR decision is interpreted as a probability score taking values0 and1, its score $R=1$ bin is calibrated but its score $R=0$ bin is not. In G0, $\Pr(Y=1\mid R=0)$=400/8400=1/21, rather than0. The classifier therefore does not contradict the companion score theorem.

Thresholding a calibrated score can also change binary error rates. A score-level guarantee is not automatically identical to every thresholded binary parity statement.

## Exercises and worked solutions

1. With shared t=.8,f=.2 and bases.2,.6, compute the decision-rate gap. It is(.8-.2)(.6-.2)=.24, matching.56-.32.
2. If those unequal bases must also have demographic parity under shared error rates, what relation is necessary? t=f. At t=f=.5,PPV remains the base rate, so PPV parity still fails.
3. Does zero FPR and TPR = 0.8 imply no errors? No. The false-negative rate is.2, though PPV equals1 whenever positive decisions occur.
4. If no positive decisions occur, is positive predictive parity true? It is undefined in this rate-comparison convention, because each required PPV denominator is0.
5. Why can equalized odds coexist with different accuracy? Accuracy is t pi+(1-f)(1-pi), which depends on pi unless t=1-f or the bases match.
6. Does a failure at one arbitrary set of unequal group rates prove a universal incompatibility theorem? No. The general conclusion requires a derivation over the intended feasible class and explicit assumptions.
7. Is a perfectly reversed classifier a calibrated probability score if its labels are used as scores0/1? No. Its positive-score bin has actual positive fraction0, contradicting calibration at score $R=1$.

## Toward the score theorem

The next unit states a different set of simultaneous requirements: within-group probability-score calibration and equality of both positive-class and negative-class score means. Its perfect-prediction conclusion follows from a bounded-score expectation argument. Keeping that theorem separate prevents a binary boundary example from being mislabeled a contradiction.
