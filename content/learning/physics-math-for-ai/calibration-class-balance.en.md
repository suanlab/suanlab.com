## Probability scores and binary decisions are different objects

A binary decision is a value0 or1. A probability score R in[0,1] represents a degree of belief, and its calibration concerns the relationship between each score level and actual labels. A thresholded decision is a further function of that score.

The previous unit showed that binary equalized odds and PPV parity can coexist for an imperfect zero-FPR classifier. This unit studies another theorem: within-group score calibration plus equality of both class-conditional score means. Its assumptions are different, so its exceptional cases must not be transferred to the binary theorem by analogy.

We assume both groups have interior positive-label base rates. Both actual label classes therefore have positive mass in each group. This lets the two class-conditional means be defined without arbitrary versions on null events.

## Define calibration on occupied score levels

Within-group calibration is

$$
\mathbb E[Y\mid R,G]=R
$$

almost surely. For a finite score bin r with positive group mass, it says $\Pr(Y=1\mid R=r,G)=r$. A bin with zero mass has no conditional fraction to estimate or compare; it is not a failed calibration point.

Calibration does not mean that every prediction is correct or that a score must be discriminative. If every member of a group receives its true base rate pi, then that constant score is calibrated even though it provides no within-group ranking information.

The reliability diagram plots occupied score bins against their actual positive fractions. In this explicitly constructed model all points lie on the diagonal by proof and exact integer checks. An empirical diagram with noisy estimated bins would require a separate uncertainty analysis.

## State the two mean-balance conditions

Positive-class score balance requires

$$
\mathbb E[R\mid Y=1,G=0]=\mathbb E[R\mid Y=1,G=1]=\mu_+.
$$

Negative-class score balance similarly requires a common mu_minus among $Y=0$ in both groups. These are means of a probability score, not necessarily TPR and FPR of a thresholded decision.

[Kleinberg, Mullainathan and Raghavan](https://arxiv.org/pdf/1609.05807) analyze score calibration and class-balance constraints. The short expectation proof below makes the relevant compatibility mechanism explicit. We do not equate these score constraints with a single binary predictive-parity condition.

## Derive the general compatibility theorem

Taking group-conditional expectations of calibration gives

$$
\mathbb E[R\mid G=g]=\mathbb E[Y\mid G=g]=\pi_g.
$$

If both class means are shared across groups, split the same expectation by Y:

$$
\pi_g=\pi_g\mu_+ +(1-\pi_g)\mu_-.
$$

Equivalently pi_g(1-mu_plus+mu_minus)=mu_minus. Subtract the equations for two unequal base rates. Their nonzero difference forces1-mu_plus+mu_minus=0. Substitution then gives $\mu_-=0$ and $\mu_+=1$.

Because0<=R<=1, a conditional mean0 among negatives implies $R=0$ almost surely there. A conditional mean1 among positives implies $1-R=0$ almost surely there. Therefore R=Y almost surely in each group: scores perfectly distinguish the target.

The bounded-score assumption and the positive class masses matter. The algebra and endpoint-expectation argument are the proof; a finite parameter grid or reliability diagram alone would not establish the theorem for arbitrary score distributions.

## Understand the equal-base exception

If pi0=pi1, subtracting the mean equations gives no new restriction. Nonperfect calibrated scores can then have matching class means. A constant score equal to the common base rate is the simplest example.

The third guide uses a more informative family at common base.5. It has positive-class mean3/4,negative-class mean1/4 and nonzero Brier risk1/8 in both groups. Calibration and both balances hold without perfect scores.

Thus the theorem says that unequal interior bases together with all the stated requirements force perfection. It does not claim that calibration alone forces perfection, or that all compatible models must be perfect when bases are equal.

## Build a transparent feature model

Let Y have group base rate pi. Generate an informative flag A independently of Y with probability rho. If $A=1$, the observed feature reveals Y; otherwise the feature is bottom. The Bayes probability given this observed feature and group is

$$
R=\begin{cases}Y,&A=1,\\\pi_g,&A=0.\end{cases}
$$

This is an idealized generative feature/target law. It is not a deployed predictor assumed to receive an unobserved outcome. Its purpose is to provide an exactly enumerable family in which calibration and the theorem's exceptions can be inspected.

For an interior base rate the possible score levels are0,pi and1. Their positive/negative probability masses are respectively

$$
(0,\rho(1-\pi)),\quad
((1-\rho)\pi,(1-\rho)(1-\pi)),\quad
(\rho\pi,0).
$$

At the middle score, the actual positive fraction is pi. At scores0 and1 it is0 and1 whenever those bins occur. Thus every occupied bin is calibrated.

## Calculate class means and the balance curve

Conditioning on Y gives

$$
\mu_{+,g}=\rho+(1-\rho)\pi_g,\qquad
\mu_{-,g}=(1-\rho)\pi_g.
$$

Both group differences therefore equal

$$
(1-\rho)(\pi_1-\pi_0).
$$

For this family, the differences vanish when the base rates match or rho1. This agrees with the general theorem, but the curve is a property of this feature model rather than a parametrization of all calibrated score distributions.

The lab preserves the class-mean fractions and decides equality with integer cross-products. Decimal bars and differences are display approximations, not the logical test of equality.

## Work the uninformative-feature guide

With bases.2 and.6 and rho0, each group's score is its own base rate. Both score laws are calibrated. In G0 both class means are.2; in G1 both are.6. Each balance gap is.4.

At threshold $\tau=0.5$, all G0 decisions are negative and all G1 decisions positive. Their TPR/FPR pairs are(0,0) and(1,1), so thresholded equalized odds fails. A calibrated score does not by itself equalize binary errors.

Expected score remains the base rate in each group, as the calibration identity requires. The output also illustrates why looking only at a global reliability diagram can miss the distinction between within-group calibration and other requirements.

## Work the perfect-information guide

Set rho1 with bases still.2 and.6. Scores now equal Y. Both positive-class means are1, both negative-class means0 and Brier risk is0. Middle score bins have zero mass and are omitted from the reliability plot.

Those empty bins are not miscalibrated. Only the endpoint bins0 and1 occur, and their positive fractions exactly match their scores. Every interior threshold gives the same perfect binary decisions.

This is the unequal-base exception required by the theorem. It is an idealized mathematical endpoint, not evidence that a particular research dataset admits such a predictor with its actual observed features.

## Quantify probability-score error

Brier risk is the mean squared probability error E[(R-Y)^2]. In this family the informative branch has zero error and the uninformative branch predicts pi, giving

$$
\mathbb E[(R-Y)^2\mid G]=(1-\rho)\pi_g(1-\pi_g).
$$

For bases.2 and.6 at rho0, the risks are.16 and.24. At common base rate $\pi=0.5$ and $\rho=0.5$, both are.125=1/8. Calibration alone therefore does not imply zero risk or equal risk across different groups.

Brier risk is different from the zero-one error of a thresholded decision. Changing the threshold leaves this score risk, the score means and calibration unchanged, while potentially changing which binary decisions are made.

## Treat threshold ties explicitly

The lab declares a positive decision when R>=tau. At base rate $\pi=0.5$ and $\rho=0.5$, score $R=0.5$ has positive mass. With $\tau=0.5$ it is included, yielding $TPR=1$ and $FPR=0.5$. With $\tau=0.51$ it is excluded, yielding $TPR=0.5$ and $FPR=0$.

The score distribution and its calibration are identical in both cases. A threshold convention is part of the binary classifier definition, not an inconsequential numerical detail. The integer-percent implementation compares score numerators directly to the threshold, so the tie decision is exact.

At differing bases, a single shared threshold can treat the two middle score levels differently. The resulting TPR/FPR discrepancy is a property of thresholding these scores, not a failure of their calibration.

## Connect back to the binary exception

Interpreting a hard decision as a probability score imposes more than PPV parity. A calibrated score taking only0 and1 must have positive fraction1 among $R=1$ and 0 among $R=0$. Both decision errors must disappear on occupied bins.

The imperfect zero-FPR classifier from the previous lesson has $PPV=1$, but its negative decision still contains false negatives. Its score $R=0$ bin would have a positive target fraction rather than0. It therefore fails score calibration and does not contradict the perfect-score theorem.

This distinction prevents an apparently conflicting pair of statements from being resolved by dropping an assumption. Binary predictive values, continuous score calibration, class-mean score balance and thresholded error parity answer different conditional questions.

## Read the diagrams and exact tables

The reliability panel plots only occupied bins; its table lists all three potential levels and shows undefined for empty-bin conditionals. Blue filled points and orange rings distinguish overlapping groups. The second panel compares positive and negative class means.

The final plot varies rho in this particular family. Its blue curve represents both class-mean gaps, which coincide here, and orange is the zero line. The selected threshold does not alter this curve. It is not an optimizer over arbitrary models.

Joint masses have denominator10000 for integer controls. Conditional means, expected scores and Brier risk retain exact integer fractions. Displayed decimals and coordinates are rounded. Interior base-rate controls enforce the class-support assumptions of the mean-balance theorem.

## Exercises and worked solutions

1. At bases.2,.6 and rho.5, find the two class-mean gaps. Both equal.5 times.4=.2. Calibration still holds in both groups.
2. At the same settings, find the positive means. They are.6 and.8; negative means are.1 and.3. They differ from the thresholded TPR/FPR pairs.
3. Why does calibration imply E[R|G]=pi? Apply iterated expectation to E[Y|R,G]=R.
4. If calibrated scores share both class means at unequal interior bases, what values must those means have? Positive mean1 and negative mean0, forcing R=Y almost surely by boundedness.
5. If a middle bin has zero mass at rho1, is calibration false there? No conditional fraction is defined there; it is omitted rather than treated as a zero-rate failure.
6. Can changing tau repair a score-mean balance gap? No. Thresholding changes the binary decision, not the original score means.
7. Why do equal bases allow nonperfect compatible scores? The two expectation equations coincide and do not force the endpoint means. The rho.5/common-base.5 example has Brier risk1/8.

## Research connection

When comparing fairness results, write the exact random object, conditioning events, support conditions and equality requirements. An incompatibility theorem can delimit feasible constraints, but does not choose a normative objective or prove a proposed training method attains a desired point. The next original scope topic turns to robust statistics and the effect of contamination on estimation.
