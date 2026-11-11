## Define the target before declaring two numbers equal

A metric is a function of a specified distribution. Two classifiers can have matching rates in an evaluation dataset while having different rates in the population where decisions will be made. Before comparing groups, state which distribution the comparison concerns and how the available observations were obtained.

This lesson uses two groups G0 and G1, a binary target Y and a binary decision Yhat. Group definitions and target labels are supplied as meaningful for this mathematical exercise; a confusion matrix cannot itself validate them. The aim is to calculate conditional quantities and identify what assumptions connect an observed distribution to a target distribution.

The experiment uses expected cell masses. A population cell mass40 included with probability1/2 contributes expected selected mass20. These are distributional calculations, not a promise that every finite random sample contains exactly20 observations in that cell.

## Read the confusion matrix with a fixed orientation

Use the order(TP,FP,FN,TN): true positive, false positive, false negative and true negative. In the diagram, rows are Y1 and Y0; columns are prediction1 and prediction0. Consequently TP and TN are diagonal cells, while FN is above TN and FP is below TP.

For example(20,5,5,20) has total mass50. It contains25 actual positives,25 actual negatives,25 positive decisions and25 negative decisions. The cells are not probabilities until normalized by the relevant denominator.

The word “selection” has two roles that must be separated. Metric0 is the positive-decision rate Pr(Yhat1|G). Data inclusion is instead a separate indicator S: whether an individual appears in the evaluation data. We write q for an inclusion probability. A decision need not mean inclusion in the dataset.

## Derive the seven displayed rates

For a group with total N, the positive-decision rate and positive-label base rate are

$$
\Pr(\widehat Y=1\mid G)=\frac{TP+FP}{N},\qquad
\Pr(Y=1\mid G)=\frac{TP+FN}{N}.
$$

The conditional accuracy measures are

$$
TPR=\frac{TP}{TP+FN},\quad FPR=\frac{FP}{FP+TN},\quad
PPV=\frac{TP}{TP+FP},\quad NPV=\frac{TN}{TN+FN}.
$$

Overall accuracy is(TP+TN)/N. TPR asks about decisions among actual positives; PPV asks about actual labels among positive decisions. Reversing the conditioning changes the quantity, even though both formulas contain TP.

For(20,5,5,20), the rates are positive-decision rate .5, TPR .8, FPR .2, PPV .8, NPV .8, and accuracy .8 and prevalence.5. Each is obtained by adding the appropriate cells; no model fitting is needed to compute a rate of this known distribution.

## Map parity definitions to conditional rates

Demographic parity compares positive-decision rates across groups. Equal opportunity compares TPR. For binary decisions, equalized odds compares both TPR and FPR. Predictive parity compares PPV. These concern different conditioning events.

[Hardt, Price and Srebro, §2](https://arxiv.org/pdf/1610.02413) define equalized odds through conditional independence of the decision and group given the target, and equal opportunity through the positive-target condition. Here we use these definitions to calculate gaps, rather than treat any single criterion as a complete normative judgment.

Every signed gap in the experiment is G1 minus G0. The equalized-odds discrepancy is

$$
\max\{|TPR_1-TPR_0|,\ |FPR_1-FPR_0|\}
$$

when all required conditional rates exist. Zero discrepancy means those two particular rates match in the specified distribution. It does not prove causal fairness, label validity or agreement of every other metric. Equalized odds does not in general force equal overall accuracy when group base rates differ.

## A zero denominator is not a zero rate

If a group has no actual positives, TPR is undefined: the conditioning event has probability0. If there are no positive decisions, PPV is undefined. Setting either value to0 invents a conditional probability and can create a false gap or a false appearance of equality.

Scenario4 exposes these cases. One group has no positive labels. Depending on the world, the other group may have no positive decisions or no negative labels. The crossed rate markers and em dash indicate undefined values. A maximum requiring an undefined TPR or FPR is also undefined, not zero.

This differs from nonidentification. With complete data, the distribution can be known while a conditional probability on an empty event remains undefined. With selected data, a perfectly well-defined population conditional rate can instead be unknown because multiple target distributions fit the observations.

## Specify what the observer actually knows

Let S1 denote inclusion in the observed dataset. Its within-group cell distribution is generally

$$
\Pr(Y,\widehat Y\mid G,S=1),
$$

whereas the target is Pr(Y,Yhat|G). Equality of these distributions needs an assumption; it does not follow from using the same names for the variables.

The simulator has a population world and an inclusion mechanism so that it can generate examples. The observer is given the selected table and only the selection information stated by the scenario. The population view and orange comparison bars are simulator oracle truth, not an estimator secretly recovered from the observed data.

Both population groups have public mass100 in these examples. Even knowing those group sizes does not recover the missing cell composition in the counterexamples below.

## Construct two populations with identical observed data

WorldA has both group tables(40,10,10,40). Include every cell with probability1/2. Both observed expected tables are then(20,5,5,20), with TPR.8 and FPR.2.

WorldB leaves G0 unchanged but gives G1 population table(20,40,20,20), with inclusion probabilities(1,1/8,1/4,1). Multiplying each cell by its probability again produces(20,5,5,20). The observer sees exactly the same selected tables as in worldA.

Yet G1 population TPR is20/(20+20)=.5, and FPR is40/(40+20)=2/3. Relative to G0, the true TPR gap is-.3 and the true FPR gap is7/15, approximately.466667. Positive-decision rate is.6 instead of.5, and PPV is1/3 instead of.8.

Under unknown unrestricted selection, these are two observationally equivalent explanations with different target answers. This proves nonidentification: no function of the selected table alone can equal the population answer in both worlds. Collecting more data from the same selected distribution would estimate that distribution more precisely but would not distinguish these two population explanations.

## Invert known positive selection probabilities

Let C_gj be a target cell mass and q_gj its inclusion probability. The observed expected mass is

$$
O_{gj}=q_{gj}C_{gj}.
$$

If q_gj is known independently and positive for every relevant cell, C_gj=O_gj/q_gj. In worldB, dividing(20,5,5,20) by(1,1/8,1/4,1) gives(20,40,20,20), so the corrected TPR is.5.

The available information has changed: worldA and worldB have different known inclusion vectors. They are indistinguishable from selected tables alone, but not from selected tables plus their externally supplied inclusion probabilities. That distinction is essential to a valid identification claim.

This elementary expected-mass identity explains inverse weighting in the experiment. For actual random samples, weighted cell estimates and ratios still have sampling error. A ratio of estimated masses is not generally an exactly unbiased estimator of the corresponding population conditional rate. This lab does not turn a distributional identity into a finite-sample certainty claim.

[Kallus and Zhou](https://arxiv.org/pdf/1806.02887) study how censoring can separate evaluation metrics from target-population fairness. The numerical worlds here are independently constructed small examples; they are not a reproduction of that paper's empirical study or all its theorems.

## Known zero selection does not recover hidden errors

Scenario2 uses known inclusion probabilities(1/2,0,0,1/2). The observed table in each group is(20,0,0,20), which looks like perfect prediction among the included cases.

Consider G1 population(40,10,10,40) and alternative(40,15,5,40). Both have mass100 and the same known inclusion mechanism, and both yield that same observed table. Their hidden false-positive and false-negative masses differ. Their target TPR and FPR therefore differ as well.

Division by the zero inclusion probabilities cannot recover those cells. Dropping them and computing a “corrected” rate would change the target population to the supported subset. The lab instead marks the correction unavailable. In the selected table TPR and FPR are well-defined as1 and0; the issue is that these selected rates do not identify the population rates.

## Representative sampling is a different sufficient assumption

Known cell probabilities are one route, not the only route. Suppose inclusion is independent of(Y,Yhat) conditional on G, and each group has positive inclusion probability. Then the group-specific factor cancels when conditioning on G,S1, giving the same within-group cell proportions as the population.

Scenario3 implements this assumption. G0 inclusion is r/100 and G1 inclusion is r/200 for input r between1 and100. The group sample sizes differ, but every cell within a group has the same probability. All within-group metrics transport without knowing those group fractions.

Because population group masses100 are public in this experiment, the corrected display normalizes each observed group to100. The rate transport itself does not require those total masses. The representative assumption is not established by observing equal metrics, nor is it generally testable from the selected table alone.

Changing r changes expected sample masses and would affect estimation precision in a real dataset. It does not change these population or within-group expected rates. In other scenarios the r control is explicitly ignored.

## Read the table source and comparison plots

The table selector offers selected data, population oracle and assumption-based correction. An unavailable correction displays question marks for cells and undefined readouts rather than a zero-filled matrix. The oracle view is clearly marked even when population quantities are not identified for the observer.

The comparison panel shows the selected and oracle rates for both groups. The last chart shows all seven signed gaps: blue for selected data and orange for population oracle. Missing points denote undefined rates. They do not denote zero disparity.

A positive identification flag records the stated mathematical conditions, not a claim that assumptions were verified from real-world data. Distinguish the simulator's true inclusion probabilities from probabilities supplied to the observer. Also distinguish identification from numerical evaluation: displayed decimals are approximate, while the counterexample calculations can be checked with exact fractions.

## Research extension: Propagate uncertainty in inclusion probabilities

Identification with a known positive inclusion probability is an ideal limit. How sensitive is an inverse-weighted conditional rate to an imperfect probability? Keep the expected-mass model $O_j=q_j C_j$ and assume every relevant $q_j>0$. Suppose externally supplied estimates satisfy a simultaneous deterministic relative-error bound

$$
(1-\eta)q_j\le\widehat q_j\le(1+\eta)q_j,\qquad 0\le\eta<1.
$$

Then corrected masses equal $C_j q_j/\widehat q_j$, so each mass is multiplied by a factor in $[1/(1+\eta),1/(1-\eta)]$. For a binary conditional event whose true numerator proportion is $p$, minimize its corrected rate by shrinking numerator masses and enlarging complement masses, and reverse those choices to maximize it:

$$
\frac{p(1-\eta)}{p(1-\eta)+(1-p)(1+\eta)}
\le\widehat p\le
\frac{p(1+\eta)}{p(1+\eta)+(1-p)(1-\eta)}.
$$

These bounds are sharp when the extremal inclusion estimates are legal probabilities. Equivalently, the corrected odds are within factors $(1-\eta)/(1+\eta)$ and its reciprocal of the true odds when those odds are defined. The reciprocal odds bounds also give an interval for true $p$ by replacing $p$ in the two endpoints with the computed $\widehat p$. This conditional normalization matters: bounding unnormalized cell errors alone is not the final rate analysis.

For an auxiliary population with TP=FN=20 and both inclusion probabilities $1/2$, true TPR is $1/2$:

| Relative error bound $\eta$ | Smallest corrected TPR | Largest corrected TPR |
|---|---:|---:|
| 0 | 1/2 | 1/2 |
| 1/10 | 9/20 | 11/20 |
| 1/4 | 3/8 | 5/8 |

For two group intervals $[L_g,U_g]$, the signed gap belongs to $[L_1-U_0,U_1-L_0]$. If the inclusion bounds are statistical, their simultaneous coverage must first be justified; the deterministic transformation does not create a confidence level. Unknown zero inclusion and error bounds allowing zero remain identification problems outside this result.

The code checks the extremal corrected rates by enumerating relative-error corners, including two nonsymmetric true rates. It changes no native selection scenario and makes no finite-sample unbiasedness claim.

```python
from fractions import Fraction as F
from itertools import product
for p in [F(1,5),F(1,2),F(4,5)]:
    for eta in [F(0),F(1,10),F(1,4)]:
        lo=p*(1-eta)/(p*(1-eta)+(1-p)*(1+eta))
        hi=p*(1+eta)/(p*(1+eta)+(1-p)*(1-eta))
        rates=[]
        for e0,e1 in product([-eta,eta],repeat=2):
            q=F(1,2); estimated=[q*(1+e0),q*(1+e1)]
            assert all(0<v<=1 for v in estimated)
            corrected=[p*q/estimated[0],(1-p)*q/estimated[1]]
            hat=corrected[0]/sum(corrected)
            reverse_lo=hat*(1-eta)/(hat*(1-eta)+(1-hat)*(1+eta))
            reverse_hi=hat*(1+eta)/(hat*(1+eta)+(1-hat)*(1-eta))
            assert reverse_lo<=p<=reverse_hi
            rates.append(hat)
        assert min(rates)==lo and max(rates)==hi
        if p==F(1,2):print(eta,lo,hi)
```

## Exercises and worked solutions

1. In(20,5,5,20), compute TPR and PPV. Both are.8 here, but their denominators are actual positives and positive decisions respectively; equality in this symmetric table does not make the definitions interchangeable.
2. For worldB G1, compute the positive-decision rate and PPV. They are60/100=.6 and20/60=1/3.
3. Can the unknown-selection observer choose the correct population TPR from the identical observed tables? No. WorldsA andB provide the same observations with different target TPRs, which is a constructive nonidentification proof.
4. Correct selected FP mass5 when its known inclusion probability is1/8. The expected population FP mass is40. A real sample's weighted estimate would still be random.
5. If q_FP=0 and observed FP0, must population FP0? No. Every nonnegative hidden FP mass would generate zero selected FP mass before other constraints; the paired world example preserves the known group total as well.
6. If a group has no positive decisions, what is PPV? Undefined, because its conditioning event has zero mass.
7. Is known cell selection necessary for all identification? No. Stated positive representative sampling within groups is another sufficient route for group-conditional rates. It remains an assumption, not a consequence of observed parity.

## Research connection

A research report should state the target population, observation mechanism, label meaning, group definition, metric conditioning events and estimation uncertainty. A matched observed metric answers a statistical question about its evaluation distribution. Moving to a new population, causal intervention or normative conclusion needs the corresponding assumptions and evidence.

The next lesson studies compatibility among metrics. Even with a perfectly known target distribution, different parity constraints can conflict because they condition on different events.
