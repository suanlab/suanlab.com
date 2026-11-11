## Why the rules are part of the problem

Imagine allocating one shared laboratory instrument. Three researchers know how valuable access is to them, but the allocator does not. Asking everyone for a number creates a new problem: a researcher can submit a number that differs from their actual value. The allocation rule and the payment rule together determine whether exaggerating or understating is worthwhile.

A game asks what participants do under given rules. Mechanism design also asks which rules produce desirable behavior. Begin with a small direct mechanism: each participant submits a report about a private value. The word “direct” describes the message format; it does not mean the report is automatically honest.

The lab uses three agents, identifiers0,1,2, true values v_i and submitted reports b_i in the integer set {0,…,10}. Utility is quasilinear: received benefit minus payment. There are no budget constraints, risk aversion, collusion or interdependent values. Values are hypothetical ground truth available to the teaching evaluator. The allocation/payment functions never read them.

## Read the arrows in the transfer diagram

The upper boxes contain both v_i and b_i, but only b_i flows into the rule. The rule returns an allocation x_i and payment p_i. True values enter afterward, to calculate

$$
u_i(v_i;b)=v_i x_i(b)-p_i(b).
$$

Here u_i is utility, not a probability; positive means better for that agent. In an auction, x_i is1 for the winner and0 otherwise. In a public project, all agents have x_i=1 if it is built, because everyone receives their own benefit from the same project. That public allocation is not three copies of a privately sold item.

Transfers move money between participants and the operator. They do not themselves consume resources. The operator's net is revenue minus real resource cost. Therefore

$$
\sum_i u_i+\left(\sum_i p_i-\text{resource cost}\right)
=\sum_i v_i x_i-\text{resource cost}=W.
$$

W is true welfare. The accounting identity remains valid when an agent loses money, when the operator runs a deficit, and when reports induce a bad decision.

## One allocation rule, two auction payments

Both auction options award one item to the highest report meeting a fixed reserve r. Ties go to the lowest identifier. If every report is below r, there is no sale and all payments are0. A zero bid is an actual bid; with reserve0, three zero bids award the item to agent0.

First price charges the winner their own report. Second price charges the winner the larger of the reserve and the highest OTHER report. All losers pay0. A report-independent reserve and fixed identifier priority prevent a hidden change in the rules during a comparison.

Consider true values and reports (9,5,2), reserve0. Agent0 wins under both rules. First price charges9, leaving utility0. Lowering its report to5 still wins the tie and yields utility4. Against these fixed opponents the unique first-price maximizing report is5.

Second price charges5, leaving utility4 at report9. Every report from5 through10 wins and pays5, so all six reports are optimal. Truth is optimal without being the unique optimum. The utility table and the diagram preserve all these ties instead of marking one arbitrarily selected “best” bid.

## Three different quantifiers

A conditional best response maximizes utility while holding the other two reports fixed:

$$
B_i(b_{-i};v_i)=\arg\max_{z\in\{0,\ldots,10\}}u_i(v_i;z,b_{-i}).
$$

The blue discrete curve shows exactly these eleven comparisons. The orange horizontal level is the utility of reporting the selected true value against those same opponents. The current-report dot can lie below that level. The truth dot need not be the only maximum.

Truth is a weakly dominant report for a particular true type when it does at least as well as every alternative report against EVERY opponent-report profile. The lab fixes the selected agent and its true value, then checks121 opponent profiles times11 own reports. For each cell it computes

$$
\Delta_i(b_{-i};v_i)=\max_z u_i(v_i;z,b_{-i})-u_i(v_i;v_i,b_{-i})\ge0.
$$

Orange cells have positive Delta. Blue cells have0. The outlined cell marks the entered opponents. A displayed witness gives opponents and a profitable alternative report when any exists. This is a finite exhaustive check for one true type, not a statistical estimate.

DSIC requires the same inequality for EVERY agent and EVERY possible true type as well. A zero maximum for one type does not establish that mechanism-wide statement. First-price type0 can pass the selected-type check: winning at a positive price cannot help someone who values the item at0. An unreachable reserve can also make all reports equivalent. Neither fact makes first price DSIC over the full domain.

Bayesian incentive compatibility instead compares expected utilities over a specified distribution of other types, with truthful behavior in the comparison. No prior distribution is supplied here; a single opponent profile or this selected-type heatmap is not a BIC calculation.

## Why truthful second-price reports work

Fix the other reports. Let t be the larger of the reserve and their highest report. Winning costs t independently of one's own winning report. If v_i>t, winning is better than losing and truth secures it. If v_i<t, winning gives negative utility and truth avoids it. At equality either outcome has utility0, so fixed tie priority does not invalidate weak truthfulness.

This argument applies to every value and every opponent report in the stated private-value, quasilinear domain. It proves an all-type claim for second price under these rules. It does not say the current entered report is truthful, that winning always helps, or that arbitrary auction formats share the property.

For example, change true values to (4,5,2) but reports to (9,5,2). Agent0 wins and pays5, so its utility is−1. Reporting4 would lose and give0. Individual rationality is a guarantee for truthful reporting under the model, not protection against an arbitrary overbid.

## A reserve is not automatically a real cost

The auction seller's own value for keeping the item is explicitly0. Reserve r is a selling policy, not a resource cost in W. Thus the unrestricted welfare optimum is max_i v_i. With truthful values/reports (4,3,2) and reserve5, the second-price mechanism makes no sale: actual welfare0, unrestricted optimum4, gap4.

Truthful incentives survive this reserve, but unrestricted allocative efficiency need not. If a different model gives the seller a genuine opportunity value, that value must enter the welfare objective consistently. Silently subtracting the reserve would answer a different question.

## A public project and pivotal payments

Now consider building one shared resource at known real cost C. Everyone benefits if it is built; nobody benefits if it is not. The provider's cost is known, not a strategic type elicited by this lab. Select build when the total reports meet C, with equality building:

$$
x(b)=1\{\sum_i b_i\ge C\}.
$$

For agent i write s_{-i}=sum of other reports. Without i, the maximum reported welfare of the other agents plus provider is h_i=max(0,s_{-i}−C). Under the actual chosen outcome it is (s_{-i}−C)x(b). The Clarke pivot transfer is their difference:

$$
p_i(b)=\max(0,s_{-i}-C)-(s_{-i}-C)x(b)
=\begin{cases}\max(0,C-s_{-i}),&x(b)=1,\\0,&x(b)=0.\end{cases}
$$

The no-build expression is0 because nonnegative own reports imply s_{-i}<C whenever total reports fail the build threshold. Removing an agent removes its benefit; it does not remove the project's real cost.

To see the incentive, hold other reports fixed. The selected agent's utility is

$$
u_i=v_i x-p_i=(v_i+s_{-i}-C)x-h_i.
$$

The last term does not depend on its report. Reporting v_i makes the allocation rule maximize the first term over build/no-build. Truth therefore weakly dominates all other reports. This is an exact two-outcome argument, not an appeal to a simulated learning curve.

Under truth, a pivotal agent's required payment is at most its value whenever build is selected; a nonpivotal agent pays0. Together with no-build utility0 this gives truthful individual rationality. When everyone is truthful the chosen outcome maximizes true welfare max(0,sum_i v_i−C). These properties require the stated valuation/cost model.

## Efficiency can coexist with an operator deficit

Load the third example: values and reports (6,6,6), C=10. The total18 justifies building. For every agent the others already report12, so removing that agent still leads to build. Each pivot payment is0.

The three utilities are6,6,6; revenue is0; real cost is10; operator net is−10. True welfare is18−10=8. Adding utilities18 and operator net−10 reproduces8. A zero payment means “not pivotal to this decision,” not “no benefit” or “no resource cost.”

This example separates truthful incentives, efficient choice, individual rationality and cost recovery. It is not a general impossibility theorem about every public-good mechanism. Such theorems have additional type-domain and institutional assumptions.

## Try the boundaries before generalizing

Set project cost0: build always occurs and payments are0. Set cost30 and reports (10,10,10): equality builds and each payment is10. Move one report down to9: no build. A high report can also trigger a project whose TRUE benefits do not justify its cost. For values (0,0,0), reports (10,10,10), C=30, true welfare is−30, despite reported surplus0.

Changing the inspected agent changes the utility curve, pivot decomposition and type check, but must leave current allocation/payments unchanged. Changing hidden true values can change utilities/welfare and the truth reference, but must likewise leave the report-based current decision unchanged. Use these as conceptual checks of the arrows.

## Research extension: derive payments from monotone allocation

For an auxiliary continuous private type $v\in[0,\bar v]$, fix opponents and write expected allocation $q(v)$ and payment p(v), with quasilinear utility $U(v)=vq(v)-p(v)$. DSIC inequalities for v>w imply

$$q(w)(v-w)\le U(v)-U(w)\le q(v)(v-w).
$$

Adding the two report inequalities first proves q is nondecreasing. Summing these bounds over increasingly fine partitions gives the envelope $U(v)=U(0)+\int_0^v q(z)dz$. Therefore

$$p(v)=vq(v)-\int_0^v q(z)dz+p(0).
$$

Conversely, a monotone q with this payment formula makes truth optimal: the gain over a lower report w is $\int_w^v[q(z)-q(w)]dz\ge0$, and a higher report has the analogous reversed integral. For probabilistic q this is truthfulness in expected utility; it is not automatically universal truthfulness of every random realization.

For a threshold allocation $q(v)=1\{v\ge\theta\}$ and normalization p(0)=0, winners pay theta and losers pay zero. At true v5:

| Threshold | Allocation | Payment | Utility |
|---|---:|---:|---:|
| 0 | 1 | 0 | 5 |
| 3 | 1 | 3 | 2 |
| 7 | 0 | 0 | 0 |

```python
from fractions import Fraction as F
for theta in [F(0),F(3),F(7)]:
    for j in range(21):
        v=F(j,2); truthful=v*int(v>=theta)-(theta if v>=theta else 0)
        for k in range(21):
            report=F(k,2); q=int(report>=theta)
            integral=max(F(0),report-theta)
            payment=report*q-integral
            assert payment==(theta if q else 0)
            assert truthful>=v*q-payment
    v=F(5); q=int(v>=theta); payment=theta if q else 0
    print(theta,q,payment,v*q-payment)
values=[4,4,4]; cost=10
payments=[max(0,cost-sum(values)+v) for v in values]
assert sum(values)>=cost and sum(payments)==6<cost
```

This continuous-type characterization explains the threshold logic; the native integer report grid remains a finite mechanism with its own exhaustive checks. Its discrete DSIC inequalities alone do not identify every continuous payment formula. Neither the envelope nor truthful incentives guarantee public cost recovery.

## Exercises and worked solutions

1. Under second price, true/report vector (5,5,0), reserve0, inspect agent1. Which reports maximize its utility? Agent0 wins a tie at5. Reporting above5 wins at price5 with utility0, and reporting5 or below loses with utility0. Every report0..10 is optimal; truth need not win to be optimal.
2. For a project with true/reported values (8,3,1), C=10, find all payments and reconcile welfare. Payments are6,1,0; utilities are2,2,1; revenue7, operator net−3, welfare2. Utilities5 plus net−3 equal2. Charging every agent their full benefit would be a different mechanism.
3. Does a positive improvement from the current report imply truth is not dominant? No. Under second price with value4 and report9 against (5,2), switching to truth improves utility from−1 to0. Truth remains weakly dominant; the current misreport is the problem. The conditional improvement and the maximum truth violation answer different questions.

## Connection to AI research

Strategic data contributors, model providers or evaluators may change their messages in response to a reward rule. Distinguish hidden quality/value from submitted scores, define the feasible outcomes, specify who pays and who bears real costs, then quantify the deviations the guarantee covers. Optimization accuracy also matters: replacing exact welfare maximization with an arbitrary approximation does not automatically preserve the same payment argument.

This finite lab checks its own allocation/payment implementation. It does not validate an arbitrary AI reward pipeline, resistance to coalitions, private information protection or computational tractability on large outcome spaces. Those require separate models and evidence.

For standard background on private-value utility and second-price incentives, see [Roughgarden, Lecture2](https://timroughgarden.org/f13/l/l2.pdf). For the general reported-welfare and externality-payment framework, see [Lecture7](https://timroughgarden.org/f13/l/l7.pdf). The numerical fixtures, finite-domain heatmap and public-cost accounting above are derived for this lab.
