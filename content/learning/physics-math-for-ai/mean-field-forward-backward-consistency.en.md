## A small individual and a changing population

Suppose a large population chooses between two destinations. Each person cares about congestion, travel and location quality. One person is too small to change the population distribution appreciably, yet everyone's choices together determine it. A mean-field game couples these two viewpoints: optimize while freezing the population, then require the resulting population flow to match what was frozen.

This unit minimizes costs, so smaller is better. The interactive model has two locations, two dates and one movement decision. It is a nonatomic model: individual agents have negligible mass. It is not a two-player Markov game, a sampled particle simulation or an assertion that replacing every many-agent system by its average is justified.

Initial mass at location1 is mu, with1−mu at location0. Agents have the same cost rules, while their starting locations can differ. The question is both an individual control problem and a distributional consistency problem.

## Terminal costs depend on the distribution

At a hypothesized terminal mass m in location1, define

$$
c_0(m)=3+a(1-m)-\frac b2,\qquad
c_1(m)=3+am+\frac b2.
$$

Moving between locations costs k≥0; staying costs0. The teaching parameters are explicit rather than fitted to a real city. Positive a makes a crowded destination more expensive, negative a gives a coordination benefit, and b tilts the relative location cost. The common baseline3 keeps costs positive over the stated controls without affecting private comparisons.

The cost difference is

$$
D(m)=c_1(m)-c_0(m)=a(2m-1)+b.
$$

When one infinitesimal agent compares destinations, m is held fixed. A planner changing a positive fraction of the population cannot make that assumption.

## Solve backward for each starting location

Let alpha be the probability that an agent starting at0 ends at1, and beta the corresponding probability for an agent starting at1. Their action-cost rows, ordered by destination0 or1, are

$$
(c_0,c_1+k)\quad\text{and}\quad(c_0+k,c_1).
$$

The backward values are the minima of those rows. The complete optimal probability sets are

$$
\alpha\in A(m)=
\begin{cases}\{1\}&D<-k\\{[0,1]}&D=-k\\\{0\}&D>-k,\end{cases}
\qquad
\beta\in B(m)=
\begin{cases}\{1\}&D<k\\{[0,1]}&D=k\\\{0\}&D>k.\end{cases}
$$

At a tie, every mixture is optimal. Choosing one arbitrary action would discard possible equilibria. Notice that beta is a probability of ending at1, not a probability of moving: an initial-location1 agent stays when beta=1.

Our feedback convention requires optimality in both starting states. At mu0 or1, an unused cohort still has a specified optimal-policy condition. The interface separately reports the population-weighted improvement, which can be zero while an unused cohort's policy is suboptimal. This convention is stronger than evaluating only the initially occupied states.

## Send the population forward

Given alpha and beta, terminal mass is

$$
m'=(1-\mu)\alpha+\mu\beta.
$$

For a frozen m, optimal responses can generate any mass in the interval

$$
\mathcal B(m)=(1-\mu)A(m)+\mu B(m).
$$

Equilibrium requires m∈B(m). The first graph plots frozen mass horizontally and possible response mass vertically. Its diagonal is m'=m. Vertical tie segments and rectangular response regions are actual sets of possible flows; they are not lines connecting a discontinuity in a time series.

The entered query policy always generates its own forward mass. We then evaluate that policy's private optimality at that mass. A query mass may belong to the equilibrium mass set even when its actual alpha/beta are not optimal. The graph therefore reports mass membership and policy equilibrium separately.

## Find all solutions without relying on the query grid

The affine difference D changes response regimes only at D=−k or D=k. Split[0,1] at these exact rational thresholds. On each open interval, the response ranges are constant; intersect their possible flow interval with that mass interval. Check the threshold points separately, then merge adjoining solution intervals.

All endpoints remain valid because the best-response correspondence has a closed graph and the population map is continuous. The implementation checks those endpoints exactly. This produces isolated roots, multiple components or whole intervals without restricting solutions to masses reachable by the input sliders.

For example, a2, b0, k0.01 and mu0.8 give equilibrium mass201/400=0.5025. Its consistent policy is alpha0, beta201/320. The exact solution survives even though a hundredth-mass grid would miss it and the policy sliders cannot represent that particular beta.

## A selected mass still permits a policy set

At a selected equilibrium mass m*, consistent optimal policies satisfy

$$
(\alpha,\beta)\in A(m^*)\times B(m^*),\qquad
(1-\mu)\alpha+\mu\beta=m^*.
$$

This intersects a response rectangle with one flow line. The result is a point or segment. The lower diagram preserves that entire slice, including free policies of a zero-weight cohort when they are optimal. Selecting a mass does not automatically select a unique policy.

When a0, b0 and k0, every terminal cost equals3. Every policy is privately optimal and every mass in[0,1] is an equilibrium. At mu1/2 and selected mass1/2, the consistent-policy slice is alpha+beta=1, not one arbitrarily chosen pair. Changing the position control explores the whole equilibrium interval; on an isolated mass it changes nothing.

## Worked example: five distinct equilibria

Take a=−1, b=0, k=1/5 and mu=1/2. Then D=1−2m. Below2/5, both cohorts prefer ending at0, allowing the endpoint equilibrium m0. Above3/5, both prefer1, allowing endpoint m1. Between the thresholds, each cohort stays, yielding m1/2.

At m2/5, the initial-location1 cohort is indifferent, while alpha0. Consistency requires beta4/5. At m3/5, beta1 and alpha1/5. All equilibrium masses are therefore

$$
\left\{0,\frac25,\frac12,\frac35,1\right\}.
$$

The two threshold equilibria need mixing. A solver that always replaces an indifferent probability interval by an endpoint can lose them. A solver that finds only one root cannot claim the equilibrium is unique.

## Aggregate welfare counts distributional externalities

Actual population switching is

$$
M=(1-\mu)\alpha+\mu(1-\beta).
$$

The query policy's social cost is its population-weighted expected cost, equivalently

$$
C_{\rm query}=(1-m)c_0(m)+mc_1(m)+kM.
$$

For a given terminal mass, minimal switching is |m−mu|. It is attainable by sending only the necessary net movement in one direction. The planner can therefore minimize

$$
W(m)=3+a\bigl((1-m)^2+m^2\bigr)+b(m-\tfrac12)+k|m-\mu|.
$$

We retain all global minimizing points and flat intervals. Each side of mu is quadratic or linear: inspect its interior critical point when convex, endpoints when concave, and any constant segment when flat.

[Carmona, Delarue and Lachapelle](https://arxiv.org/abs/1210.5771) distinguish mean-field-game optimization from distribution-dependent control. Here the difference is explicit: away from the movement kink, the planner derivative contains2a(2m−1)+b, while private choice uses a(2m−1)+b. The planner accounts for the cost change experienced by the population.

## Worked example: private equilibrium need not be socially optimal

Use a1, b3/5, k1/5 and mu4/5. The unique equilibrium mass is3/10. The policy alpha0, beta3/8 produces it. At this mass c0=17/5 and c1=18/5. An initial-location1 agent is indifferent between staying for18/5 and moving for17/5+1/5; the other cohort strictly prefers staying.

Switching mass is1/2, so social cost is89/25=3.56. The planner instead chooses mass2/5, with cost177/50=3.54. The gap is1/50. For m<mu, solving4m−2+3/5−1/5=0 gives that planner mass. Solving the private threshold2m−1+3/5=1/5 gives the different mass3/10.

The gap is an externality effect in this specified model. It is not a general empirical efficiency estimate or a claim that every equilibrium of every MFG is inefficient.

## A correct mass can hide unnecessary round trips

Take a1, b0, k1/5 and mu1/2. Query alpha=beta=1/2. It generates mass1/2, which is an equilibrium mass. But both cohorts could stay and avoid movement cost: each has private improvement1/10. The query policy is not an equilibrium.

Its switching mass is1/2, while minimal net switching is0. Actual social cost exceeds W(1/2) by1/10. The blue curve shows W, but the blue query dot shows actual policy cost and correctly sits above it. Moving the dot onto the curve would erase a real cost.

## Uniqueness and numerical convergence are separate

For this finite model, equilibrium masses satisfy the scalar variational condition for

$$
J(m)=a(m^2-m)+bm+k|m-\mu|,\qquad 0\le m\le1.
$$

Its smooth derivative is D(m); movement contributes the absolute-value subgradient. Positive a makes J strictly convex, giving a unique equilibrium mass. This does not generally give a unique policy, and J is a potential, not the social objective W. For negative a, stationary equilibrium conditions can have multiple solutions; minimizing J globally would omit some of them.

A deterministic best-response selector need not converge even in the unique case. With a1,b0,k1/5,mu4/5, freezing m0 makes everyone choose1, then freezing m1 makes everyone choose0. Iterating gives0→1→0→1, despite a valid mixed equilibrium at3/5. This is a diagnostic counterexample, not an implemented physical trajectory. Existence of a set-valued fixed point does not certify a chosen update rule.

## The continuous forward–backward system

A continuous MFG has an individual value u(t,x) and population density m(t,x). One standard separable diffusive form is

$$
-\partial_tu-\nu\Delta u+H_0(x,\nabla u)=F(x,m),
$$

$$
\partial_tm-\nu\Delta m-\nabla\cdot\bigl(mD_pH_0(x,\nabla u)\bigr)=0,
\qquad m(0)=m_0,\quad u(T,x)=G(x,m_T).
$$

Under this Hamiltonian sign convention, the optimal drift is−D_pH0. The value problem runs backward from terminal cost; the density runs forward from its initial distribution. On a periodic domain, suitable regularity allows divergence terms to integrate away, preserving total mass. Other domains need appropriate boundary or decay conditions.

[Cardaliaguet's notes, §1](https://www.ceremade.dauphine.fr/~cardaliaguet/MFG20130420.pdf) present this coupling. Convexity, regularity, boundary conditions and monotonicity enter its analysis. The two-date lab exhibits the same freeze-optimize-regenerate-consistency logic, but it is not a numerical PDE solver and does not prove a PDE existence or uniqueness theorem.

For our affine terminal costs, the monotonicity pairing between two populations is2a(m−n)^2. Its sign helps explain congestion versus coordination. Transferring a continuum uniqueness result still requires all of that result's assumptions; a positive slider alone is not a general PDE certificate.

## What a research claim would need

A finite-player system gives each individual nonzero influence. Establishing an approximate Nash result from a nonatomic solution requires a specified finite-player model, information structure, regularity and an error argument. No such rate follows merely from a crowd-looking diagram.

Likewise, equilibrium computation, solver stability, mean-field control and physical mean-field closure are distinct tasks. The sources provide mathematical frameworks, not a declaration of today's open-problem status. The interactive certificates concern the complete rational solution set of the stated small game and its separate planner problem.

## Exercises and explained solutions

Exercise1: at a1,b3/5,k1/5,mu4/5, verify private indifference for cohort1 at m3/10 and derive the planner's m2/5.

Solution: D(3/10)=1/5, so c1=c0+k. On m<mu, the planner derivative is4m−2+3/5−1/5; setting it to zero gives2/5. The two equations differ because aggregate cost depends on the distribution.

Exercise2: why can mass membership and population-weighted zero gain both be weaker than all-state policy optimality?

Solution: round trips can generate an equilibrium mass with costly nonoptimal actions. If mu0, an unused cohort can have a positive private gain that receives zero population weight. The lab keeps those tests separate rather than inferring one from the other.

Exercise3: what must be retained at a tie, and why is an oscillating best-response sequence not a nonexistence proof?

Solution: keep the full response interval and its flow-consistent policy slice. A deterministic tie selector can omit the mixture needed for a fixed point. The exact mass decomposition can certify equilibria independently of whether that selected iteration converges.
