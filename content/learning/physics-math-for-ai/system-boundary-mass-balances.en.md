## A conservation claim needs an accounting boundary

A mass reading inside one container can decrease while mass conservation remains perfectly satisfied. The missing mass may have crossed the chosen boundary. Before treating a residual as a failure of physics, specify what belongs to the system, what is outside, and which transfers cross between them.

This lesson uses three mass stores: A, B, and a tracked environment E. Their values are in kilograms. The diagrams are schematic; distances and arrow widths do not encode lengths or mass quantities. You can select A, A+B, or A+B+E as the accounting region while keeping the physical evolution unchanged.

The word closed is used specifically for mass crossing in this model. It does not establish thermal isolation or the absence of energy exchange. Those require additional physical quantities and boundaries, which are not modeled here.

## Declare a discrete physical operation rather than an unspecified solver

Once per minute, exchange fraction p of the existing A and B masses simultaneously. Then transfer fraction ell of the newly mixed B mass into E. The sequence matters:

$$
A_{n+1}=(1-p)A_n+pB_n,\qquad
\widetilde B_n=pA_n+(1-p)B_n,
$$

$$
B_{n+1}=(1-\ell)\widetilde B_n,\qquad
E_{n+1}=E_n+\ell\widetilde B_n.
$$

Both fractions lie between zero and one. Initially E is zero. Leaking before the exchange would define a different operation and generally produce different values. The laboratory implements the declared exchange-then-leak order explicitly.

This is a batch-transfer model with a stated one-minute operation. It is not presented as the exact solution of a continuous diffusion equation or as a numerical time-step approximation that can be refined without changing its interpretation. For example, full exchange can swap the two masses each minute when there is no leakage. That behavior is valid for the declared operation but should not be silently identified with ordinary symmetric continuous diffusion.

## Prove nonnegative mass and expanded conservation

Each updated store is a sum of nonnegative old masses multiplied by nonnegative fractions. Starting from nonnegative masses therefore preserves nonnegativity. Adding the three update equations gives

$$
A_{n+1}+B_{n+1}+E_{n+1}=A_n+B_n+E_n.
$$

The leakage term is not destroyed; it enters the tracked environment. A column-stochastic transition matrix provides another view of the same finite model. Each column distributes one unit of old mass across possible destinations with total weight one.

This argument proves conservation within the declared model. Applying the model to a real apparatus also requires evidence that the compartments and transfer law are appropriate. A correct accounting identity does not establish that every relevant physical process has been included.

## Derive the boundary balance for A

For the boundary containing A alone, incoming mass in one operation is p times the old B mass and outgoing mass is p times the old A mass. Thus

$$
A_{n+1}-A_n=pB_n-pA_n.
$$

The leakage from mixed B to the environment is not a direct crossing of this A-only boundary. It can affect future incoming transfers by changing B, but it should not be subtracted again from A in the same balance. Following the transfer paths prevents this double counting.

Summing the interval balances yields the cumulative relation

$$
M_n-M_0=I_{0:n}-O_{0:n}.
$$

At minute zero cumulative transfers are zero, while a last interval has not yet occurred. The interface displays last-interval values as unavailable rather than inventing an interval with a measured zero flow.

## Enlarge the boundary without changing the trajectory

For A+B, the transfers between A and B become internal. Their contributions cancel, leaving only the transfer to E as an external outflow:

$$
(A_{n+1}+B_{n+1})-(A_n+B_n)=-\ell\widetilde B_n.
$$

For A+B+E, that transfer also becomes internal, and the net boundary crossing is zero. The environment mass stays in the state and in the accounting. Expanding the region does not erase earlier transfers; it changes which transfers are counted as external.

The three guided examples keep the initial masses and both fractions fixed. Only the accounting boundary changes. This lets you compare alternative descriptions of the same evolution instead of accidentally comparing three different physical models.

## Work the first-minute examples

Start with A equal to eight kilograms and B equal to two. Let p be 0.2 and ell be 0.1. The exchange gives A equal to 6.8 and mixed B equal to 3.2. Leakage transfers 0.32 to E, leaving B equal to 2.88 kilograms.

For A alone, mass change is minus 1.2 kilograms. Incoming mass 0.4 minus outgoing mass 1.6 gives exactly that change. For A+B, stored mass is 9.68 and change is minus 0.32, accounted for by the environmental transfer. For A+B+E, stored mass is ten and change is zero. All three balances are valid simultaneously.

The blue history shows stored mass in the selected boundary. Orange reconstructs it from initial storage plus cumulative incoming minus outgoing mass. Green shows the expanded total including E. The first two agree because of the balance identity; the green total helps locate mass that left a smaller region.

## Separate structural closure from zero net crossing

A structurally closed boundary has no possible mass crossing under the specified nonzero transfer links. A single trajectory with constant stored mass does not prove that property.

If A and B each contain five kilograms, p is 0.2, and there is no leakage, A receives one kilogram and sends one kilogram during the first operation. Its mass is unchanged even though its boundary is open to exchange. Gross incoming and outgoing amounts are nonzero while their difference is zero.

Likewise, a modeled leakage link can exist while a particular initial state supplies no mass to it. With p equal to zero and B initially empty, no mass reaches E even if the leakage fraction is positive. That realized zero flow does not remove the possible link from the model. The closure readout refers to the parameterized transfer graph, while the flow table refers to the queried history.

## Preserve exact arithmetic on the supported control grid

Inputs use integer initial kilogram values and integer percentage codes. Each operation can therefore be represented with integer mass numerators over a common denominator multiplied by ten thousand per step. At step n the denominator is ten thousand to the power n.

The implementation computes stored-mass change and cumulative boundary transfers separately, then checks their residual using integer arithmetic. It also checks nonnegative mass and the expanded total at every step. Decimal readouts and plot positions remain numerical approximations. A visually overlapping line is not used as the proof of an exact balance.

Independent tests compare the implementation against rational matrix powers and separately accumulated crossings. These checks include zero masses, zero transfers, full exchange, full leakage, and long histories. They assess the finite transfer implementation rather than certify an arbitrary physical simulator.

## Exercises with worked solutions

1. **Exercise:** Why should leakage not be subtracted directly from the A-only balance? **Solution:** The leakage edge connects B to E. A has already accounted for its own exchange crossing; subtracting B's leakage from A would count a different boundary's outflow again.
2. **Exercise:** In the guide, where is the 0.32 kilograms missing from A+B? **Solution:** It is stored in E. Adding E restores the expanded total of ten kilograms.
3. **Exercise:** Can zero mass change coexist with one kilogram incoming and one kilogram outgoing? **Solution:** Yes. Storage change is their difference, not their sum. This does not establish structural closure.
4. **Exercise:** With full exchange and full leakage, what follows from initial masses eight and two? **Solution:** After the first operation A is two, B is zero, and E is eight. After the second operation E contains ten and A,B are zero.
5. **Exercise:** Does closed mass accounting prove the apparatus cannot exchange heat? **Solution:** No. Mass closure and energy or thermal isolation are different claims requiring different modeled transfers.
6. **Exercise:** Does a correct total-mass constraint determine the exchange fraction? **Solution:** Not without additional information. With no leakage, every permitted exchange fraction preserves the expanded total. The conservation constraint alone does not uniquely fix the internal dynamics.

## Source and research connection

The [MIT control-volume mass balance discussion](https://web.mit.edu/16.unified/www/SPRING/thermodynamics/notes/node18.html) explains the distinction between storage change and boundary crossings. The three-store discrete model and exact arithmetic above are derived directly for this lesson. In physics-informed AI, a conservation penalty can be useful, but its boundary terms and state definition must match the intended physical system. A small residual for an incomplete accounting region is not automatically evidence of a correct learned model.
