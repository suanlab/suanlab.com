/** Synthetic learning curves used to isolate the effect of a matched budget. */
export function calculateBaselineBudget(budget: number) {
  if (!Number.isFinite(budget) || budget <= 0) throw new RangeError('Expected a positive budget');
  return { first: .3 + .4 / Math.sqrt(budget), second: .2 + .8 / Math.sqrt(budget) };
}
