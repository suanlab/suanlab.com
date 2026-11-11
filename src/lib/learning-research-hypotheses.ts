/** Descriptive paired effects for two models measured on the same episodes. */
export function calculateResearchEffect(baseline: readonly number[], candidate: readonly number[]) {
  if (baseline.length < 2 || baseline.length !== candidate.length ||
      ![...baseline, ...candidate].every(value => Number.isFinite(value) && value >= 0)) {
    throw new RangeError('Expected at least two matched pairs of finite nonnegative losses');
  }
  const differences = baseline.map((value, index) => value - candidate[index]);
  const average = differences.reduce((sum, value) => sum + value, 0) / differences.length;
  const variance = differences.reduce((sum, value) => sum + (value - average) ** 2, 0) / (differences.length - 1);
  return { differences, average, standardError: Math.sqrt(variance / differences.length) };
}
