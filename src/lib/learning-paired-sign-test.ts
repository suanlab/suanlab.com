/** Exact one-sided binomial tail for independent fair signs, without ties. */
export function calculatePairedSignTail(total: number, positive: number) {
  if (!Number.isInteger(total) || !Number.isInteger(positive) || total < 1 || total > 50 || positive < 0 || positive > total) {
    throw new RangeError('Expected 1–50 non-tied independent pairs and a valid positive count');
  }
  const probabilities = [2 ** -total];
  for (let k = 1; k <= total; k++) probabilities.push(probabilities[k - 1] * (total - k + 1) / k);
  return {
    probabilities,
    tail: Math.min(1, probabilities.slice(positive).reduce((sum, value) => sum + value, 0)),
  };
}
