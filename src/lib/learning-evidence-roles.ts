/** Simultaneous Hoeffding radius for a fixed family of bounded-loss models. */
export function calculateHoeffdingEvidence(samples: number, candidates: number, delta: number, empiricalRisk: number) {
  if (!Number.isInteger(samples) || samples < 1 || !Number.isInteger(candidates) || candidates < 1 ||
      ![delta, empiricalRisk].every(Number.isFinite) || delta <= 0 || delta >= 1 || empiricalRisk < 0 || empiricalRisk > 1) {
    throw new RangeError('Expected positive sample and candidate counts, delta in (0,1), and risk in [0,1]');
  }
  const radius = Math.sqrt((Math.log(2) + Math.log(candidates) - Math.log(delta)) / (2 * samples));
  return { radius, lower: Math.max(0, empiricalRisk - radius), upper: Math.min(1, empiricalRisk + radius) };
}
