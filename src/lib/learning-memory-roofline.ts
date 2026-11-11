/** Arithmetic and transfer ceilings with ideal overlap. All rates use decimal SI units. */
export function calculateMemoryRoofline(operations: number, bytes: number, peak: number, bandwidth: number) {
  if (![operations, bytes, peak, bandwidth].every(value => Number.isFinite(value) && value > 0)) {
    throw new RangeError('Expected positive operations, bytes and rates');
  }
  const intensity = operations / bytes;
  const ridge = peak / bandwidth;
  const computeTime = operations / peak;
  const transferTime = bytes / bandwidth;
  const memoryCeiling = bandwidth * intensity;
  const balanced = Math.abs(intensity - ridge) <= Number.EPSILON * Math.max(intensity, ridge) * 4;
  return {
    intensity,
    ridge,
    computeTime,
    transferTime,
    runtimeFloor: Math.max(computeTime, transferTime),
    throughputCeiling: Math.min(peak, memoryCeiling),
    bottleneck: balanced ? 'balanced' as const : intensity < ridge ? 'memory' as const : 'compute' as const,
  };
}
