/** Two-mode diagonal forward map with the true parameter vector (1, 1). */
export function calculateSingularModeInverse(strength: number, error: number, direction: 'strong' | 'weak') {
  if (![strength, error].every(Number.isFinite) || strength < 0 || strength > 1) {
    throw new RangeError('Expected strength in [0, 1] and a finite perturbation');
  }
  const firstReading = direction === 'strong' ? 1 + error : 1;
  const secondReading = direction === 'weak' ? strength + error : strength;
  const state = strength > 0 ? 'unique' : secondReading === 0 ? 'nonunique' : 'inconsistent';
  return {
    firstReading,
    secondReading,
    condition: strength === 0 ? null : 1 / strength,
    firstEstimate: state === 'inconsistent' ? null : firstReading,
    secondEstimate: strength === 0 ? null : 1 + (direction === 'weak' ? error / strength : 0),
    estimateError: strength === 0 ? null : Math.abs(direction === 'weak' ? error / strength : error),
    state,
  } as const;
}
