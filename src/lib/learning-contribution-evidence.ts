/** A matched 2×2 comparison of context length and a physical term. */
export function calculateContributionEvidence(shortBase: number, longBase: number, shortPhysics: number, longPhysics: number) {
  if (![shortBase, longBase, shortPhysics, longPhysics].every(value => Number.isFinite(value) && value >= 0)) {
    throw new RangeError('Expected four finite, nonnegative errors');
  }
  const shortGain = shortBase - shortPhysics;
  const longGain = longBase - longPhysics;
  return {
    shortGain,
    longGain,
    interaction: longGain - shortGain,
    combinedGain: shortBase - longPhysics,
    contextGainWithoutPhysics: shortBase - longBase,
  };
}
