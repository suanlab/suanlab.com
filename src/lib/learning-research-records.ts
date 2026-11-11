/** Describe a declared selection among completed validation runs. */
export function calculateResearchRecordSummary(losses: readonly number[], selectedCount: number) {
  if (losses.length === 0 || !losses.every(value => Number.isFinite(value) && value >= 0) ||
      !Number.isInteger(selectedCount) || selectedCount < 1 || selectedCount > losses.length) {
    throw new RangeError('Expected completed nonnegative losses and a valid selection count');
  }
  const ordered = [...losses].sort((a, b) => a - b);
  const selected = ordered.slice(0, selectedCount);
  return {
    ordered,
    selected,
    allMean: losses.reduce((sum, value) => sum + value, 0) / losses.length,
    selectedMean: selected.reduce((sum, value) => sum + value, 0) / selected.length,
    completedCount: losses.length,
  };
}
