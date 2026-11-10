/** Fixed concept preparations; no calculator or saved experiment state. */
import { ensemblePresetInput, type EnsembleKind } from './learning-ensembles';
import { snapshotEnsembleInput, type EnsembleInput } from './learning-ensemble-protocol';
const kB = 1.380649e-23;
export function ensembleConceptInputs(kind: EnsembleKind, step: number): readonly EnsembleInput[] {
  if (!Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Invalid ensemble concept step');
  if (kind === 'finite-binary-reservoir-canonical-comparison') {
    if (step < 3) return [ensemblePresetInput(kind, 0)];
    return [0, 2, 6, 18, 98].map(b => snapshotEnsembleInput({ kind, parameters: { subsystemSites: 2, reservoirSites: b, totalExcitations: (b + 2) / 2 } }));
  }
  if (kind === 'canonical-binary-energy-fluctuations') {
    if (step < 2) return [ensemblePresetInput(kind, 1)];
    return (step === 2 ? [.25, .5, 1, 2, 4, 8] : [1, 4, 16, 64]).map(v => snapshotEnsembleInput({ kind, parameters: { sites: step === 2 ? 4 : v, energyGap: kB, temperature: step === 2 ? v : 1 } }));
  }
  if (kind === 'grand-canonical-finite-occupancy-fluctuations') {
    if (step < 2) return [ensemblePresetInput(kind, 0)];
    if (step === 2) return [-2, -1, 0, 1, 2, 3, 4].map(mu => snapshotEnsembleInput({ kind, parameters: { sites: 2, energyGap: kB, chemicalPotential: mu * kB, temperature: 1 } }));
    return [3, 4, 5].map(i => ensemblePresetInput(kind, i));
  }
  throw new RangeError('Invalid ensemble kind');
}
