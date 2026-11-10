/** Preparation IDs only. These do not register routes or mark topics complete. */
export const quantumInformationDraftIds = [
  'binary-qubit-state-discrimination',
  'quantum-entropy-compression-conditional-information',
  'classical-quantum-ensembles-holevo-information',
  'quantum-relative-entropy-data-processing',
  'no-cloning-bb84-information-disturbance',
  'teleportation-superdense-coding-resources',
  'quantum-error-correction-syndrome-knill-laflamme',
  'pure-state-entanglement-majorization-resources',
  'quantum-channel-capacities-coding-limits',
] as const;
export type QuantumInformationDraftId = typeof quantumInformationDraftIds[number];
