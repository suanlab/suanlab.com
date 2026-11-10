/** Reader-only identity map; registration is owned by the data layer. */
import {quantumInformationDraftIds,type QuantumInformationDraftId} from './learning-quantum-information-topics';
import type {QuantumInformationKind} from './learning-quantum-information-input';
export const informationTeachingIds=quantumInformationDraftIds;
export type InformationTeachingId=QuantumInformationDraftId;
export const isInformationTeachingId=(id:string):id is InformationTeachingId=>(informationTeachingIds as readonly string[]).includes(id);
export const informationTeachingKinds={
 'binary-qubit-state-discrimination':'discrimination',
 'quantum-entropy-compression-conditional-information':'entropy',
 'classical-quantum-ensembles-holevo-information':'ensemble',
 'quantum-relative-entropy-data-processing':'relative-entropy',
 'no-cloning-bb84-information-disturbance':'bb84',
 'teleportation-superdense-coding-resources':'teleportation',
 'quantum-error-correction-syndrome-knill-laflamme':'error-correction',
 'pure-state-entanglement-majorization-resources':'majorization',
 'quantum-channel-capacities-coding-limits':'capacity',
}as const satisfies Record<InformationTeachingId,QuantumInformationKind>;
