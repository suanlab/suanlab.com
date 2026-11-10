/** Reader registry predicate; no data-layer or numerical imports. */
export const manyBodyTeachingIds=['finite-spin-chain-local-hamiltonians','fermion-fock-parity-operator-algebra','boson-fock-cutoff-boundary-algebra','matrix-product-state-contraction-gauge','schmidt-cuts-bond-rank-truncation'] as const;
export type ManyBodyTeachingId=typeof manyBodyTeachingIds[number];
export function isManyBodyTeachingId(value:string):value is ManyBodyTeachingId{return(manyBodyTeachingIds as readonly string[]).includes(value);}
