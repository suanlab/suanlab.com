/** Reader identity; registration belongs to the data layer. */
export const regularPerturbationTeachingId = 'regular-perturbation-residual-certificates' as const;
export type RegularPerturbationTeachingId = typeof regularPerturbationTeachingId;
export const isRegularPerturbationTeachingId = (id: string): id is RegularPerturbationTeachingId => id === regularPerturbationTeachingId;
