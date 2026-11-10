/** Reader identity; registration stays in the data layer. */
export const spatialSpectrumTeachingId = 'spatial-spectra-decay-and-correlation-length' as const;
export type SpatialSpectrumTeachingId = typeof spatialSpectrumTeachingId;
export const isSpatialSpectrumTeachingId = (id: string): id is SpatialSpectrumTeachingId => id === spatialSpectrumTeachingId;
