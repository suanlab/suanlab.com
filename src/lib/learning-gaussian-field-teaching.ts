/** Reader identity; registration belongs to the data layer. */
export const gaussianFieldTeachingId='finite-gaussian-field-sources-connected-correlations' as const;
export type GaussianFieldTeachingId=typeof gaussianFieldTeachingId;
export const isGaussianFieldTeachingId=(id:string):id is GaussianFieldTeachingId=>id===gaussianFieldTeachingId;
