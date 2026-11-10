/** Reader identity; registration belongs to the data layer. */
export const nongaussianFieldTeachingId='interacting-nongaussian-connected-correlations' as const;
export type NongaussianFieldTeachingId=typeof nongaussianFieldTeachingId;
export const isNongaussianFieldTeachingId=(id:string):id is NongaussianFieldTeachingId=>id===nongaussianFieldTeachingId;
