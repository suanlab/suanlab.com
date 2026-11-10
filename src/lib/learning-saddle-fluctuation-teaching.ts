/** Reader identity; registration remains in the data layer. */
export const saddleFluctuationTeachingId='saddle-fluctuation-and-nongaussian-approximation' as const;
export type SaddleFluctuationTeachingId=typeof saddleFluctuationTeachingId;
export const isSaddleFluctuationTeachingId=(id:string):id is SaddleFluctuationTeachingId=>id===saddleFluctuationTeachingId;
