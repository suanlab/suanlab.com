/** Reader identity only; registration is owned by the data layer. */
export const scalarFieldTeachingId='scalar-field-action-boundary-variation' as const;
export type ScalarFieldTeachingId=typeof scalarFieldTeachingId;
export const isScalarFieldTeachingId=(id:string):id is ScalarFieldTeachingId=>id===scalarFieldTeachingId;
