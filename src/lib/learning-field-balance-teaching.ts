/** Reader identity; registration is owned by the data layer. */
export const fieldBalanceTeachingId='field-noether-current-boundary-balance' as const;
export type FieldBalanceTeachingId=typeof fieldBalanceTeachingId;
export const isFieldBalanceTeachingId=(id:string):id is FieldBalanceTeachingId=>id===fieldBalanceTeachingId;
