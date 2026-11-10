/** Reader identity; registration is owned by the data layer. */
export const fieldModesTeachingId='field-fourier-modes-lattice-continuum' as const;
export type FieldModesTeachingId=typeof fieldModesTeachingId;
export const isFieldModesTeachingId=(id:string):id is FieldModesTeachingId=>id===fieldModesTeachingId;
