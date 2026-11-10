/** Reader identity; curriculum registration stays in the data layer. */
export const duffingSecularTeachingId='duffing-secular-frequency-long-time' as const;
export type DuffingSecularTeachingId=typeof duffingSecularTeachingId;
export const isDuffingSecularTeachingId=(id:string):id is DuffingSecularTeachingId=>id===duffingSecularTeachingId;
