/** Reader identity; registration belongs to the data layer. */
export const quarticAsymptoticTeachingId = 'quartic-integral-asymptotic-truncation' as const;
export type QuarticAsymptoticTeachingId = typeof quarticAsymptoticTeachingId;
export const isQuarticAsymptoticTeachingId = (id: string): id is QuarticAsymptoticTeachingId => id === quarticAsymptoticTeachingId;
