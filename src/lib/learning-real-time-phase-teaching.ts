/** Reader identity; catalogue registration belongs to the data layer. */
export const realTimePhaseTeachingId='real-time-phase-and-classical-limit' as const;
export type RealTimePhaseTeachingId=typeof realTimePhaseTeachingId;
export const isRealTimePhaseTeachingId=(id:string):id is RealTimePhaseTeachingId=>id===realTimePhaseTeachingId;
