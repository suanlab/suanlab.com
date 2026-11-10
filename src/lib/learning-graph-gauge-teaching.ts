/** Reader identity; registration is owned by the data layer. */
export const graphGaugeTeachingId='graph-u1-local-frames-wilson-loops' as const;
export type GraphGaugeTeachingId=typeof graphGaugeTeachingId;
export const isGraphGaugeTeachingId=(id:string):id is GraphGaugeTeachingId=>id===graphGaugeTeachingId;
