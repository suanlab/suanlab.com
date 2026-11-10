/** Reader identity; registration is owned by the data layer. */
export const graphGaussTeachingId='graph-gauss-constraints-physical-phase-space' as const;
export type GraphGaussTeachingId=typeof graphGaussTeachingId;
export const isGraphGaussTeachingId=(id:string):id is GraphGaussTeachingId=>id===graphGaussTeachingId;
