/** Reader identity; registration belongs to the data layer. */
export const boundaryLayerTeachingId='singular-perturbation-boundary-layer-matching' as const;
export type BoundaryLayerTeachingId=typeof boundaryLayerTeachingId;
export const isBoundaryLayerTeachingId=(id:string):id is BoundaryLayerTeachingId=>id===boundaryLayerTeachingId;
