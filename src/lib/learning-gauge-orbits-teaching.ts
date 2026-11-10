/** Reader identity; registration belongs to the data layer. */
export const gaugeOrbitTeachingId='gauge-orbits-tree-fixing-boundary-anchors' as const;
export type GaugeOrbitTeachingId=typeof gaugeOrbitTeachingId;
export const isGaugeOrbitTeachingId=(id:string):id is GaugeOrbitTeachingId=>id===gaugeOrbitTeachingId;
