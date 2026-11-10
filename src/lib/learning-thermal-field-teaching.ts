export const thermalFieldTeachingId='finite-field-measure-and-thermal-free-energy' as const;
export type ThermalFieldTeachingId=typeof thermalFieldTeachingId;
export const isThermalFieldTeachingId=(id:string):id is ThermalFieldTeachingId=>id===thermalFieldTeachingId;
