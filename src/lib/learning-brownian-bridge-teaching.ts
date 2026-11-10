/** Reader identity; registration remains in the data layer. */
export const brownianBridgeTeachingId = 'brownian-bridge-time-slicing-measure' as const;
export type BrownianBridgeTeachingId = typeof brownianBridgeTeachingId;
export const isBrownianBridgeTeachingId = (id: string): id is BrownianBridgeTeachingId => id === brownianBridgeTeachingId;
