export const blockOrderTeachingId = 'microscopic-blocks-and-local-order-parameters' as const;
export type BlockOrderTeachingId = typeof blockOrderTeachingId;
export const isBlockOrderTeachingId = (id: string): id is BlockOrderTeachingId => id === blockOrderTeachingId;
