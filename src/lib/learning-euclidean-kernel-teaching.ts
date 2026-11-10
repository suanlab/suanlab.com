/** Reader identity; registration remains in the data layer. */
export const euclideanKernelTeachingId = 'euclidean-quantum-kernels-and-time-slicing' as const;
export type EuclideanKernelTeachingId = typeof euclideanKernelTeachingId;
export const isEuclideanKernelTeachingId = (id: string): id is EuclideanKernelTeachingId => id === euclideanKernelTeachingId;
