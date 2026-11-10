import { calculateIsingSweep } from './learning-ising-cylinder';

export type IsingSweepReply = { sweep: ReturnType<typeof calculateIsingSweep> } | { error: true };

self.onmessage = ({ data }: MessageEvent<number>) => {
  try {
    self.postMessage({ sweep: calculateIsingSweep(data) } satisfies IsingSweepReply);
  } catch {
    self.postMessage({ error: true } satisfies IsingSweepReply);
  }
};
