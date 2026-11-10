import { calculateScalarGibbs, type ScalarGibbsResult } from './learning-scalar-gibbs';

export type ScalarGibbsReply = { result: ScalarGibbsResult } | { error: true };

self.onmessage = ({ data }: MessageEvent<unknown>) => {
  try {
    self.postMessage({ result: calculateScalarGibbs(data) } satisfies ScalarGibbsReply);
  } catch {
    self.postMessage({ error: true } satisfies ScalarGibbsReply);
  }
};
