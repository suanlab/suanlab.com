import { calculateLegendreResponse, type LegendreResponseModel } from './learning-legendre-response';

export type LegendreResponseReply = { rows: ReturnType<typeof calculateLegendreResponse> } | { error: true };

self.onmessage = ({ data }: MessageEvent<LegendreResponseModel>) => {
  try {
    self.postMessage({ rows: calculateLegendreResponse(data) } satisfies LegendreResponseReply);
  } catch {
    self.postMessage({ error: true } satisfies LegendreResponseReply);
  }
};
