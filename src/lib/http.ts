/** Bounded downloads for local generation tools. Limits include response bodies. */
export interface DownloadOptions extends RequestInit {
  timeoutMs?: number;
  maxBytes?: number;
}

export class HttpError extends Error {
  constructor(readonly status: number, readonly retryAfter: string | null) {
    super(`HTTP ${status}`);
    this.name = 'HttpError';
  }
}

export async function downloadBytes(url: string, options: DownloadOptions = {}): Promise<Buffer> {
  if (!['http:', 'https:'].includes(new URL(url).protocol)) throw new Error('Only HTTP(S) downloads are supported.');
  const { timeoutMs = 60_000, maxBytes = 50 * 1024 * 1024, signal, ...init } = options;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || !Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error('Invalid download limits.');
  const controller = new AbortController();
  const abort = () => controller.abort(signal?.reason);
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) abort();
  const timer = setTimeout(() => controller.abort(new Error('Download timed out.')), timeoutMs);
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    reader = response.body?.getReader();
    if (!response.ok) throw new HttpError(response.status, response.headers.get('retry-after'));
    if (Number(response.headers.get('content-length')) > maxBytes) throw new Error('Download exceeds the size limit.');
    if (!reader) throw new Error('Download returned an empty body.');
    const chunks: Uint8Array[] = [];
    let size = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) throw new Error('Download exceeds the size limit.');
      chunks.push(value);
    }
    return Buffer.concat(chunks, size);
  } catch (error) {
    if (controller.signal.aborted) throw controller.signal.reason;
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
    void reader?.cancel().catch(() => {});
    controller.abort();
  }
}

export async function downloadJson<T>(url: string, options: DownloadOptions = {}): Promise<T> {
  const bytes = await downloadBytes(url, { maxBytes: 32 * 1024 * 1024, ...options });
  return JSON.parse(bytes.toString('utf8')) as T;
}
