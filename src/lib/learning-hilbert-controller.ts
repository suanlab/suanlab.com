import { isHilbertInput, hilbertInputKey, hilbertRecord, snapshotHilbertInput, isHilbertSuccessForInput,
  type HilbertInput, type HilbertSuccess } from './learning-hilbert-protocol';
export type HilbertWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type HilbertCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating'; requestId: number; input: HilbertInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: HilbertInput; inputKey: string; response: HilbertSuccess }
  | { status: 'error'; requestId: number; input: HilbertInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' };
export const idleHilbertState = (): HilbertCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
/** One worker per active request. New work terminates the prior computation,
 * instead of queuing behind a long synchronous calculation in that worker.
 */
export class HilbertCalculationController {
  private worker: HilbertWorkerPort | null = null;
  private active: { id: number; input: HilbertInput; inputKey: string } | null = null;
  private sequence = 0;
  private disposed = false;
  private state: HilbertCalculationState = idleHilbertState();
  constructor(private readonly factory: () => HilbertWorkerPort, private readonly notify: (state: HilbertCalculationState) => void) {}
  getSnapshot() { return this.state; }
  private publish(state: HilbertCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private stop() {
    const port = this.worker; this.worker = null; this.active = null;
    if (port) { port.onmessage = null; port.onerror = null; port.onmessageerror = null; port.terminate(); }
  }
  run(value: unknown): number | null {
    if (this.disposed) return null;
    this.stop(); const id = ++this.sequence;
    if (!isHilbertInput(value)) { this.publish({ status: 'error', requestId: id, input: null, inputKey: 'invalid-input', error: 'invalid-input' }); return id; }
    const input = snapshotHilbertInput(value), inputKey = hilbertInputKey(input), token = { id, input, inputKey };
    this.active = token; this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey });
    if (this.disposed || this.active !== token) return id;
    let port: HilbertWorkerPort;
    try { port = this.factory(); }
    catch {
      if (!this.disposed && this.active === token) { this.active = null; this.publish({ status: 'error', requestId: id, input, inputKey, error: 'unavailable' }); }
      return id;
    }
    if (this.disposed || this.active !== token) { port.terminate(); return id; }
    this.worker = port;
    const hilbert = () => !this.disposed && this.active === token && this.worker === port;
    const fail = (error: 'worker-failed' | 'invalid-response' | 'invalid-input') => {
      if (!hilbert()) return;
      this.stop(); this.publish({ status: 'error', requestId: id, input, inputKey, error });
    };
    port.onmessage = event => {
      if (!hilbert()) return;
      const message: unknown = event.data;
      if (hilbertRecord(message) && Number.isSafeInteger(message.requestId) && (message.requestId as number) > 0 && message.requestId !== id) return;
      if (!hilbertRecord(message) || !Object.prototype.hasOwnProperty.call(message, 'version') || !Object.prototype.hasOwnProperty.call(message, 'requestId') || message.version !== 1 || message.requestId !== id) { fail('invalid-response'); return; }
      if (message.phase === 'started' && message.kind === input.kind && Reflect.ownKeys(message).length === 4 && ['phase', 'kind'].every(k => Object.prototype.hasOwnProperty.call(message, k))) {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey }); return;
      }
      if (message.ok === false) {
        if (Reflect.ownKeys(message).length !== 4 || !['ok', 'error'].every(k => Object.prototype.hasOwnProperty.call(message, k)) || !['invalid-request', 'invalid-input', 'calculation-failed'].includes(message.error as string)) { fail('invalid-response'); return; }
        fail(message.error === 'invalid-input' ? 'invalid-input' : 'worker-failed'); return;
      }
      if (!isHilbertSuccessForInput(message, input, id)) { fail('invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: id, input, inputKey, response: message as unknown as HilbertSuccess });
    };
    port.onerror = event => { event.preventDefault(); fail('worker-failed'); };
    port.onmessageerror = () => fail('invalid-response');
    try { port.postMessage({ version: 1, requestId: id, input }); }
    catch { fail('worker-failed'); }
    return id;
  }
  cancel() { if (this.disposed) return; this.stop(); this.publish(idleHilbertState()); }
  dispose() { if (this.disposed) return; this.disposed = true; this.stop(); }
}
