import { bindWorkerMessageError, clearWorkerMessageError } from './learning-worker-message-error';
import { isEnsembleInput, ensembleInputKey, ensembleRecord, snapshotEnsembleInput, ensembleResultMatchesInput,
  type EnsembleInput, type EnsembleSuccess } from './learning-ensemble-protocol';
export type EnsembleWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type EnsembleCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating'; requestId: number; input: EnsembleInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: EnsembleInput; inputKey: string; response: EnsembleSuccess }
  | { status: 'error'; requestId: number; input: EnsembleInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' };
export const idleEnsembleState = (): EnsembleCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
/** One worker per active request. New work terminates the prior computation,
 * instead of queuing behind a long synchronous calculation in that worker.
 */
export class EnsembleCalculationController {
  private worker: EnsembleWorkerPort | null = null;
  private active: { id: number; input: EnsembleInput; inputKey: string } | null = null;
  private sequence = 0;
  private disposed = false;
  private state: EnsembleCalculationState = idleEnsembleState();
  constructor(private readonly factory: () => EnsembleWorkerPort, private readonly notify: (state: EnsembleCalculationState) => void) {}
  getSnapshot() { return this.state; }
  private publish(state: EnsembleCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private stop() {
    const port = this.worker; this.worker = null; this.active = null;
    if (port) { port.onmessage = null; port.onerror = null; clearWorkerMessageError(port); port.terminate(); }
  }
  run(value: unknown): number | null {
    if (this.disposed) return null;
    this.stop(); const id = ++this.sequence;
    if (!isEnsembleInput(value)) { this.publish({ status: 'error', requestId: id, input: null, inputKey: 'invalid-input', error: 'invalid-input' }); return id; }
    const input = snapshotEnsembleInput(value), inputKey = ensembleInputKey(input), token = { id, input, inputKey };
    this.active = token; this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey });
    if (this.disposed || this.active !== token) return id;
    let port: EnsembleWorkerPort;
    try { port = this.factory(); }
    catch {
      if (!this.disposed && this.active === token) { this.active = null; this.publish({ status: 'error', requestId: id, input, inputKey, error: 'unavailable' }); }
      return id;
    }
    if (this.disposed || this.active !== token) { port.terminate(); return id; }
    this.worker = port;
    const current = () => !this.disposed && this.active === token && this.worker === port;
    const fail = (error: 'worker-failed' | 'invalid-response' | 'invalid-input') => {
      if (!current()) return;
      this.stop(); this.publish({ status: 'error', requestId: id, input, inputKey, error });
    };
    port.onmessage = event => {
      if (!current()) return;
      const message: unknown = event.data;
      if (ensembleRecord(message) && typeof message.requestId === 'number' && message.requestId !== id) return;
      if (!ensembleRecord(message) || message.version !== 1 || message.requestId !== id) { fail('invalid-response'); return; }
      if (message.phase === 'started' && message.kind === input.kind) {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey }); return;
      }
      if (message.ok === false) { fail(message.error === 'invalid-input' ? 'invalid-input' : 'worker-failed'); return; }
      if (message.ok !== true || message.kind !== input.kind || !ensembleResultMatchesInput(input, message.result)) { fail('invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: id, input, inputKey, response: message as unknown as EnsembleSuccess });
    };
    port.onerror = event => { event.preventDefault(); fail('worker-failed'); };
    bindWorkerMessageError(port, () => fail('invalid-response'));
    try { port.postMessage({ version: 1, requestId: id, input }); }
    catch { fail('worker-failed'); }
    return id;
  }
  cancel() { if (this.disposed) return; this.stop(); this.publish(idleEnsembleState()); }
  dispose() { if (this.disposed) return; this.disposed = true; this.stop(); }
}
