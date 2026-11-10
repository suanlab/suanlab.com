import { bindWorkerMessageError, clearWorkerMessageError } from './learning-worker-message-error';
import { isBoltzmannGibbsInput, boltzmannGibbsInputKey, boltzmannRecord, snapshotBoltzmannGibbsInput, isBoltzmannGibbsSuccessForInput,
  type BoltzmannGibbsInput, type BoltzmannGibbsSuccess } from './learning-boltzmann-gibbs-protocol';
export type BoltzmannGibbsWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type BoltzmannGibbsCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating'; requestId: number; input: BoltzmannGibbsInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: BoltzmannGibbsInput; inputKey: string; response: BoltzmannGibbsSuccess }
  | { status: 'error'; requestId: number; input: BoltzmannGibbsInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' };
export const idleBoltzmannGibbsState = (): BoltzmannGibbsCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
/** One worker per active request. New work terminates the prior computation,
 * instead of queuing behind a long synchronous calculation in that worker.
 */
export class BoltzmannGibbsCalculationController {
  private worker: BoltzmannGibbsWorkerPort | null = null;
  private active: { id: number; input: BoltzmannGibbsInput; inputKey: string } | null = null;
  private sequence = 0;
  private disposed = false;
  private state: BoltzmannGibbsCalculationState = idleBoltzmannGibbsState();
  constructor(private readonly factory: () => BoltzmannGibbsWorkerPort, private readonly notify: (state: BoltzmannGibbsCalculationState) => void) {}
  getSnapshot() { return this.state; }
  private publish(state: BoltzmannGibbsCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private stop() {
    const port = this.worker; this.worker = null; this.active = null;
    if (port) { port.onmessage = null; port.onerror = null; clearWorkerMessageError(port); port.terminate(); }
  }
  run(value: unknown): number | null {
    if (this.disposed) return null;
    this.stop(); const id = ++this.sequence;
    if (!isBoltzmannGibbsInput(value)) { this.publish({ status: 'error', requestId: id, input: null, inputKey: 'invalid-input', error: 'invalid-input' }); return id; }
    const input = snapshotBoltzmannGibbsInput(value), inputKey = boltzmannGibbsInputKey(input), token = { id, input, inputKey };
    this.active = token; this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey });
    if (this.disposed || this.active !== token) return id;
    let port: BoltzmannGibbsWorkerPort;
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
      if (boltzmannRecord(message) && Number.isSafeInteger(message.requestId) && (message.requestId as number) > 0 && message.requestId !== id) return;
      if (!boltzmannRecord(message) || message.version !== 1 || message.requestId !== id) { fail('invalid-response'); return; }
      if (message.phase === 'started' && message.kind === input.kind) {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey }); return;
      }
      if (message.ok === false) { fail(message.error === 'invalid-input' ? 'invalid-input' : 'worker-failed'); return; }
      if (!isBoltzmannGibbsSuccessForInput(message, input, id)) { fail('invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: id, input, inputKey, response: message as unknown as BoltzmannGibbsSuccess });
    };
    port.onerror = event => { event.preventDefault(); fail('worker-failed'); };
    bindWorkerMessageError(port, () => fail('invalid-response'));
    try { port.postMessage({ version: 1, requestId: id, input }); }
    catch { fail('worker-failed'); }
    return id;
  }
  cancel() { if (this.disposed) return; this.stop(); this.publish(idleBoltzmannGibbsState()); }
  dispose() { if (this.disposed) return; this.disposed = true; this.stop(); }
}
