import { bindWorkerMessageError, clearWorkerMessageError } from './learning-worker-message-error';
import { isPhaseInput, phaseInputKey, phaseRecord, snapshotPhaseInput, isPhaseSuccessForInput,
  type PhaseInput, type PhaseSuccess } from './learning-phase-transitions-protocol';
export type PhaseWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type PhaseCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating'; requestId: number; input: PhaseInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: PhaseInput; inputKey: string; response: PhaseSuccess }
  | { status: 'error'; requestId: number; input: PhaseInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' };
export const idlePhaseState = (): PhaseCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
/** One worker per active request. New work terminates the prior computation,
 * instead of queuing behind a long synchronous calculation in that worker.
 */
export class PhaseCalculationController {
  private worker: PhaseWorkerPort | null = null;
  private active: { id: number; input: PhaseInput; inputKey: string } | null = null;
  private sequence = 0;
  private disposed = false;
  private state: PhaseCalculationState = idlePhaseState();
  constructor(private readonly factory: () => PhaseWorkerPort, private readonly notify: (state: PhaseCalculationState) => void) {}
  getSnapshot() { return this.state; }
  private publish(state: PhaseCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private stop() {
    const port = this.worker; this.worker = null; this.active = null;
    if (port) { port.onmessage = null; port.onerror = null; clearWorkerMessageError(port); port.terminate(); }
  }
  run(value: unknown): number | null {
    if (this.disposed) return null;
    this.stop(); const id = ++this.sequence;
    if (!isPhaseInput(value)) { this.publish({ status: 'error', requestId: id, input: null, inputKey: 'invalid-input', error: 'invalid-input' }); return id; }
    const input = snapshotPhaseInput(value), inputKey = phaseInputKey(input), token = { id, input, inputKey };
    this.active = token; this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey });
    if (this.disposed || this.active !== token) return id;
    let port: PhaseWorkerPort;
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
      if (phaseRecord(message) && Number.isSafeInteger(message.requestId) && (message.requestId as number) > 0 && message.requestId !== id) return;
      if (!phaseRecord(message) || message.version !== 1 || message.requestId !== id) { fail('invalid-response'); return; }
      if (message.phase === 'started' && message.kind === input.kind) {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey }); return;
      }
      if (message.ok === false) { fail(message.error === 'invalid-input' ? 'invalid-input' : 'worker-failed'); return; }
      if (!isPhaseSuccessForInput(message, input, id)) { fail('invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: id, input, inputKey, response: message as unknown as PhaseSuccess });
    };
    port.onerror = event => { event.preventDefault(); fail('worker-failed'); };
    bindWorkerMessageError(port, () => fail('invalid-response'));
    try { port.postMessage({ version: 1, requestId: id, input }); }
    catch { fail('worker-failed'); }
    return id;
  }
  cancel() { if (this.disposed) return; this.stop(); this.publish(idlePhaseState()); }
  dispose() { if (this.disposed) return; this.disposed = true; this.stop(); }
}
