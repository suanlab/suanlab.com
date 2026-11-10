import { isMicroMacroInput, microMacroInputKey, microMacroRecord, snapshotMicroMacroInput,
  type MicroMacroInput, type MicroMacroSuccess } from './learning-micro-macro-protocol';
export type MicroMacroWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type MicroMacroCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating'; requestId: number; input: MicroMacroInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: MicroMacroInput; inputKey: string; response: MicroMacroSuccess }
  | { status: 'error'; requestId: number; input: MicroMacroInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' };
export const idleMicroMacroState = (): MicroMacroCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
const scopes: Record<MicroMacroInput['kind'], string> = {
  'binary-shell-multiplicity-block-counts': 'finite-distinguishable-binary-sites',
  'finite-macro-partition-conditional-entropy': 'finite-weighted-deterministic-macro-observation',
  'finite-noisy-macro-observation-channel': 'finite-classical-noisy-observation',
};
/** One worker per active request. New work terminates the prior computation,
 * instead of queuing behind a long synchronous calculation in that worker.
 */
export class MicroMacroCalculationController {
  private worker: MicroMacroWorkerPort | null = null;
  private active: { id: number; input: MicroMacroInput; inputKey: string } | null = null;
  private sequence = 0;
  private disposed = false;
  private state: MicroMacroCalculationState = idleMicroMacroState();
  constructor(private readonly factory: () => MicroMacroWorkerPort, private readonly notify: (state: MicroMacroCalculationState) => void) {}
  getSnapshot() { return this.state; }
  private publish(state: MicroMacroCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private stop() {
    const port = this.worker; this.worker = null; this.active = null;
    if (port) { port.onmessage = null; port.onerror = null; port.onmessageerror = null; port.terminate(); }
  }
  run(value: unknown): number | null {
    if (this.disposed) return null;
    this.stop(); const id = ++this.sequence;
    if (!isMicroMacroInput(value)) { this.publish({ status: 'error', requestId: id, input: null, inputKey: 'invalid-input', error: 'invalid-input' }); return id; }
    const input = snapshotMicroMacroInput(value), inputKey = microMacroInputKey(input), token = { id, input, inputKey };
    this.active = token; this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey });
    if (this.disposed || this.active !== token) return id;
    let port: MicroMacroWorkerPort;
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
      if (microMacroRecord(message) && typeof message.requestId === 'number' && message.requestId !== id) return;
      if (!microMacroRecord(message) || message.version !== 1 || message.requestId !== id) { fail('invalid-response'); return; }
      if (message.phase === 'started' && message.kind === input.kind) {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey }); return;
      }
      if (message.ok === false) { fail(message.error === 'invalid-input' ? 'invalid-input' : 'worker-failed'); return; }
      if (message.ok !== true || message.kind !== input.kind || !microMacroRecord(message.result) || message.result.scope !== scopes[input.kind]) { fail('invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: id, input, inputKey, response: message as unknown as MicroMacroSuccess });
    };
    port.onerror = event => { event.preventDefault(); fail('worker-failed'); };
    port.onmessageerror = () => fail('invalid-response');
    try { port.postMessage({ version: 1, requestId: id, input }); }
    catch { fail('worker-failed'); }
    return id;
  }
  cancel() { if (this.disposed) return; this.stop(); this.publish(idleMicroMacroState()); }
  dispose() { if (this.disposed) return; this.disposed = true; this.stop(); }
}
