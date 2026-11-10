import { readRegularPerturbationInput, type RegularPerturbationInput } from './learning-regular-perturbation-input';
import { regularPerturbationInputKey } from './learning-regular-perturbation-protocol';
import { regularPerturbationData } from './learning-regular-perturbation-protocol';
import type { RegularPerturbationSuccess } from './learning-regular-perturbation-protocol';
import { regularPerturbationSuccessHeader } from './learning-regular-perturbation-protocol';
import { snapshotQubitWireInTasks, QubitWireLimit, type QubitCopyOptions } from './learning-qubit-wire';
export type RegularPerturbationWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type RegularPerturbationCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating' | 'validating'; requestId: number; input: RegularPerturbationInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: RegularPerturbationInput; inputKey: string; response: RegularPerturbationSuccess }
  | { status: 'error'; requestId: number; input: RegularPerturbationInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' | 'resource-limit' | 'timeout' };
export const idleRegularPerturbationState = (): RegularPerturbationCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
type Token = { id: number; input: RegularPerturbationInput; key: string; validating: boolean; accepting: boolean };
function envelope(value: unknown): Record<string, unknown> | null {
  try { if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) return null; const keys = Reflect.ownKeys(value); return keys.length <= 12 && keys.every(k => typeof k === 'string') ? regularPerturbationData(value, keys as string[]) : null; } catch { return null; }
}
/** Each run owns a calculation worker and a trusted independent validator.
 * No mathematical validator is imported into this main-thread controller.
 * Both large copies yield and abort when the active token changes.
 */
export class RegularPerturbationCalculationController {
  private calculation: RegularPerturbationWorkerPort | null = null;
  private validation: RegularPerturbationWorkerPort | null = null;
  private active: Token | null = null;
  private sequence = 0;
  private disposed = false;
  private state: RegularPerturbationCalculationState = idleRegularPerturbationState();
  private deadline: ReturnType<typeof setTimeout> | null = null;
  constructor(private readonly factories: { calculate: () => RegularPerturbationWorkerPort; validate: () => RegularPerturbationWorkerPort }, private readonly notify: (state: RegularPerturbationCalculationState) => void, private readonly copyOptions: Omit<QubitCopyOptions, 'active'> = {}, private readonly timeoutMilliseconds = 30000) { if (!Number.isInteger(timeoutMilliseconds) || timeoutMilliseconds < 10 || timeoutMilliseconds > 120000) throw new RangeError('Worker deadline10..120000ms required'); }
  getSnapshot() { return this.state; }
  private publish(state: RegularPerturbationCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private release(port: RegularPerturbationWorkerPort | null) { if (port) { port.onmessage = null; port.onerror = null; port.onmessageerror = null; try { port.terminate(); } catch { /* Keep releasing the other port. */ } } }
  private stop() { if (this.deadline !== null) clearTimeout(this.deadline); this.deadline = null; const a = this.calculation, b = this.validation; this.calculation = null; this.validation = null; this.active = null; this.release(a); this.release(b); }
  private current(t: Token) { return !this.disposed && this.active === t; }
  private fail(t: Token, error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' | 'resource-limit' | 'timeout') {
    if (!this.current(t)) return; this.stop(); this.publish({ status: 'error', requestId: t.id, input: t.input, inputKey: t.key, error });
  }
  run(value: unknown): number | null {
    if (this.disposed || !Number.isSafeInteger(this.sequence + 1)) return null;
    this.stop(); const id = ++this.sequence, input = readRegularPerturbationInput(value);
    if (!input) { this.publish({ status: 'error', requestId: id, input: null, inputKey: null, error: 'invalid-input' }); return id; }
    const t: Token = { id, input, key: regularPerturbationInputKey(input), validating: false, accepting: false }; this.active = t;
    this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey: t.key }); if (!this.current(t)) return id;
    this.deadline = setTimeout(() => this.fail(t, 'timeout'), this.timeoutMilliseconds);
    let port: RegularPerturbationWorkerPort;
    try { port = this.factories.calculate(); } catch { this.fail(t, 'unavailable'); return id; }
    if (!this.current(t)) { this.release(port); return id; } this.calculation = port;
    port.onmessage = event => {
      if (!this.current(t) || this.calculation !== port) return;
      const message = envelope(event.data); if (!message) { this.fail(t, 'invalid-response'); return; }
      if (typeof message.requestId === 'number' && Number.isSafeInteger(message.requestId) && message.requestId > 0 && message.requestId !== id) return;
      if (message.version !== 1 || message.requestId !== id) { this.fail(t, 'invalid-response'); return; }
      if (regularPerturbationData(message, ['version', 'requestId', 'phase', 'kind']) && message.phase === 'started' && message.kind === 'regular-perturbation') {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey: t.key }); return;
      }
      if (!regularPerturbationSuccessHeader(message, input, id)) {
        const failure = regularPerturbationData(message, ['version', 'requestId', 'ok', 'error']); this.fail(t, failure?.ok === false ? failure.error === 'resource-limit' ? 'resource-limit' : failure.error === 'invalid-input' ? 'invalid-input' : ['calculation-failed', 'invalid-request'].includes(failure.error as string) ? 'worker-failed' : 'invalid-response' : 'invalid-response'); return;
      }
      this.calculation = null; this.release(port); t.validating = true;
      this.publish({ status: 'pending', phase: 'validating', requestId: id, input, inputKey: t.key });
      if (this.current(t)) void this.validate(t, message);
    };
    port.onerror = event => { if (this.current(t) && this.calculation === port) { event.preventDefault(); this.fail(t, 'worker-failed'); } };
    port.onmessageerror = () => { if (this.current(t) && this.calculation === port) this.fail(t, 'invalid-response'); };
    try { port.postMessage({ version: 1, requestId: id, input }); } catch { this.fail(t, 'worker-failed'); }
    return id;
  }
  private async validate(t: Token, response: unknown) {
    try {
      const owned = await snapshotQubitWireInTasks(response, { ...this.copyOptions, active: () => this.current(t) }); if (!this.current(t)) return;
      let port: RegularPerturbationWorkerPort;
      try { port = this.factories.validate(); } catch { this.fail(t, 'unavailable'); return; }
      if (!this.current(t)) { this.release(port); return; } this.validation = port;
      port.onmessage = event => {
        if (!this.current(t) || this.validation !== port || t.accepting) return;
        const receipt = envelope(event.data); if (!receipt) { this.fail(t, 'invalid-response'); return; }
        if (typeof receipt.requestId === 'number' && Number.isSafeInteger(receipt.requestId) && receipt.requestId > 0 && receipt.requestId !== t.id) return;
        if (regularPerturbationData(receipt, ['version', 'requestId', 'phase', 'error']) && receipt.version === 1 && receipt.requestId === t.id && receipt.phase === 'rejected' && receipt.error === 'resource-limit') { this.fail(t, 'resource-limit'); return; }
        if (!regularPerturbationData(receipt, ['version', 'requestId', 'phase', 'kind', 'inputKey', 'response']) || receipt.version !== 1 || receipt.requestId !== t.id || receipt.phase !== 'validated' || receipt.kind !== 'regular-perturbation' || receipt.inputKey !== t.key || !regularPerturbationSuccessHeader(receipt.response, t.input, t.id)) { this.fail(t, 'invalid-response'); return; }
        t.accepting = true; this.validation = null; this.release(port); void this.accept(t, receipt.response);
      };
      port.onerror = event => { if (this.current(t) && this.validation === port) { event.preventDefault(); this.fail(t, 'worker-failed'); } };
      port.onmessageerror = () => { if (this.current(t) && this.validation === port) this.fail(t, 'invalid-response'); };
      try { port.postMessage({ version: 1, requestId: t.id, input: t.input, response: owned }); } catch { this.fail(t, 'worker-failed'); }
    } catch (error) { if (this.current(t)) this.fail(t, error instanceof QubitWireLimit ? 'resource-limit' : 'invalid-response'); }
  }
  private async accept(t: Token, response: unknown) {
    try {
      const owned = await snapshotQubitWireInTasks(response, { ...this.copyOptions, active: () => this.current(t) }); if (!this.current(t)) return;
      if (!regularPerturbationSuccessHeader(owned, t.input, t.id)) { this.fail(t, 'invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: t.id, input: t.input, inputKey: t.key, response: owned });
    } catch (error) { if (this.current(t)) this.fail(t, error instanceof QubitWireLimit ? 'resource-limit' : 'invalid-response'); }
  }
  cancel() { if (!this.disposed) { this.stop(); this.publish(idleRegularPerturbationState()); } }
  dispose() { if (!this.disposed) { this.disposed = true; this.stop(); } }
}
