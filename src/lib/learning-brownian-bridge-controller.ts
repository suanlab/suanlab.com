import { bindWorkerMessageError, clearWorkerMessageError } from './learning-worker-message-error';
import { readBrownianBridgeInput, type BrownianBridgeInput } from './learning-brownian-bridge-input';
import { brownianBridgeInputKey } from './learning-brownian-bridge-protocol';
import { brownianBridgeData } from './learning-brownian-bridge-protocol';
import type { BrownianBridgeSuccess } from './learning-brownian-bridge-protocol';
import { brownianBridgeSuccessHeader } from './learning-brownian-bridge-protocol';
import { snapshotQubitWireInTasks, QubitWireLimit, type QubitCopyOptions } from './learning-qubit-wire';
export type BrownianBridgeWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type BrownianBridgeCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating' | 'validating'; requestId: number; input: BrownianBridgeInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: BrownianBridgeInput; inputKey: string; response: BrownianBridgeSuccess }
  | { status: 'error'; requestId: number; input: BrownianBridgeInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' | 'resource-limit' | 'timeout' };
export const idleBrownianBridgeState = (): BrownianBridgeCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
type Token = { id: number; input: BrownianBridgeInput; key: string; validating: boolean; accepting: boolean };
function envelope(value: unknown): Record<string, unknown> | null {
  try { if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) return null; const keys = Reflect.ownKeys(value); return keys.length <= 12 && keys.every(k => typeof k === 'string') ? brownianBridgeData(value, keys as string[]) : null; } catch { return null; }
}
/** Each run owns a calculation worker and a trusted independent validator.
 * No mathematical validator is imported into this main-thread controller.
 * Both large copies yield and abort when the active token changes.
 */
export class BrownianBridgeCalculationController {
  private calculation: BrownianBridgeWorkerPort | null = null;
  private validation: BrownianBridgeWorkerPort | null = null;
  private active: Token | null = null;
  private sequence = 0;
  private disposed = false;
  private state: BrownianBridgeCalculationState = idleBrownianBridgeState();
  private deadline: ReturnType<typeof setTimeout> | null = null;
  constructor(private readonly factories: { calculate: () => BrownianBridgeWorkerPort; validate: () => BrownianBridgeWorkerPort }, private readonly notify: (state: BrownianBridgeCalculationState) => void, private readonly copyOptions: Omit<QubitCopyOptions, 'active'> = {}, private readonly timeoutMilliseconds = 30000) { if (!Number.isInteger(timeoutMilliseconds) || timeoutMilliseconds < 10 || timeoutMilliseconds > 120000) throw new RangeError('Worker deadline10..120000ms required'); }
  getSnapshot() { return this.state; }
  private publish(state: BrownianBridgeCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private release(port: BrownianBridgeWorkerPort | null) { if (port) { port.onmessage = null; port.onerror = null; clearWorkerMessageError(port); try { port.terminate(); } catch { /* Keep releasing the other port. */ } } }
  private stop() { if (this.deadline !== null) clearTimeout(this.deadline); this.deadline = null; const a = this.calculation, b = this.validation; this.calculation = null; this.validation = null; this.active = null; this.release(a); this.release(b); }
  private current(t: Token) { return !this.disposed && this.active === t; }
  private fail(t: Token, error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' | 'resource-limit' | 'timeout') {
    if (!this.current(t)) return; this.stop(); this.publish({ status: 'error', requestId: t.id, input: t.input, inputKey: t.key, error });
  }
  run(value: unknown): number | null {
    if (this.disposed || !Number.isSafeInteger(this.sequence + 1)) return null;
    this.stop(); const id = ++this.sequence, input = readBrownianBridgeInput(value);
    if (!input) { this.publish({ status: 'error', requestId: id, input: null, inputKey: null, error: 'invalid-input' }); return id; }
    const t: Token = { id, input, key: brownianBridgeInputKey(input), validating: false, accepting: false }; this.active = t;
    this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey: t.key }); if (!this.current(t)) return id;
    this.deadline = setTimeout(() => this.fail(t, 'timeout'), this.timeoutMilliseconds);
    let port: BrownianBridgeWorkerPort;
    try { port = this.factories.calculate(); } catch { this.fail(t, 'unavailable'); return id; }
    if (!this.current(t)) { this.release(port); return id; } this.calculation = port;
    port.onmessage = event => {
      if (!this.current(t) || this.calculation !== port) return;
      const message = envelope(event.data); if (!message) { this.fail(t, 'invalid-response'); return; }
      if (typeof message.requestId === 'number' && Number.isSafeInteger(message.requestId) && message.requestId > 0 && message.requestId !== id) return;
      if (message.version !== 1 || message.requestId !== id) { this.fail(t, 'invalid-response'); return; }
      if (brownianBridgeData(message, ['version', 'requestId', 'phase', 'kind']) && message.phase === 'started' && message.kind === 'brownian-bridge') {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey: t.key }); return;
      }
      if (!brownianBridgeSuccessHeader(message, input, id)) {
        const failure = brownianBridgeData(message, ['version', 'requestId', 'ok', 'error']); this.fail(t, failure?.ok === false ? failure.error === 'resource-limit' ? 'resource-limit' : failure.error === 'invalid-input' ? 'invalid-input' : ['calculation-failed', 'invalid-request'].includes(failure.error as string) ? 'worker-failed' : 'invalid-response' : 'invalid-response'); return;
      }
      this.calculation = null; this.release(port); t.validating = true;
      this.publish({ status: 'pending', phase: 'validating', requestId: id, input, inputKey: t.key });
      if (this.current(t)) void this.validate(t, message);
    };
    port.onerror = event => { if (this.current(t) && this.calculation === port) { event.preventDefault(); this.fail(t, 'worker-failed'); } };
    bindWorkerMessageError(port, () => { if (this.current(t) && this.calculation === port) this.fail(t, 'invalid-response'); });
    try { port.postMessage({ version: 1, requestId: id, input }); } catch { this.fail(t, 'worker-failed'); }
    return id;
  }
  private async validate(t: Token, response: unknown) {
    try {
      const owned = await snapshotQubitWireInTasks(response, { ...this.copyOptions, active: () => this.current(t) }); if (!this.current(t)) return;
      let port: BrownianBridgeWorkerPort;
      try { port = this.factories.validate(); } catch { this.fail(t, 'unavailable'); return; }
      if (!this.current(t)) { this.release(port); return; } this.validation = port;
      port.onmessage = event => {
        if (!this.current(t) || this.validation !== port || t.accepting) return;
        const receipt = envelope(event.data); if (!receipt) { this.fail(t, 'invalid-response'); return; }
        if (typeof receipt.requestId === 'number' && Number.isSafeInteger(receipt.requestId) && receipt.requestId > 0 && receipt.requestId !== t.id) return;
        if (brownianBridgeData(receipt, ['version', 'requestId', 'phase', 'error']) && receipt.version === 1 && receipt.requestId === t.id && receipt.phase === 'rejected' && receipt.error === 'resource-limit') { this.fail(t, 'resource-limit'); return; }
        if (!brownianBridgeData(receipt, ['version', 'requestId', 'phase', 'kind', 'inputKey', 'response']) || receipt.version !== 1 || receipt.requestId !== t.id || receipt.phase !== 'validated' || receipt.kind !== 'brownian-bridge' || receipt.inputKey !== t.key || !brownianBridgeSuccessHeader(receipt.response, t.input, t.id)) { this.fail(t, 'invalid-response'); return; }
        t.accepting = true; this.validation = null; this.release(port); void this.accept(t, receipt.response);
      };
      port.onerror = event => { if (this.current(t) && this.validation === port) { event.preventDefault(); this.fail(t, 'worker-failed'); } };
      bindWorkerMessageError(port, () => { if (this.current(t) && this.validation === port) this.fail(t, 'invalid-response'); });
      try { port.postMessage({ version: 1, requestId: t.id, input: t.input, response: owned }); } catch { this.fail(t, 'worker-failed'); }
    } catch (error) { if (this.current(t)) this.fail(t, error instanceof QubitWireLimit ? 'resource-limit' : 'invalid-response'); }
  }
  private async accept(t: Token, response: unknown) {
    try {
      const owned = await snapshotQubitWireInTasks(response, { ...this.copyOptions, active: () => this.current(t) }); if (!this.current(t)) return;
      if (!brownianBridgeSuccessHeader(owned, t.input, t.id)) { this.fail(t, 'invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: t.id, input: t.input, inputKey: t.key, response: owned });
    } catch (error) { if (this.current(t)) this.fail(t, error instanceof QubitWireLimit ? 'resource-limit' : 'invalid-response'); }
  }
  cancel() { if (!this.disposed) { this.stop(); this.publish(idleBrownianBridgeState()); } }
  dispose() { if (!this.disposed) { this.disposed = true; this.stop(); } }
}
