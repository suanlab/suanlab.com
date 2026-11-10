import { bindWorkerMessageError, clearWorkerMessageError } from './learning-worker-message-error';
import { manyBodyData, readManyBodyInput, manyBodyInputKey, type ManyBodyInput } from './learning-many-body-input';
import type { ManyBodySuccess } from './learning-many-body-protocol';
import { manyBodySuccessHeader } from './learning-many-body-validation-protocol';
import { snapshotManyBodyWireInTasks, type ManyBodyCopyOptions } from './learning-many-body-owned-wire-async';
export type ManyBodyWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type ManyBodyCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating' | 'validating'; requestId: number; input: ManyBodyInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: ManyBodyInput; inputKey: string; response: ManyBodySuccess }
  | { status: 'error'; requestId: number; input: ManyBodyInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' };
export const idleManyBodyState = (): ManyBodyCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
type Token = { id: number; input: ManyBodyInput; key: string; validating: boolean; accepting: boolean };
function envelope(value: unknown): Record<string, unknown> | null {
  try { if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) return null; const keys = Reflect.ownKeys(value); return keys.length <= 12 && keys.every(k => typeof k === 'string') ? manyBodyData(value, keys as string[]) : null; } catch { return null; }
}
/** Each run owns a calculation worker and a trusted independent validator.
 * No mathematical validator is imported into this main-thread controller.
 * Both large copies yield and abort when the active token changes.
 */
export class ManyBodyCalculationController {
  private calculation: ManyBodyWorkerPort | null = null;
  private validation: ManyBodyWorkerPort | null = null;
  private active: Token | null = null;
  private sequence = 0;
  private disposed = false;
  private state: ManyBodyCalculationState = idleManyBodyState();
  constructor(private readonly factories: { calculate: () => ManyBodyWorkerPort; validate: () => ManyBodyWorkerPort }, private readonly notify: (state: ManyBodyCalculationState) => void, private readonly copyOptions: Omit<ManyBodyCopyOptions, 'active'> = {}) {}
  getSnapshot() { return this.state; }
  private publish(state: ManyBodyCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private release(port: ManyBodyWorkerPort | null) { if (port) { port.onmessage = null; port.onerror = null; clearWorkerMessageError(port); try { port.terminate(); } catch { /* Keep releasing the other port. */ } } }
  private stop() { const a = this.calculation, b = this.validation; this.calculation = null; this.validation = null; this.active = null; this.release(a); this.release(b); }
  private current(t: Token) { return !this.disposed && this.active === t; }
  private fail(t: Token, error: 'unavailable' | 'worker-failed' | 'invalid-response') {
    if (!this.current(t)) return; this.stop(); this.publish({ status: 'error', requestId: t.id, input: t.input, inputKey: t.key, error });
  }
  run(value: unknown): number | null {
    if (this.disposed || !Number.isSafeInteger(this.sequence + 1)) return null;
    this.stop(); const id = ++this.sequence, input = readManyBodyInput(value);
    if (!input) { this.publish({ status: 'error', requestId: id, input: null, inputKey: null, error: 'invalid-input' }); return id; }
    const t: Token = { id, input, key: manyBodyInputKey(input), validating: false, accepting: false }; this.active = t;
    this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey: t.key }); if (!this.current(t)) return id;
    let port: ManyBodyWorkerPort;
    try { port = this.factories.calculate(); } catch { this.fail(t, 'unavailable'); return id; }
    if (!this.current(t)) { this.release(port); return id; } this.calculation = port;
    port.onmessage = event => {
      if (!this.current(t) || this.calculation !== port) return;
      const message = envelope(event.data); if (!message) { this.fail(t, 'invalid-response'); return; }
      if (typeof message.requestId === 'number' && Number.isSafeInteger(message.requestId) && message.requestId > 0 && message.requestId !== id) return;
      if (message.version !== 1 || message.requestId !== id) { this.fail(t, 'invalid-response'); return; }
      if (manyBodyData(message, ['version', 'requestId', 'phase', 'kind']) && message.phase === 'started' && message.kind === input.kind) {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey: t.key }); return;
      }
      if (!manyBodySuccessHeader(message, input, id)) {
        const failure = manyBodyData(message, ['version', 'requestId', 'ok', 'error']); this.fail(t, failure?.ok === false && ['calculation-failed', 'invalid-input', 'invalid-request'].includes(failure.error as string) ? 'worker-failed' : 'invalid-response'); return;
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
      const owned = await snapshotManyBodyWireInTasks(response, { ...this.copyOptions, active: () => this.current(t) }); if (!this.current(t)) return;
      let port: ManyBodyWorkerPort;
      try { port = this.factories.validate(); } catch { this.fail(t, 'unavailable'); return; }
      if (!this.current(t)) { this.release(port); return; } this.validation = port;
      port.onmessage = event => {
        if (!this.current(t) || this.validation !== port || t.accepting) return;
        const receipt = envelope(event.data); if (!receipt) { this.fail(t, 'invalid-response'); return; }
        if (typeof receipt.requestId === 'number' && Number.isSafeInteger(receipt.requestId) && receipt.requestId > 0 && receipt.requestId !== t.id) return;
        if (!manyBodyData(receipt, ['version', 'requestId', 'phase', 'kind', 'inputKey', 'response']) || receipt.version !== 1 || receipt.requestId !== t.id || receipt.phase !== 'validated' || receipt.kind !== t.input.kind || receipt.inputKey !== t.key || !manyBodySuccessHeader(receipt.response, t.input, t.id)) { this.fail(t, 'invalid-response'); return; }
        t.accepting = true; this.validation = null; this.release(port); void this.accept(t, receipt.response);
      };
      port.onerror = event => { if (this.current(t) && this.validation === port) { event.preventDefault(); this.fail(t, 'worker-failed'); } };
      bindWorkerMessageError(port, () => { if (this.current(t) && this.validation === port) this.fail(t, 'invalid-response'); });
      try { port.postMessage({ version: 1, requestId: t.id, input: t.input, response: owned }); } catch { this.fail(t, 'worker-failed'); }
    } catch { if (this.current(t)) this.fail(t, 'invalid-response'); }
  }
  private async accept(t: Token, response: unknown) {
    try {
      const owned = await snapshotManyBodyWireInTasks(response, { ...this.copyOptions, active: () => this.current(t) }); if (!this.current(t)) return;
      if (!manyBodySuccessHeader(owned, t.input, t.id)) { this.fail(t, 'invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: t.id, input: t.input, inputKey: t.key, response: owned });
    } catch { if (this.current(t)) this.fail(t, 'invalid-response'); }
  }
  cancel() { if (!this.disposed) { this.stop(); this.publish(idleManyBodyState()); } }
  dispose() { if (!this.disposed) { this.disposed = true; this.stop(); } }
}
