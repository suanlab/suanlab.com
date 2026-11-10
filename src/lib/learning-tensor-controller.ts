import { isTensorInput, tensorInputKey, tensorRecord, snapshotTensorInput, isTensorSuccessForInput,
  type TensorInput, type TensorSuccess } from './learning-tensor-protocol';
export type TensorWorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
export type TensorCalculationState =
  | { status: 'idle'; requestId: null; input: null; inputKey: null }
  | { status: 'pending'; phase: 'starting' | 'calculating'; requestId: number; input: TensorInput; inputKey: string }
  | { status: 'ready'; requestId: number; input: TensorInput; inputKey: string; response: TensorSuccess }
  | { status: 'error'; requestId: number; input: TensorInput | null; inputKey: string | null; error: 'invalid-input' | 'unavailable' | 'worker-failed' | 'invalid-response' };
export const idleTensorState = (): TensorCalculationState => ({ status: 'idle', requestId: null, input: null, inputKey: null });
/** One worker per active request. New work terminates the prior computation,
 * instead of queuing behind a long synchronous calculation in that worker.
 */
export class TensorCalculationController {
  private worker: TensorWorkerPort | null = null;
  private active: { id: number; input: TensorInput; inputKey: string } | null = null;
  private sequence = 0;
  private disposed = false;
  private state: TensorCalculationState = idleTensorState();
  constructor(private readonly factory: () => TensorWorkerPort, private readonly notify: (state: TensorCalculationState) => void) {}
  getSnapshot() { return this.state; }
  private publish(state: TensorCalculationState) { this.state = Object.freeze(state); this.notify(this.state); }
  private stop() {
    const port = this.worker; this.worker = null; this.active = null;
    if (port) { port.onmessage = null; port.onerror = null; port.onmessageerror = null; port.terminate(); }
  }
  run(value: unknown): number | null {
    if (this.disposed) return null;
    this.stop(); const id = ++this.sequence;
    if (!isTensorInput(value)) { this.publish({ status: 'error', requestId: id, input: null, inputKey: 'invalid-input', error: 'invalid-input' }); return id; }
    const input = snapshotTensorInput(value), inputKey = tensorInputKey(input), token = { id, input, inputKey };
    this.active = token; this.publish({ status: 'pending', phase: 'starting', requestId: id, input, inputKey });
    if (this.disposed || this.active !== token) return id;
    let port: TensorWorkerPort;
    try { port = this.factory(); }
    catch {
      if (!this.disposed && this.active === token) { this.active = null; this.publish({ status: 'error', requestId: id, input, inputKey, error: 'unavailable' }); }
      return id;
    }
    if (this.disposed || this.active !== token) { port.terminate(); return id; }
    this.worker = port;
    const tensor = () => !this.disposed && this.active === token && this.worker === port;
    const fail = (error: 'worker-failed' | 'invalid-response' | 'invalid-input') => {
      if (!tensor()) return;
      this.stop(); this.publish({ status: 'error', requestId: id, input, inputKey, error });
    };
    port.onmessage = event => {
      if (!tensor()) return;
      const message: unknown = event.data;
      if (tensorRecord(message) && Number.isSafeInteger(message.requestId) && (message.requestId as number) > 0 && message.requestId !== id) return;
      if (!tensorRecord(message) || !Object.prototype.hasOwnProperty.call(message, 'version') || !Object.prototype.hasOwnProperty.call(message, 'requestId') || message.version !== 1 || message.requestId !== id) { fail('invalid-response'); return; }
      if (message.phase === 'started' && message.kind === input.kind && Reflect.ownKeys(message).length === 4 && ['phase', 'kind'].every(k => Object.prototype.hasOwnProperty.call(message, k))) {
        this.publish({ status: 'pending', phase: 'calculating', requestId: id, input, inputKey }); return;
      }
      if (message.ok === false) {
        if (Reflect.ownKeys(message).length !== 4 || !['ok', 'error'].every(k => Object.prototype.hasOwnProperty.call(message, k)) || !['invalid-request', 'invalid-input', 'calculation-failed'].includes(message.error as string)) { fail('invalid-response'); return; }
        fail(message.error === 'invalid-input' ? 'invalid-input' : 'worker-failed'); return;
      }
      if (!isTensorSuccessForInput(message, input, id)) { fail('invalid-response'); return; }
      this.stop(); this.publish({ status: 'ready', requestId: id, input, inputKey, response: message as unknown as TensorSuccess });
    };
    port.onerror = event => { event.preventDefault(); fail('worker-failed'); };
    port.onmessageerror = () => fail('invalid-response');
    try { port.postMessage({ version: 1, requestId: id, input }); }
    catch { fail('worker-failed'); }
    return id;
  }
  cancel() { if (this.disposed) return; this.stop(); this.publish(idleTensorState()); }
  dispose() { if (this.disposed) return; this.disposed = true; this.stop(); }
}
