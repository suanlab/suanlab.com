/** Descriptor-only owned copy in short, cancellable tasks. No math on the UI thread. */
export type ManyBodyCopyOptions = {
  active?: () => boolean;
  pause?: () => Promise<void>;
  nodesPerTask?: number;
};
export class ManyBodyCopyCancelled extends Error { constructor() { super('Many-body response copy cancelled'); this.name = 'AbortError'; } }
async function pauseTask() {
  // scheduler.yield() continuations can outrank timer tasks for the entire
  // copy. A timer yields fairly to input, animation and other timer queues.
  await new Promise<void>(resolve => setTimeout(resolve, 0));
}
export async function snapshotManyBodyWireInTasks(value: unknown, options: ManyBodyCopyOptions = {}): Promise<unknown> {
  const { active = () => true, pause = pauseTask, nodesPerTask = 2048 } = options;
  if (!Number.isSafeInteger(nodesPerTask) || nodesPerTask < 1 || nodesPerTask > 2048) throw new RangeError('Copy task budget1..2048 required');
  const completed = new WeakMap<object, unknown>(), ancestors = new WeakSet<object>(); let nodes = 0;
  function* copy(value: unknown, depth: number): Generator<void, unknown> {
    if (++nodes > 500000 || depth > 64) throw new RangeError('Wire data budget exceeded');
    yield;
    if (value === null || typeof value === 'boolean') return value;
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.length <= 8192) return value;
    if (!value || typeof value !== 'object') throw new RangeError('Owned wire data required');
    if (ancestors.has(value)) throw new RangeError('Cyclic wire data'); if (completed.has(value)) return completed.get(value);
    const array = Array.isArray(value), prototype = Object.getPrototypeOf(value), keys = Reflect.ownKeys(value);
    if (prototype !== (array ? Array.prototype : Object.prototype) || keys.some(k => typeof k !== 'string')) throw new RangeError('Plain owned wire data required');
    ancestors.add(value); let result: unknown;
    if (array) {
      const length = Object.getOwnPropertyDescriptor(value, 'length');
      if (!length || !Object.hasOwn(length, 'value') || !Number.isSafeInteger(length.value) || length.value < 0 || length.value > 65536 || keys.length !== length.value + 1) throw new RangeError('Dense bounded wire array required');
      const entries: unknown[] = [];
      for (let i = 0; i < length.value; i++) {
        const d = Object.getOwnPropertyDescriptor(value, String(i)); if (!d || !Object.hasOwn(d, 'value')) throw new RangeError('Owned array entries required'); entries.push(yield* copy(d.value, depth + 1));
      }
      result = Object.freeze(entries);
    } else {
      if (keys.length > 512) throw new RangeError('Wire record too large'); const record: Record<string, unknown> = {};
      for (const key of keys as string[]) { const d = Object.getOwnPropertyDescriptor(value, key); if (!d || !Object.hasOwn(d, 'value')) throw new RangeError('Owned record values required'); Object.defineProperty(record, key, { value: yield* copy(d.value, depth + 1), enumerable: true }); }
      result = Object.freeze(record);
    }
    ancestors.delete(value); completed.set(value, result); return result;
  }
  const iterator = copy(value, 0); let count = 0, start = performance.now();
  for (;;) {
    if (!active()) throw new ManyBodyCopyCancelled();
    const next = iterator.next(); if (next.done) return next.value;
    if (++count >= nodesPerTask || performance.now() - start >= 4) { await pause(); if (!active()) throw new ManyBodyCopyCancelled(); count = 0; start = performance.now(); }
  }
}
