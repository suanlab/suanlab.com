/** Bounded descriptor-only snapshot before inspecting a worker response.
 * Shared subtrees are allowed, cycles are not. No caller get/getter is run.
 */
export function snapshotManyBodyWire(value: unknown, stringBudget = 8192): unknown {
  if (!Number.isSafeInteger(stringBudget) || stringBudget < 1 || stringBudget > 16384) throw new RangeError('Bounded string budget required');
  const completed = new WeakMap<object, unknown>(), active = new WeakSet<object>();
  let nodes = 0;
  function copy(value: unknown, depth: number): unknown {
    if (++nodes > 500000 || depth > 64) throw new RangeError('Wire data budget exceeded');
    if (value === null || typeof value === 'boolean') return value;
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.length <= stringBudget) return value;
    if (!value || typeof value !== 'object') throw new RangeError('Owned wire data required');
    if (active.has(value)) throw new RangeError('Cyclic wire data');
    if (completed.has(value)) return completed.get(value);
    const array = Array.isArray(value), prototype = Object.getPrototypeOf(value);
    if (prototype !== (array ? Array.prototype : Object.prototype)) throw new RangeError('Plain wire prototype required');
    const keys = Reflect.ownKeys(value);
    if (keys.some(key => typeof key !== 'string')) throw new RangeError('Symbol wire fields');
    let result: unknown;
    active.add(value);
    if (array) {
      const length = Object.getOwnPropertyDescriptor(value, 'length');
      if (!length || !Object.hasOwn(length, 'value') || !Number.isSafeInteger(length.value)
        || length.value < 0 || length.value > 65536 || keys.length !== length.value + 1) throw new RangeError('Dense bounded wire array required');
      const entries: unknown[] = [];
      for (let i = 0; i < length.value; i++) {
        const descriptor = Object.getOwnPropertyDescriptor(value, String(i));
        if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new RangeError('Owned array entries required');
        entries.push(copy(descriptor.value, depth + 1));
      }
      result = Object.freeze(entries);
    } else {
      if (keys.length > 512) throw new RangeError('Wire record too large');
      const record: Record<string, unknown> = {};
      for (const key of keys as string[]) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new RangeError('Owned record values required');
        // defineProperty keeps __proto__ an ordinary owned key for the schema
        // to reject, rather than changing the snapshot's prototype.
        Object.defineProperty(record, key, { value: copy(descriptor.value, depth + 1), enumerable: true });
      }
      result = Object.freeze(record);
    }
    active.delete(value); completed.set(value, result); return result;
  }
  return copy(value, 0);
}
