/** Reader-safe input dispatch. Numerical engines belong in the worker only. */
import { isSpinChainParameters, snapshotSpinChainParameters } from './learning-spin-chain-input';
import { isFermionFockParameters, snapshotFermionFockParameters } from './learning-fermion-fock-input';
import { isBosonFockParameters, snapshotBosonFockParameters } from './learning-boson-fock-input';
import { isBosonCoherentParameters, snapshotBosonCoherentParameters } from './learning-boson-coherent-input';
import { isMpsParameters, snapshotMpsParameters } from './learning-mps-input';
import { isSchmidtParameters, snapshotSchmidtParameters } from './learning-schmidt-input';
import { parseNumberJsonDraft } from './learning-json-number-input';

export const manyBodyKinds = Object.freeze(['spin-chain', 'fermion-fock', 'boson-fock', 'boson-coherent', 'mps', 'schmidt'] as const);
export type ManyBodyKind = typeof manyBodyKinds[number];
export type ManyBodyInput<K extends ManyBodyKind = ManyBodyKind> = Readonly<{ kind: K; parameters: Readonly<Record<string, number>> }>;
const inputs = {
  'spin-chain': { valid: isSpinChainParameters, snapshot: snapshotSpinChainParameters },
  'fermion-fock': { valid: isFermionFockParameters, snapshot: snapshotFermionFockParameters },
  'boson-fock': { valid: isBosonFockParameters, snapshot: snapshotBosonFockParameters },
  'boson-coherent': { valid: isBosonCoherentParameters, snapshot: snapshotBosonCoherentParameters },
  mps: { valid: isMpsParameters, snapshot: snapshotMpsParameters },
  schmidt: { valid: isSchmidtParameters, snapshot: snapshotSchmidtParameters },
} as const;

/** Copy owned data descriptors; never invoke a caller getter or get trap. */
export function manyBodyData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype
      || Reflect.ownKeys(value).length !== keys.length) return null;
    const result: Record<string, unknown> = {};
    for (const key of keys) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (!descriptor || !Object.hasOwn(descriptor, 'value')) return null;
      result[key] = descriptor.value;
    }
    return result;
  } catch { return null; }
}

export function readManyBodyInput(value: unknown): ManyBodyInput | null {
  try {
    const data = manyBodyData(value, ['kind', 'parameters']);
    if (!data || typeof data.kind !== 'string' || !Object.hasOwn(inputs, data.kind)) return null;
    const kind = data.kind as ManyBodyKind, family = inputs[kind];
    if (!family.valid(data.parameters)) return null;
    return Object.freeze({ kind, parameters: family.snapshot(data.parameters) });
  } catch { return null; }
}
export const isManyBodyInput = (value: unknown): value is ManyBodyInput => readManyBodyInput(value) !== null;
export function snapshotManyBodyInput<K extends ManyBodyKind>(input: ManyBodyInput<K>): ManyBodyInput<K> {
  const saved = readManyBodyInput(input);
  if (!saved) throw new RangeError('Invalid many-body preparation');
  return saved as ManyBodyInput<K>;
}
export const manyBodyInputKey = (input: ManyBodyInput): string => JSON.stringify(snapshotManyBodyInput(input));

const MAX_DRAFT_LENGTH = 32768;
export function parseManyBodyInputDraft(raw: string): ManyBodyInput | null {
  try {
    if (typeof raw !== 'string' || raw.length > MAX_DRAFT_LENGTH) return null;
    return readManyBodyInput(parseNumberJsonDraft(raw, ['parameters']));
  } catch { return null; }
}
export function parseManyBodyStoredInput(raw: string): ManyBodyInput | null {
  try {
    if (typeof raw !== 'string' || raw.length > MAX_DRAFT_LENGTH) return null;
    const data = manyBodyData(parseNumberJsonDraft(raw, ['version', 'parameters']), ['version', 'value']);
    return data?.version === 1 ? readManyBodyInput(data.value) : null;
  } catch { return null; }
}
