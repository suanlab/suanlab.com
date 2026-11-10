/** Small, owned inputs; no numerical solver is imported into the reader. */
export interface RegularPerturbationInput {
  stiffness: string;
  coupling: string;
  source: string;
  order: number;
}

export const regularPerturbationLimits = Object.freeze({ order: 16, fractionDigits: 12 });
const fraction = /^(-?(?:0|[1-9]\d{0,11}))(?:\/([1-9]\d{0,11}))?$/;

export function readRegularPerturbationInput(value: unknown): RegularPerturbationInput | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  try {
    const descriptors = Object.getOwnPropertyDescriptors(value);
    const keys = ['stiffness', 'coupling', 'source', 'order'];
    if (Reflect.ownKeys(descriptors).length !== keys.length || keys.some(key => !descriptors[key] || !('value' in descriptors[key]))) return null;
    const { stiffness, coupling, source, order } = Object.fromEntries(keys.map(key => [key, descriptors[key].value]));
    if (!Number.isInteger(order) || order < 0 || order > regularPerturbationLimits.order) return null;
    const parsed = [stiffness, coupling, source].map(text => typeof text === 'string' ? fraction.exec(text) : null);
    if (parsed.some(x => !x)) return null;
    const [[kn, kd], [ln, ld], [hn, hd]] = parsed.map(x => [Number(x![1]), Number(x![2] ?? '1')]);
    // These integer cross-products remain below 2^53 for the declared grammar.
    if (kn <= 0 || 1000 * kn < kd || kn > 16 * kd || ln < 0 || ln > 16 * ld || Math.abs(hn) > 16 * hd) return null;
    return Object.freeze({ stiffness, coupling, source, order });
  } catch { return null; }
}

export const regularPerturbationDefault: Readonly<RegularPerturbationInput> = Object.freeze({
  stiffness: '1', coupling: '1/10', source: '1', order: 4,
});

export const regularPerturbationExamples = Object.freeze([
  { id: 'weak-source', input: regularPerturbationDefault },
  { id: 'linear-limit', input: { stiffness: '2', coupling: '0', source: '3', order: 6 } },
  { id: 'zero-source', input: { stiffness: '1', coupling: '16', source: '0', order: 16 } },
  { id: 'negative-source', input: { stiffness: '1', coupling: '1/10', source: '-1', order: 4 } },
  { id: 'small-coupling-large-source', input: { stiffness: '1', coupling: '1/100', source: '4', order: 10 } },
  { id: 'soft-stiffness', input: { stiffness: '1/10', coupling: '1/1000', source: '1', order: 8 } },
  { id: 'stiff-response', input: { stiffness: '4', coupling: '1', source: '1', order: 8 } },
  { id: 'critical-radius', input: { stiffness: '3', coupling: '4', source: '1', order: 12 } },
  { id: 'outside-radius', input: { stiffness: '1', coupling: '1', source: '1', order: 10 } },
  { id: 'exact-rational-root', input: { stiffness: '1', coupling: '1', source: '2', order: 4 } },
  { id: 'zeroth-order', input: { stiffness: '1', coupling: '1/10', source: '1', order: 0 } },
  { id: 'equivalent-scale', input: { stiffness: '2', coupling: '1/5', source: '2', order: 4 } },
  { id: 'fractional-coefficients', input: { stiffness: '3/2', coupling: '2/7', source: '-5/3', order: 7 } },
  { id: 'maximum-soft', input: { stiffness: '1/1000', coupling: '16', source: '16', order: 16 } },
].map(example => Object.freeze({ id: example.id, input: Object.freeze(example.input) })));
