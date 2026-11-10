/** Reader-safe bounded input and preparations; no arithmetic engine. */
export interface QuarticAsymptoticInput { coupling: string; order: number; }
export function readQuarticAsymptoticInput(value: unknown): Readonly<QuarticAsymptoticInput> | null {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const d = Object.getOwnPropertyDescriptors(value);
    if (Reflect.ownKeys(d).length !== 2 || !d.coupling || !d.order || !('value' in d.coupling) || !('value' in d.order)) return null;
    const coupling = d.coupling.value, order = d.order.value;
    if (typeof coupling !== 'string' || !/^(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/.test(coupling) || !Number.isInteger(order) || order < 0 || order > 32) return null;
    const [n, q = '1'] = coupling.split('/');
    if (Number(n) > 16 * Number(q)) return null;
    return Object.freeze({ coupling, order });
  } catch { return null; }
}
export const quarticAsymptoticDefault = Object.freeze({ coupling: '1/10', order: 12 });
export const quarticAsymptoticExamples = Object.freeze([
  { id: 'useful-truncation', input: quarticAsymptoticDefault },
  { id: 'gaussian-exact', input: { coupling: '0', order: 12 } },
  { id: 'smaller-coupling', input: { coupling: '1/100', order: 32 } },
  { id: 'budget-before-minimum', input: { coupling: '1/1000', order: 32 } },
  { id: 'strong-coupling', input: { coupling: '1', order: 12 } },
  { id: 'maximum-coupling', input: { coupling: '16', order: 32 } },
  { id: 'zero-order', input: { coupling: '1/10', order: 0 } },
  { id: 'first-order', input: { coupling: '1/10', order: 1 } },
  { id: 'least-bound-order', input: { coupling: '1/10', order: 2 } },
  { id: 'beyond-useful-order', input: { coupling: '1/10', order: 32 } },
  { id: 'equal-omitted-terms', input: { coupling: '8/35', order: 8 } },
  { id: 'equivalent-spelling', input: { coupling: '2/20', order: 12 } },
  { id: 'tiny-positive', input: { coupling: '1/999999999989', order: 32 } },
  { id: 'maximum-coprime', input: { coupling: '999999999983/62499999999', order: 32 } },
].map(example => Object.freeze({ id: example.id, input: Object.freeze(example.input) })));
