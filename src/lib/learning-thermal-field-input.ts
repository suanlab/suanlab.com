/** Reader-safe finite geometry and local thermal jets. No model engine imports. */
export interface ThermalJetInput { value: string; first: string; second: string }
export interface ThermalFieldInput {
  shape: number[];
  boundary: 'open' | 'periodic' | 'dirichlet';
  constraint: 'none' | 'mean-zero';
  spacing: string;
  temperature: string;
  referenceScale: string;
  mass: ThermalJetInput;
  stiffness: ThermalJetInput;
  backgroundDensity: ThermalJetInput;
  sourceDensity: ThermalJetInput[];
}
export function thermalFieldData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== keys.length) return null;
    const own: Record<string, unknown> = {};
    for (const key of keys) {
      const d = Object.getOwnPropertyDescriptor(value, key);
      if (!d || !Object.hasOwn(d, 'value')) return null;
      Object.defineProperty(own, key, { value: d.value, enumerable: true });
    }
    return own;
  } catch { return null; }
}
function array(value: unknown, minimum: number, maximum = minimum): unknown[] | null {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) return null;
  const descriptor = Object.getOwnPropertyDescriptor(value, 'length');
  if (!descriptor || !Object.hasOwn(descriptor, 'value') || descriptor.value < minimum || descriptor.value > maximum) return null;
  const length = descriptor.value;
  if (Reflect.ownKeys(value).length !== length + 1) return null;
  const owned = [];
  for (let i = 0; i < length; i++) {
    const d = Object.getOwnPropertyDescriptor(value, String(i));
    if (!d || !Object.hasOwn(d, 'value')) return null;
    owned.push(d.value);
  }
  return owned;
}
function rational(value: unknown): string | null {
  if (typeof value !== 'string' || value.length > 38 || !/^-?(?:0|[1-9][0-9]{0,17})(?:\/[1-9][0-9]{0,17})?$/.test(value)) return null;
  const [n, d = '1'] = value.split('/');
  let numerator = BigInt(n), denominator = BigInt(d), a = numerator < BigInt(0) ? -numerator : numerator, b = denominator;
  while (b) [a, b] = [b, a % b];
  numerator /= a; denominator /= a;
  return denominator === BigInt(1) ? String(numerator) : `${numerator}/${denominator}`;
}
function jet(value: unknown): ThermalJetInput | null {
  const own = thermalFieldData(value, ['value', 'first', 'second']);
  if (!own) return null;
  const value0 = rational(own.value), first = rational(own.first), second = rational(own.second);
  return value0 !== null && first !== null && second !== null ? Object.freeze({ value: value0, first, second }) : null;
}
const fields = ['shape', 'boundary', 'constraint', 'spacing', 'temperature', 'referenceScale', 'mass', 'stiffness', 'backgroundDensity', 'sourceDensity'] as const;
export function readThermalFieldInput(value: unknown): Readonly<ThermalFieldInput> | null {
  try {
    const own = thermalFieldData(value, fields);
    if (!own) return null;
    const shape = array(own.shape, 1, 2);
    if (!shape || !shape.every(n => typeof n === 'number' && Number.isInteger(n) && n >= 2 && n <= 32)) return null;
    const dimensions = shape as number[], sites = dimensions.reduce((a, b) => a * b, 1);
    if (sites > 32 || !['open', 'periodic', 'dirichlet'].includes(own.boundary as string) || !['none', 'mean-zero'].includes(own.constraint as string)) return null;
    const spacing = rational(own.spacing), temperature = rational(own.temperature), referenceScale = rational(own.referenceScale);
    if (![spacing, temperature, referenceScale].every(x => x !== null && BigInt(x.split('/')[0]) > BigInt(0))) return null;
    const mass = jet(own.mass), stiffness = jet(own.stiffness), backgroundDensity = jet(own.backgroundDensity), sources = array(own.sourceDensity, sites);
    if (!mass || !stiffness || !backgroundDensity || !sources || stiffness.value.startsWith('-')) return null;
    const sourceDensity = sources.map(jet);
    if (sourceDensity.some(x => x === null)) return null;
    return Object.freeze({ shape: Object.freeze(dimensions) as unknown as number[], boundary: own.boundary as ThermalFieldInput['boundary'],
      constraint: own.constraint as ThermalFieldInput['constraint'], spacing: spacing!, temperature: temperature!, referenceScale: referenceScale!,
      mass, stiffness, backgroundDensity, sourceDensity: Object.freeze(sourceDensity) as ThermalJetInput[] });
  } catch { return null; }
}
const jet0 = (value: string, first = '0', second = '0'): ThermalJetInput => ({ value, first, second });
const rawDefault: ThermalFieldInput = { shape: [2], boundary: 'open', constraint: 'none', spacing: '1', temperature: '2', referenceScale: '1',
  mass: jet0('1'), stiffness: jet0('1'), backgroundDensity: jet0('0'), sourceDensity: [jet0('1'), jet0('0')] };
export const thermalFieldDefault = readThermalFieldInput(rawDefault)!;
const example = (id: string, patch: Partial<ThermalFieldInput>) => {
  const input = readThermalFieldInput({ ...thermalFieldDefault, ...patch });
  if (!input) throw Error(`Invalid thermal field example ${id}`);
  return Object.freeze({ id, input });
};
export const thermalFieldExamples = Object.freeze([
  example('coupled-source', {}),
  example('fixed-independent', { stiffness: jet0('0'), sourceDensity: [jet0('0'), jet0('0')] }),
  example('linear-temperature', { mass: jet0('2', '1'), stiffness: jet0('0'), sourceDensity: [jet0('0'), jet0('0')] }),
  example('squared-temperature', { mass: jet0('4', '4', '2'), stiffness: jet0('0'), sourceDensity: [jet0('0'), jet0('0')] }),
  example('same-law-background', { backgroundDensity: jet0('41', '33', '14') }),
  example('reference-measure', { referenceScale: '2' }),
  example('coordinate-rescaling', { mass: jet0('4'), stiffness: jet0('4'), sourceDensity: [jet0('2'), jet0('0')], referenceScale: '1/2' }),
  example('massless-divergence', { boundary: 'periodic', mass: jet0('0'), sourceDensity: [jet0('0'), jet0('0')] }),
  example('massless-constraint', { boundary: 'periodic', constraint: 'mean-zero', mass: jet0('0'), sourceDensity: [jet0('0'), jet0('0')] }),
  example('negative-mass-stabilized', { boundary: 'periodic', constraint: 'mean-zero', mass: jet0('-1') }),
  example('zero-pivot-indefinite', { mass: jet0('-1') }),
  example('thermal-source', { sourceDensity: [jet0('1', '2', '3'), jet0('-1', '1', '-2')], mass: jet0('2', '-1/3', '1/5') }),
  example('two-dimensional', { shape: [2, 3], boundary: 'dirichlet', spacing: '3/2', sourceDensity: Array.from({ length: 6 }, (_, i) => jet0(i === 2 ? '1' : '0')) }),
  example('tiny-determinant', { shape: [32], spacing: '1/999999999999999999', mass: jet0('1/999999999999999999'),
    stiffness: jet0('0'), sourceDensity: Array.from({ length: 32 }, () => jet0('0')) }),
]);
