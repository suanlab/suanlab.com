/** Reader-safe owned integer inputs. n0 is the most significant basis bit. */
export const spinChainAxes = Object.freeze(['X', 'Y', 'Z'] as const);
export type SpinChainAxis = typeof spinChainAxes[number];
export type SpinChainParameters = Record<string, number>;
export const spinChainParameterKeys: readonly string[] = Object.freeze([
  'sites', 'graph', 'hDenominator', 'offset',
  ...Array.from({ length: 4 }, (_, i) => spinChainAxes.map(a => `field${a}${i}`)).flat(),
  ...Array.from({ length: 4 }, (_, i) => Array.from({ length: 3 - i }, (_, k) => {
    const j = i + k + 1;
    return spinChainAxes.flatMap(a => spinChainAxes.map(b => `coupling${a}${b}${i}${j}`));
  }).flat()).flat(),
  ...Array.from({ length: 16 }, (_, i) => [`stateReal${i}`, `stateImag${i}`]).flat(),
]);

/** 0=open chain,1=ring without duplicate bonds,2=all distinct custom pairs. */
export function spinChainEdges(sites: number, graph: number): [number, number][] {
  if (!Number.isInteger(sites) || sites < 1 || sites > 4 || !Number.isInteger(graph) || graph < 0 || graph > 2) {
    throw new RangeError('Spin graph needs 1..4 sites and graph code 0..2');
  }
  const edges: [number, number][] = [];
  for (let i = 0; i < sites; i++) for (let j = i + 1; j < sites; j++) {
    if (graph === 2 || j === i + 1 || graph === 1 && sites >= 3 && i === 0 && j === sites - 1) edges.push([i, j]);
  }
  return edges;
}

function readSpinChainParameters(value: unknown): SpinChainParameters | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype
      || Reflect.ownKeys(value).length !== spinChainParameterKeys.length) return null;
    const descriptors = Object.getOwnPropertyDescriptors(value);
    if (!spinChainParameterKeys.every(key => Object.hasOwn(descriptors, key) && Object.hasOwn(descriptors[key], 'value')
      && Number.isSafeInteger(descriptors[key].value) && Math.abs(descriptors[key].value) <= 1000000)) return null;
    // Validate and retain one descriptor snapshot; do not re-read caller properties.
    const p = Object.fromEntries(spinChainParameterKeys.map(key => [key, descriptors[key].value + 0])) as SpinChainParameters;
    if (p.sites < 1 || p.sites > 4 || p.graph < 0 || p.graph > 2 || p.hDenominator < 1) return null;
    const edges = new Set(spinChainEdges(p.sites, p.graph).map(([i, j]) => `${i}${j}`));
    for (let i = 0; i < 4; i++) {
      if (i >= p.sites && spinChainAxes.some(a => p[`field${a}${i}`] !== 0)) return null;
      for (let j = i + 1; j < 4; j++) if (!edges.has(`${i}${j}`)
        && spinChainAxes.some(a => spinChainAxes.some(b => p[`coupling${a}${b}${i}${j}`] !== 0))) return null;
    }
    let nonzero = false;
    for (let i = 0; i < 16; i++) {
      const nonzeroEntry = p[`stateReal${i}`] !== 0 || p[`stateImag${i}`] !== 0;
      if (i >= 2 ** p.sites && nonzeroEntry) return null;
      nonzero ||= nonzeroEntry;
    }
    return nonzero ? p : null;
  } catch { return null; }
}

export function isSpinChainParameters(value: unknown): value is SpinChainParameters {
  return readSpinChainParameters(value) !== null;
}

export function snapshotSpinChainParameters(value: SpinChainParameters): Readonly<SpinChainParameters> {
  const parameters = readSpinChainParameters(value);
  if (!parameters) throw new RangeError('102 owned integers bounded by 10^6; 1..4 sites, graph0..2, positive denominator, nonzero state and zero inactive entries required');
  return Object.freeze(parameters);
}

/** A new zero Hamiltonian and first basis vector; no previous state is truncated. */
export function createSpinChainParameters(sites = 2, graph = 0): Readonly<SpinChainParameters> {
  spinChainEdges(sites, graph);
  return snapshotSpinChainParameters({ ...Object.fromEntries(spinChainParameterKeys.map(key => [key, 0])), sites, graph, hDenominator: 1, stateReal0: 1 });
}
