/** Explicit finite-bit coordinate sampler, separate from the scientific law.
 * Q describes independent uniform grid draws; the seeded LCG trace is a
 * deterministic illustration, not an iid real-uniform source or physical time.
 */
import type { finiteGibbsModel, GibbsKernelName } from './learning-finite-gibbs';
import type { PartitionObservation, PartitionRational } from './learning-partition-exact-core';
export interface FiniteGibbsPathOptions { seed: number; bits: number; steps: number; mode: GibbsKernelName }
export type FiniteGibbsPathResult = {
  status: 'unavailable'; reason: 'uncertain-grid-boundary'; options: FiniteGibbsPathOptions;
} | {
  status: 'ready'; options: FiniteGibbsPathOptions; prng: 'LCG32-high-bits'; initialStateIndex: number;
  states: number[]; observedCounts: number[]; randomDrawCount: number; finalPrngState: number;
  coordinatesResampled: number; quantizedKernel: { denominatorPower: number; numerators: string[][] };
  quantizedIrreducible: boolean; oneStepTVBound: { numerator: number; denominatorPower: number };
  contract: string;
};
const Z = BigInt(0), O = BigInt(1);
interface Fraction { n: bigint; d: bigint }
function gcd(a: bigint, b: bigint) { a = a < Z ? -a : a; while (b) { const r = a % b; a = b; b = r; } return a; }
function fraction(n: bigint, d: bigint): Fraction { const g = gcd(n, d); return { n: n / g, d: d / g }; }
function read(q: PartitionRational): Fraction {
  let n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient);
  const shift = q.numerator.exponent - q.denominator.exponent;
  if (shift >= 0) n <<= BigInt(shift); else d <<= BigInt(-shift);
  return fraction(n, d);
}
const add = (a: Fraction, b: Fraction) => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
const ceil = (n: bigint, d: bigint) => (n + d - O) / d;
function energyKey(v: PartitionObservation) { const q = read(v.bounds!.lower); return `${q.n}/${q.d}`; }
/** Exact CDF thresholds for the left grid R/2^b < CDF. Never use p.value:
 * it may be null or may have rounded a tiny complement away.
 */
function quantizeRow(row: PartitionObservation[], possible: boolean[], energyKeys: string[], bits: number): number[] | null {
  const M = O << BigInt(bits), eligible = possible.flatMap((v, i) => v ? [i] : []), thresholds = [Z];
  let lower: Fraction = { n: Z, d: O }, upper = lower;
  for (let j = 0; j < row.length - 1; j++) {
    if (!row[j].bounds) return null;
    lower = add(lower, read(row[j].bounds!.lower)); upper = add(upper, read(row[j].bounds!.upper));
    const prefix = eligible.filter(i => i <= j), suffix = eligible.filter(i => i > j);
    if (!prefix.length) { thresholds.push(Z); continue; }
    if (!suffix.length) { thresholds.push(M); continue; }
    // Equal energy multisets imply exactly equal Boltzmann sums at either
    // sign of nonzero temperature. This resolves common exact half boundaries.
    if (prefix.length === suffix.length && prefix.map(i => energyKeys[i]).sort().join('|') === suffix.map(i => energyKeys[i]).sort().join('|')) {
      thresholds.push(M / BigInt(2)); continue;
    }
    let lo = ceil(lower.n * M, lower.d), hi = ceil(upper.n * M, upper.d);
    if (lo === Z) lo = O; // A nonempty allowed prefix is strictly positive.
    if (hi > M) hi = M; // A CDF is analytically at most one.
    if (lo !== hi) return null;
    thresholds.push(lo);
  }
  thresholds.push(M);
  const counts = row.map((_, i) => thresholds[i + 1] - thresholds[i]);
  return counts.some(n => n < Z) ? null : counts.map(Number);
}
export function finiteGibbsPath(model: ReturnType<typeof finiteGibbsModel>, options: FiniteGibbsPathOptions): FiniteGibbsPathResult {
  if (!options || !Number.isInteger(options.seed) || options.seed < 0 || options.seed > 0xffffffff
    || !Number.isInteger(options.bits) || options.bits < 1 || options.bits > 32 || !Number.isInteger(options.steps) || options.steps < 1 || options.steps > 2048
    || !['first', 'second', 'random', 'systematic', 'blocked'].includes(options.mode)) throw new RangeError('Invalid finite-bit Gibbs path options');
  const snapshot = { seed: options.seed, bits: options.bits, steps: options.steps, mode: options.mode };
  const n = model.states.length, keys = model.states.map(s => energyKey(s.energy)), M = 2 ** options.bits;
  const Q1: number[][] = [], Q2: number[][] = [], QB: number[][] = [];
  for (let i = 0; i < n; i++) {
    const first = model.states.map(s => s.spins[1] === model.states[i].spins[1]), second = model.states.map(s => s.spins[0] === model.states[i].spins[0]);
    const a = quantizeRow(model.kernels.first[i], first, keys, options.bits), b = quantizeRow(model.kernels.second[i], second, keys, options.bits);
    const blocked = quantizeRow(model.target, model.states.map(() => true), keys, options.bits);
    // Only quantize kernels used by this requested update rule.
    if ((options.mode === 'first' || options.mode === 'random' || options.mode === 'systematic') && !a
      || (options.mode === 'second' || options.mode === 'random' || options.mode === 'systematic') && !b
      || options.mode === 'blocked' && !blocked) return { status: 'unavailable', reason: 'uncertain-grid-boundary', options: snapshot };
    Q1.push(a ?? []); Q2.push(b ?? []); QB.push(blocked ?? []);
  }
  const make = (rows: number[][]) => rows.map(row => row.map(BigInt));
  let matrix: bigint[][], denominatorPower = options.bits;
  if (options.mode === 'first') matrix = make(Q1);
  else if (options.mode === 'second') matrix = make(Q2);
  else if (options.mode === 'blocked') matrix = make(QB);
  else if (options.mode === 'random') { denominatorPower++; matrix = Q1.map((row, i) => row.map((v, j) => BigInt(v) + BigInt(Q2[i][j]))); }
  else { denominatorPower *= 2; matrix = Q1.map(row => model.states.map((_, j) => row.reduce((s, v, k) => s + BigInt(v) * BigInt(Q2[k][j]), Z))); }
  const reachable = matrix.map(row => row.map(v => v > Z));
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) reachable[i][j] ||= reachable[i][k] && reachable[k][j];
  let seed = options.seed >>> 0, randomDrawCount = 0;
  const next = () => { randomDrawCount++; seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
  const grid = () => options.bits === 32 ? next() : next() >>> (32 - options.bits);
  function draw(row: number[]) {
    const r = grid(); let cumulative = 0;
    for (let i = 0; i < n; i++) { cumulative += row[i]; if (r < cumulative) return i; }
    throw new Error('Unnormalized quantized row');
  }
  let state = model.initialStateIndex;
  const states = [state], observedCounts = model.states.map((_, i) => i === state ? 1 : 0);
  for (let t = 0; t < options.steps; t++) {
    if (options.mode === 'first') state = draw(Q1[state]);
    else if (options.mode === 'second') state = draw(Q2[state]);
    else if (options.mode === 'blocked') state = draw(QB[state]);
    else if (options.mode === 'random') state = (next() >>> 31) === 0 ? draw(Q1[state]) : draw(Q2[state]);
    else { state = draw(Q1[state]); state = draw(Q2[state]); }
    states.push(state); observedCounts[state]++;
  }
  // Each coordinate conditional has <=2 allowed outcomes (TV<=1/M).
  // Mixture preserves the bound; composing two updates adds their bounds.
  const numerator = options.mode === 'systematic' ? 2 : options.mode === 'blocked' ? n - 1 : 1;
  return { status: 'ready', options: snapshot, prng: 'LCG32-high-bits', initialStateIndex: model.initialStateIndex,
    states, observedCounts, randomDrawCount, finalPrngState: seed, coordinatesResampled: options.steps * (options.mode === 'systematic' || options.mode === 'blocked' ? 2 : 1),
    quantizedKernel: { denominatorPower, numerators: matrix.map(row => row.map(String)) }, quantizedIrreducible: reachable.every(row => row.every(Boolean)),
    oneStepTVBound: { numerator, denominatorPower: options.bits },
    contract: `Left grid R/${M}; CDF ceilings certified from rational bounds, with exact energy symmetry where applicable. Q and its one-step TV bound assume independent uniform grid draws. LCG32-high-bits is a deterministic illustrative trace, not an iid source, exact real Gibbs sampling or physical time. The scientific target/kernel are unchanged; quantization may remove their positive edges.` };
}
