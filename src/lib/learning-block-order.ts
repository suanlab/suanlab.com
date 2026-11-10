/** Exact finite spatial blocking. Foundation only; no registered reader or Worker yet. */
import { markovFraction as fraction, type MarkovFraction as Fraction } from './learning-finite-markov-exact';

export type BlockOrderInput = {
  shape: number[];
  blockShape: number[];
  boundary: 'open' | 'periodic';
  disagreementActivity: string;
  blockUpActivities: string[];
  selectedUpCounts: number[];
  witnessPair: number[];
};
const z = BigInt(0), o = BigInt(1);
const q = (n: number) => fraction(BigInt(n));
const plus = (a: Fraction, b: Fraction) => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
const minus = (a: Fraction, b: Fraction) => fraction(a.n * b.d - b.n * a.d, a.d * b.d);
const times = (a: Fraction, b: Fraction) => fraction(a.n * b.n, a.d * b.d);
const over = (a: Fraction, b: Fraction) => fraction(a.n * b.d, a.d * b.n);
const power = (a: Fraction, p: number) => p >= 0 ? fraction(a.n ** BigInt(p), a.d ** BigInt(p)) : fraction(a.d ** BigInt(-p), a.n ** BigInt(-p));
const wire = (a: Fraction) => a.d === o ? String(a.n) : `${a.n}/${a.d}`;
const product = (xs: number[]) => xs.reduce((a, b) => a * b, 1);
const parse = (s: string) => { const [n, d = '1'] = s.split('/'); return fraction(BigInt(n), BigInt(d)); };

/** JSON values only. Activities specify exp(-2J/T), exp(2H_b/T), not J or H. */
export function readBlockOrderInput(value: unknown): BlockOrderInput | null {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const keys = ['shape', 'blockShape', 'boundary', 'disagreementActivity', 'blockUpActivities', 'selectedUpCounts', 'witnessPair'];
    const descriptors = Object.getOwnPropertyDescriptors(value);
    if (Reflect.ownKeys(value).length !== keys.length || keys.some(key => !descriptors[key] || !('value' in descriptors[key]))) return null;
    const v = Object.fromEntries(keys.map(key => [key, descriptors[key].value])) as BlockOrderInput;
    const integers = (xs: unknown, length: number, low: number, high: number): xs is number[] =>
      Array.isArray(xs) && xs.length === length && Array.from(xs).every(x => Number.isInteger(x) && x >= low && x <= high);
    if (!Array.isArray(v.shape) || ![1, 2].includes(v.shape.length) || !integers(v.shape, v.shape.length, 2, 16)) return null;
    const n = product(v.shape);
    if (n > 16 || !integers(v.blockShape, v.shape.length, 1, 16) || v.shape.some((s, i) => s % v.blockShape[i] !== 0)) return null;
    const volume = product(v.blockShape), blocks = n / volume;
    if (blocks > 4 || !['open', 'periodic'].includes(v.boundary)) return null;
    const activity = (x: unknown): x is string => typeof x === 'string' && /^[1-9]\d{0,5}(\/[1-9]\d{0,5})?$/.test(x);
    if (!activity(v.disagreementActivity) || !Array.isArray(v.blockUpActivities) || v.blockUpActivities.length !== blocks || !Array.from(v.blockUpActivities).every(activity)) return null;
    if (!integers(v.selectedUpCounts, blocks, 0, volume) || !integers(v.witnessPair, blocks === 1 ? 0 : 2, 0, blocks - 1)) return null;
    if (blocks > 1 && v.witnessPair[0] >= v.witnessPair[1]) return null;
    return { shape: [...v.shape], blockShape: [...v.blockShape], boundary: v.boundary, disagreementActivity: v.disagreementActivity,
      blockUpActivities: [...v.blockUpActivities], selectedUpCounts: [...v.selectedUpCounts], witnessPair: [...v.witnessPair] };
  } catch { return null; }
}

function geometry(input: BlockOrderInput) {
  const n = product(input.shape), volume = product(input.blockShape), blocks = n / volume;
  const counts = input.shape.map((s, a) => s / input.blockShape[a]);
  const blockOf = Array.from({ length: n }, (_, site) => {
    let stride = 1, blockStride = 1, index = 0;
    input.shape.forEach((s, a) => {
      index += Math.floor((Math.floor(site / stride) % s) / input.blockShape[a]) * blockStride;
      stride *= s; blockStride *= counts[a];
    });
    return index;
  });
  const edges: [number, number][] = [];
  let stride = 1;
  input.shape.forEach(s => {
    for (let i = 0; i < n; i++) {
      const c = Math.floor(i / stride) % s;
      if (c + 1 < s) edges.push([i, i + stride]);
      else if (input.boundary === 'periodic') edges.push([i, i - (s - 1) * stride]);
    }
    stride *= s;
  });
  return { n, volume, blocks, blockOf, edges };
}

export function calculateBlockOrder(value: unknown) {
  const input = readBlockOrderInput(value);
  if (!input) throw new RangeError('Invalid finite block-order input');
  const { n, volume: v, blocks: b, blockOf, edges } = geometry(input);
  const e = edges.length, x = parse(input.disagreementActivity), ys = input.blockUpActivities.map(parse);
  const radix = v + 1, rows = radix ** b;
  const indexOf = (ks: number[]) => ks.reduce((s, k, i) => s + k * radix ** i, 0);
  const countsAt = (index: number) => Array.from({ length: b }, (_, i) => Math.floor(index / radix ** i) % radix);
  const selectedIndex = indexOf(input.selectedUpCounts);
  const histograms = Array.from({ length: rows }, () => new Uint32Array(e + 1));
  const incident = Array.from({ length: n }, () => [] as number[]);
  edges.forEach(([i, j]) => { incident[i].push(j); incident[j].push(i); });
  const bondIntegers = Array.from({ length: e + 1 }, (_, d) => x.n ** BigInt(d) * x.d ** BigInt(e - d));
  const fieldIntegers = ys.map(y => Array.from({ length: radix }, (_, k) => y.n ** BigInt(k) * y.d ** BigInt(v - k)));
  const selectedSpinSums = Array.from({ length: n }, () => z);
  const selectedExamples: { mask: number; disagreements: number; integerWeight: bigint }[] = [];
  let mask = 0, disagreements = 0;
  const up = Array.from({ length: b }, () => 0);
  // Gray traversal changes one spin and only its incident edge contributions.
  for (let step = 0; step < 2 ** n; step++) {
    if (step) {
      const site = 31 - Math.clz32(step & -step), old = (mask >> site) & 1;
      for (const j of incident[site]) disagreements += old === ((mask >> j) & 1) ? 1 : -1;
      mask ^= 1 << site; up[blockOf[site]] += old ? -1 : 1;
    }
    const index = indexOf(up);
    histograms[index][disagreements]++;
    if (index === selectedIndex) {
      const w = bondIntegers[disagreements];
      selectedSpinSums.forEach((_, i) => { selectedSpinSums[i] += ((mask >> i) & 1) ? w : -w; });
      selectedExamples.push({ mask, disagreements, integerWeight: w });
      selectedExamples.sort((a, c) => a.mask - c.mask);
      if (selectedExamples.length > 8) selectedExamples.pop();
    }
  }
  const denominator = x.d ** BigInt(e) * ys.reduce((s, y) => s * y.d ** BigInt(v), o);
  const bondDenominator = x.d ** BigInt(e);
  const weights = histograms.map((hist, i) => {
    const bond = Array.from(hist).reduce((s, count, d) => s + BigInt(count) * bondIntegers[d], z);
    const fields = countsAt(i).reduce((s, k, a) => s * fieldIntegers[a][k], o);
    return { bond, fields, weight: bond * fields };
  });
  const total = weights.reduce((s, row) => s + row.weight, z);
  const selectedBond = weights[selectedIndex].bond;
  const kSums = Array.from({ length: b }, () => z), kkSums = Array.from({ length: b }, () => Array.from({ length: b }, () => z));
  const globalUpWeights = Array.from({ length: n + 1 }, () => z);
  let dSum = z, ddSum = z;
  const profiles = weights.map(({ weight, fields, bond }, i) => {
    const ks = countsAt(i), hist = histograms[i];
    globalUpWeights[ks.reduce((s, k) => s + k, 0)] += weight;
    ks.forEach((k, a) => {
      kSums[a] += weight * BigInt(k);
      ks.forEach((l, c) => { kkSums[a][c] += weight * BigInt(k * l); });
    });
    let conditionalD = z, conditionalDD = z;
    hist.forEach((count, d) => {
      const w = BigInt(count) * bondIntegers[d];
      conditionalD += w * BigInt(d); conditionalDD += w * BigInt(d * d);
    });
    dSum += conditionalD * fields; ddSum += conditionalDD * fields;
    return { upCounts: ks, magnetizations: ks.map(k => wire(fraction(BigInt(2 * k - v), BigInt(v)))),
      multiplicity: Array.from(hist).reduce((s, count) => s + count, 0),
      disagreementHistogram: Array.from(hist).flatMap((count, d) => count ? [[d, count]] : []),
      bondPolynomial: wire(fraction(bond, bondDenominator)),
      relativeWeight: wire(fraction(weight, denominator)), probability: wire(fraction(weight, total)),
      conditionalMeanDisagreements: wire(fraction(conditionalD, bond)),
      conditionalVarianceDisagreements: wire(minus(fraction(conditionalDD, bond), power(fraction(conditionalD, bond), 2))) };
  });
  const meanK = kSums.map(s => fraction(s, total));
  const meanM = meanK.map(k => minus(times(q(2), over(k, q(v))), q(1)));
  const covariance = kkSums.map((row, a) => row.map((s, c) => wire(times(fraction(BigInt(4), BigInt(v * v)), minus(fraction(s, total), times(meanK[a], meanK[c]))))));

  // Exact within-block restricted sums use the induced graph, including wraps
  // when a block spans an entire periodic axis.
  const isolatedPolynomials = Array.from({ length: b }, (_, block) => {
    const sites = blockOf.flatMap((which, i) => which === block ? [i] : []);
    const localIndex = new Map(sites.map((site, i) => [site, i]));
    const internal = edges.filter(([i, j]) => blockOf[i] === block && blockOf[j] === block)
      .map(([i, j]) => [localIndex.get(i)!, localIndex.get(j)!]);
    let polynomial = fraction(z);
    for (let state = 0; state < 2 ** v; state++) {
      let k = 0; for (let i = 0; i < v; i++) k += (state >> i) & 1;
      if (k !== input.selectedUpCounts[block]) continue;
      const d = internal.reduce((s, [i, j]) => s + (((state >> i) & 1) === ((state >> j) & 1) ? 0 : 1), 0);
      polynomial = plus(polynomial, power(x, d));
    }
    return polynomial;
  });
  const m = input.selectedUpCounts.map(k => fraction(BigInt(2 * k - v), BigInt(v)));
  const cross = edges.filter(([i, j]) => blockOf[i] !== blockOf[j]);
  const meanCrossDisagreements = cross.reduce((s, [i, j]) => plus(s, over(minus(q(1), times(m[blockOf[i]], m[blockOf[j]])), q(2))), fraction(z));
  const isolatedProduct = isolatedPolynomials.reduce(times, q(1));
  const ratio = over(fraction(selectedBond, bondDenominator), isolatedProduct);
  const residualPower = over(power(ratio, Number(meanCrossDisagreements.d)), power(x, Number(meanCrossDisagreements.n)));
  let localityWitness = null;
  if (b > 1) {
    const [a, c] = input.witnessPair;
    const corners = [[v, v], [v, 0], [0, v], [0, 0]].map(([ka, kc]) => {
      const ks = [...input.selectedUpCounts]; ks[a] = ka; ks[c] = kc;
      return weights[indexOf(ks)].weight;
    });
    const odds = fraction(corners[0] * corners[3], corners[1] * corners[2]);
    const neighbors = cross.some(([i, j]) => (blockOf[i] === a && blockOf[j] === c) || (blockOf[i] === c && blockOf[j] === a));
    localityWitness = { pair: [a, c], pairIsNeighbor: neighbors, crossRatio: wire(odds),
      rulesOutNearestNeighborPairFactorization: !neighbors && odds.n !== odds.d,
      conditioning: 'all other block up-counts fixed; global magnetization not conditioned' };
  }
  const zRelative = fraction(total, denominator);
  const baselineSquared = ys.reduce((s, y) => times(s, power(y, -v)), power(x, -e));
  return { kind: 'finite-block-order' as const, sites: n, blocks: b, blockVolume: v, edgeCount: e,
    blockOf, edges, profiles, relativePartition: wire(zRelative), absolutePartitionSquared: wire(times(baselineSquared, power(zRelative, 2))),
    meanMagnetizations: meanM.map(wire), magnetizationCovariance: covariance,
    meanDisagreements: wire(fraction(dSum, total)), varianceDisagreements: wire(minus(fraction(ddSum, total), power(fraction(dSum, total), 2))),
    globalUpProbabilities: globalUpWeights.map(w => wire(fraction(w, total))),
    selected: { profileIndex: selectedIndex, conditionalSpinMeans: selectedSpinSums.map(s => wire(fraction(s, selectedBond))),
      examples: selectedExamples.map(({ mask: state, disagreements: d, integerWeight }) => ({ mask: state, disagreements: d, conditionalProbability: wire(fraction(integerWeight, selectedBond)) })),
      examplesAre: 'up to eight smallest numeric spin masks; examples, not random samples or a full distribution',
      isolatedBlockPolynomials: isolatedPolynomials.map(wire), meanProductCrossDisagreements: wire(meanCrossDisagreements),
      meanProductResidualPower: wire(residualPower), meanProductResidualDivisor: String(meanCrossDisagreements.d),
      residualMeaning: 'log(exact relative weight / mean-product reference weight) = log(residualPower) / residualDivisor; not a normalized KL divergence' },
    localityWitness,
    logarithmOperands: { disagreementActivity: wire(x), blockUpActivities: ys.map(wire), relativePartition: wire(zRelative),
      meanUpCounts: meanK.map(wire), absoluteLogPartitionBaseline: { logXCoefficient: wire(fraction(BigInt(-e), BigInt(2))), logYCoefficients: ys.map(() => wire(fraction(BigInt(-v), BigInt(2)))) } } };
}
