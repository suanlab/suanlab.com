/** Independent row-transfer reconstruction. Never imports producer engines,
 * histogram evaluators, fraction arithmetic, logarithms or reference fixtures.
 */
import { CheckerRational as Q, checkerInteger as integer, checkerZero as ZERO, checkerOne as ONE, checkerCanonicalReader } from './learning-real-time-validation-rational';
import { createRealTimeCheckerInterval, type CheckerInterval } from './learning-real-time-checker-interval';
import { blockOrderData, readBlockOrderInput, type BlockOrderInput } from './learning-block-order-input';
import { snapshotQubitWire } from './learning-qubit-wire';
import type { BlockOrderResult } from './learning-block-order-protocol';

const sum = (xs: Q[]) => xs.reduce((a, b) => a.add(b), ZERO);
const multiply = (xs: Q[]) => xs.reduce((a, b) => a.mul(b), ONE);
const pow = (q: Q, k: number) => k >= 0 ? Q.of(q.n ** BigInt(k), q.d ** BigInt(k)) : Q.of(q.d ** BigInt(-k), q.n ** BigInt(-k));
const bits = (value: number) => { let count = 0; for (let n = value; n; n &= n - 1) count++; return count; };

function reconstruct(input: BlockOrderInput) {
  const shape = input.shape, n = shape.reduce((a, b) => a*b, 1), v = input.blockShape.reduce((a, b) => a*b, 1), b = n/v;
  const periodic = input.boundary === 'periodic', e = shape.reduce((s, length) => s + (periodic ? n : n*(length-1)/length), 0);
  const rowCount = (v+1)**b, degreeCount = e+1, x = Q.input(input.disagreementActivity), ys = input.blockUpActivities.map(Q.input);
  const coordinate = Array.from({ length: n }, (_, i) => shape.length === 1 ? [i] : [i % shape[0], Math.floor(i/shape[0])]);
  const addresses = new Map(coordinate.map((c, i) => [c.join(','), i]));
  const blockOf = coordinate.map(c => c.reduce((s, t, axis) => s + Math.floor(t/input.blockShape[axis])*(axis ? shape[0]/input.blockShape[0] : 1), 0));
  const edges: [number,number][] = [];
  for (let axis = 0; axis < shape.length; axis++) for (const [i, c] of coordinate.entries()) {
    if (c[axis] + 1 < shape[axis] || periodic) {
      const next = [...c]; next[axis] = (next[axis]+1) % shape[axis];
      edges.push([i, addresses.get(next.join(','))!]);
    }
  }
  const axisWidth = shape.length === 1 ? -1 : shape[0] <= shape[1] ? 0 : 1;
  const width = axisWidth < 0 ? 1 : shape[axisWidth], height = n/width, states = 2**width;
  const siteAt = (row: number, col: number) => axisWidth < 0 ? row : axisWidth === 0 ? col+shape[0]*row : row+shape[0]*col;
  const rowUp = Array.from({ length: height }, (_, row) => Array.from({ length: states }, (_, state) => {
    let index = 0;
    for (let col = 0; col < width; col++) if ((state >> col) & 1) index += (v+1)**blockOf[siteAt(row,col)];
    return index;
  }));
  const rowD = Array.from({ length: states }, (_, state) => {
    let d = 0;
    for (let col = 0; col < width-1; col++) if (((state >> col) & 1) !== ((state >> (col+1)) & 1)) d++;
    if (periodic && width > 1 && (state & 1) !== ((state >> (width-1)) & 1)) d++;
    return d;
  });
  const address = (first: number, last: number, profile: number, d: number) => ((first*states+last)*rowCount+profile)*degreeCount+d;
  let transfer = new Map<number,number>();
  for (let state = 0; state < states; state++) transfer.set(address(periodic ? state : 0,state,rowUp[0][state],rowD[state]),1);
  for (let row = 1; row < height; row++) {
    const next = new Map<number,number>();
    for (const [key, count] of transfer) {
      const d = key % degreeCount, rest = Math.floor(key/degreeCount), profile = rest % rowCount;
      const last = Math.floor(rest/rowCount) % states, first = Math.floor(rest/(rowCount*states));
      for (let state = 0; state < states; state++) {
        const at = address(first,state,profile+rowUp[row][state],d+bits(last^state)+rowD[state]);
        next.set(at,(next.get(at) ?? 0)+count);
      }
    }
    transfer = next;
  }
  const hist = Array.from({ length: rowCount }, () => new Uint32Array(degreeCount));
  for (const [key, count] of transfer) {
    let d = key % degreeCount;
    const rest = Math.floor(key/degreeCount), profile = rest % rowCount, last = Math.floor(rest/rowCount) % states, first = Math.floor(rest/(rowCount*states));
    if (periodic) d += bits(first^last);
    hist[profile][d] += count;
  }
  const counts = Array.from({ length: rowCount }, (_, i) => Array.from({ length: b }, (_, a) => Math.floor(i/(v+1)**a) % (v+1)));
  const xPowers = Array.from({ length: degreeCount }, (_, d) => pow(x,d));
  const evaluate = (h: Uint32Array, moment = 0) => sum(Array.from(h).flatMap((count,d) => count ? [xPowers[d].mul(integer(count*d**moment))] : []));
  const bonds = hist.map(h => evaluate(h)), conditionalD = hist.map((h,i) => evaluate(h,1).div(bonds[i]));
  const conditionalDD = hist.map((h,i) => evaluate(h,2).div(bonds[i]));
  const weights = counts.map((ks,i) => bonds[i].mul(multiply(ks.map((k,a) => pow(ys[a],k)))));
  const total = sum(weights), probabilities = weights.map(w => w.div(total));
  const meanK = Array.from({ length: b }, (_, a) => sum(probabilities.map((p,i) => p.mul(integer(counts[i][a])))));
  const meanD = sum(probabilities.map((p,i) => p.mul(conditionalD[i]))), meanDD = sum(probabilities.map((p,i) => p.mul(conditionalDD[i])));
  const profiles = counts.map((ks,i) => ({ upCounts: ks, magnetizations: ks.map(k => Q.of(BigInt(2*k-v),BigInt(v)).text()),
    multiplicity: Array.from(hist[i]).reduce((s,c) => s+c,0), disagreementHistogram: Array.from(hist[i]).flatMap((c,d) => c ? [[d,c]] : []),
    bondPolynomial: bonds[i].text(), relativeWeight: weights[i].text(), probability: probabilities[i].text(),
    conditionalMeanDisagreements: conditionalD[i].text(), conditionalVarianceDisagreements: conditionalDD[i].sub(pow(conditionalD[i],2)).text() }));
  const selectedIndex = input.selectedUpCounts.reduce((s,k,a) => s+k*(v+1)**a,0);
  // Direct binary enumeration, independently tallying per-site up-count
  // polynomials before exact evaluation; no Gray updates or weighted spin sums.
  const upHist = Array.from({ length: n }, () => new Uint32Array(degreeCount));
  const examples = [];
  for (let state = 0; state < 2**n; state++) {
    const ks = Array.from({ length: b }, () => 0);
    for (let i = 0; i < n; i++) ks[blockOf[i]] += (state >> i) & 1;
    if (!ks.every((k,a) => k === input.selectedUpCounts[a])) continue;
    const d = edges.reduce((s,[i,j]) => s+Number(((state >> i) & 1) !== ((state >> j) & 1)),0);
    for (let i = 0; i < n; i++) if ((state >> i) & 1) upHist[i][d]++;
    if (examples.length < 8) examples.push({ mask: state, disagreements: d, conditionalProbability: xPowers[d].div(bonds[selectedIndex]).text() });
  }
  const isolated = Array.from({ length: b }, (_, a) => {
    const sites = blockOf.flatMap((which,i) => which === a ? [i] : []), local = new Map(sites.map((site,i) => [site,i]));
    const internal = edges.filter(([i,j]) => local.has(i) && local.has(j)).map(([i,j]) => [local.get(i)!,local.get(j)!]);
    const polynomial = new Uint32Array(degreeCount);
    for (let state = 0; state < 2**v; state++) if (bits(state) === input.selectedUpCounts[a]) {
      const d = internal.reduce((s,[i,j]) => s+Number(((state >> i) & 1) !== ((state >> j) & 1)),0);
      polynomial[d]++;
    }
    return evaluate(polynomial);
  });
  const selectedM = input.selectedUpCounts.map(k => Q.of(BigInt(2*k-v),BigInt(v)));
  const cross = edges.filter(([i,j]) => blockOf[i] !== blockOf[j]);
  const dbar = sum(cross.map(([i,j]) => ONE.sub(selectedM[blockOf[i]].mul(selectedM[blockOf[j]])).div(integer(2))));
  const residual = pow(bonds[selectedIndex].div(multiply(isolated)),Number(dbar.d)).div(pow(x,Number(dbar.n)));
  let localityWitness = null;
  if (b > 1) {
    const [a,c] = input.witnessPair;
    const corners = [[v,v],[v,0],[0,v],[0,0]].map(([ka,kc]) => {
      const ks = [...input.selectedUpCounts]; ks[a] = ka; ks[c] = kc;
      return weights[ks.reduce((s,k,t) => s+k*(v+1)**t,0)];
    });
    const odds = corners[0].mul(corners[3]).div(corners[1].mul(corners[2]));
    const neighbor = cross.some(([i,j]) => [blockOf[i],blockOf[j]].includes(a) && [blockOf[i],blockOf[j]].includes(c));
    localityWitness = { pair: [a,c], pairIsNeighbor: neighbor, crossRatio: odds.text(), rulesOutNearestNeighborPairFactorization: !neighbor && !odds.equals(ONE),
      conditioning: 'all other block up-counts fixed; global magnetization not conditioned' };
  }
  return { kind: 'finite-block-order' as const, sites: n, blocks: b, blockVolume: v, edgeCount: e, blockOf, edges, profiles,
    relativePartition: total.text(), absolutePartitionSquared: pow(total,2).mul(pow(x,-e)).mul(multiply(ys.map(y => pow(y,-v)))).text(),
    meanMagnetizations: meanK.map(k => k.mul(integer(2)).div(integer(v)).sub(ONE).text()),
    magnetizationCovariance: meanK.map((ka,a) => meanK.map((kc,c) => sum(probabilities.map((p,i) => p.mul(integer(counts[i][a]*counts[i][c])))).sub(ka.mul(kc)).mul(Q.of(BigInt(4),BigInt(v*v))).text())),
    meanDisagreements: meanD.text(), varianceDisagreements: meanDD.sub(pow(meanD,2)).text(),
    globalUpProbabilities: Array.from({ length: n+1 }, (_, k) => sum(probabilities.filter((_,i) => counts[i].reduce((s,t) => s+t,0) === k)).text()),
    selected: { profileIndex: selectedIndex, conditionalSpinMeans: upHist.map(h => evaluate(h).mul(integer(2)).div(bonds[selectedIndex]).sub(ONE).text()), examples,
      examplesAre: 'up to eight smallest numeric spin masks; examples, not random samples or a full distribution',
      isolatedBlockPolynomials: isolated.map(q => q.text()), meanProductCrossDisagreements: dbar.text(), meanProductResidualPower: residual.text(), meanProductResidualDivisor: String(dbar.d),
      residualMeaning: 'log(exact relative weight / mean-product reference weight) = log(residualPower) / residualDivisor; not a normalized KL divergence' },
    localityWitness,
    logarithmOperands: { disagreementActivity: x.text(), blockUpActivities: ys.map(y => y.text()), relativePartition: total.text(), meanUpCounts: meanK.map(k => k.text()),
      absoluteLogPartitionBaseline: { logXCoefficient: Q.of(BigInt(-e),BigInt(2)).text(), logYCoefficients: ys.map(() => Q.of(BigInt(-v),BigInt(2)).text()) } } };
}

const context = createRealTimeCheckerInterval(2048);
function observationTruth(exact: ReturnType<typeof reconstruct>) {
  const cache = new Map<string,CheckerInterval>();
  const log = (s: string) => {
    let r = cache.get(s);
    if (!r) {
      const q = Q.input(s);
      r = context.sub(context.log(context.point(q.n*context.S)),context.log(context.point(q.d*context.S)));
      cache.set(s,r);
    }
    return r;
  };
  const scaled = (s: string, c: string) => context.multiply(log(s),context.from(Q.input(c)));
  const lz = log(exact.relativePartition), logZ = context.divideInteger(log(exact.absolutePartitionSquared),2), form = exact.logarithmOperands;
  let expectedLogQ = context.integer(0), fields = context.integer(0);
  for (const row of exact.profiles) expectedLogQ = context.add(expectedLogQ,scaled(row.bondPolynomial,row.probability));
  for (let a = 0; a < exact.blocks; a++) fields = context.add(fields,scaled(form.blockUpActivities[a],form.meanUpCounts[a]));
  const hidden = context.sub(expectedLogQ,scaled(form.disagreementActivity,exact.meanDisagreements));
  const profile = context.sub(context.sub(lz,expectedLogQ),fields), microscopic = context.add(profile,hidden);
  const row = exact.profiles[exact.selected.profileIndex];
  const selectedHidden = context.sub(log(row.bondPolynomial),scaled(form.disagreementActivity,row.conditionalMeanDisagreements));
  const selectedFree = context.neg(context.add(log(row.probability),logZ));
  let energy = scaled(form.disagreementActivity,Q.of(BigInt(exact.edgeCount),BigInt(2)).sub(Q.input(row.conditionalMeanDisagreements)).text());
  for (let a = 0; a < exact.blocks; a++) energy = context.add(energy,scaled(form.blockUpActivities[a],Q.of(BigInt(exact.blockVolume),BigInt(2)).sub(integer(row.upCounts[a])).text()));
  return { logRelativePartition: lz, logPartition: logZ, freeEnergyOverTemperature: context.neg(logZ), microscopicEntropy: microscopic,
    profileEntropy: profile, meanHiddenEntropy: hidden, selectedHiddenEntropy: selectedHidden, selectedFreeEnergyOverTemperature: selectedFree,
    selectedMeanEnergyOverTemperature: energy,
    meanProductLogWeightResidual: context.divideInteger(log(exact.selected.meanProductResidualPower),BigInt(exact.selected.meanProductResidualDivisor)),
    localityLogCrossRatio: exact.localityWitness ? log(exact.localityWitness.crossRatio) : null };
}
function same(a: unknown, b: unknown): boolean {
  if (!b || typeof b !== 'object') return a === b;
  if (!a || typeof a !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(b)) return Array.isArray(a) && a.length === b.length && b.every((v,i) => same(a[i],v));
  const left = a as Record<string,unknown>, right = b as Record<string,unknown>, keys = Object.keys(right);
  return Object.keys(left).length === keys.length && keys.every(k => Object.hasOwn(left,k) && same(left[k],right[k]));
}
const expectedCache = new Map<string,{ exact: unknown; truth: ReturnType<typeof observationTruth> }>();
export function validateBlockOrderResult(source: BlockOrderInput, candidate: unknown): BlockOrderResult | null {
  try {
    const input = readBlockOrderInput(source);
    if (!input) return null;
    const owned = snapshotQubitWire(candidate), data = blockOrderData(owned,['exact','observations']);
    if (!data) return null;
    const key = JSON.stringify(input);
    let expected = expectedCache.get(key);
    if (!expected) {
      const raw = reconstruct(input), { profiles, ...rest } = raw;
      const chunks = [];
      for (let i = 0; i < profiles.length; i += 128) chunks.push(profiles.slice(i,i+128));
      expected = { exact: { ...rest, profileChunks: chunks }, truth: observationTruth(raw) };
      if (expectedCache.size === 2) expectedCache.delete(expectedCache.keys().next().value!);
      expectedCache.set(key,expected);
    }
    if (!same(data.exact,expected.exact)) return null;
    const fields = Object.keys(expected.truth), obs = blockOrderData(data.observations,['precisionBits','interpretation',...fields]);
    if (!obs || obs.precisionBits !== 512 || obs.interpretation !== 'finite-discrete-profile-counting-measure') return null;
    const canonical = checkerCanonicalReader(), denominator = BigInt(1) << BigInt(512);
    for (const field of fields) {
      const truth = expected.truth[field as keyof typeof expected.truth];
      if (truth === null) { if (obs[field] !== null) return null; continue; }
      const bound = blockOrderData(obs[field],['lower','upper']);
      if (!bound) return null;
      const lo = canonical(bound.lower), hi = canonical(bound.upper), width = hi.sub(lo);
      if (denominator % lo.d || denominator % hi.d || width.n < BigInt(0) || width.n*denominator > BigInt(6)*width.d ||
        lo.n*context.S > truth.lower*lo.d || hi.n*context.S < truth.upper*hi.d) return null;
    }
    return owned as BlockOrderResult;
  } catch { return null; }
}
