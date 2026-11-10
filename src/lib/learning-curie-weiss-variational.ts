/** Directed stationary roots of the constrained Curie–Weiss free energy.
 * Infinite fully connected model; finite-N mixtures and physical dynamics are
 * separate preparations. No epsilon merges roots or declares criticality.
 */
import { binaryProduct } from './learning-binary-products';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { rationalGreatestCommonDivisor as gcd } from './learning-rational-gcd';
import { partitionObservationFromRange, type PartitionObservation } from './learning-partition-exact-core';
export interface CurieWeissVariationalParameters { couplingK: number; fieldB: number }
export interface CurieWeissVariationalOptions { minimumPrecision?: number; maximumPrecision?: number }
interface Q { n: bigint; d: bigint }
interface I { lower: Q; upper: Q }
const Z = BigInt(0), O = BigInt(1), T = BigInt(2);
const abs = (n: bigint) => n < Z ? -n : n;
function q(n: bigint, d = O): Q { if (d <= Z) throw new RangeError('Expected positive rational denominator'); const g = gcd(abs(n), d); return { n: n / g, d: d / g }; }
const integer = (n: number) => q(BigInt(n));
const add = (a: Q, b: Q) => q(a.n * b.d + b.n * a.d, a.d * b.d);
const neg = (a: Q): Q => ({ n: -a.n, d: a.d });
const sub = (a: Q, b: Q) => add(a, neg(b));
const mul = (a: Q, b: Q) => q(a.n * b.n, a.d * b.d);
const div = (a: Q, b: Q) => { if (!b.n) throw new RangeError('Zero denominator'); return q(a.n * b.d * (b.n < Z ? -O : O), a.d * abs(b.n)); };
const cmp = (a: Q, b: Q) => a.n * b.d - b.n * a.d;
const min = (a: Q, b: Q) => cmp(a, b) < Z ? a : b;
const max = (a: Q, b: Q) => cmp(a, b) > Z ? a : b;
const absolute = (a: Q): Q => ({ n: abs(a.n), d: a.d });
const square = (a: Q) => mul(a, a);
const point = (a: Q): I => ({ lower: a, upper: a });
const sum = (a: I, b: I): I => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const negative = (a: I): I => ({ lower: neg(a.upper), upper: neg(a.lower) });
const difference = (a: I, b: I) => sum(a, negative(b));
function times(a: I, b: I): I { const xs = [mul(a.lower, b.lower), mul(a.lower, b.upper), mul(a.upper, b.lower), mul(a.upper, b.upper)]; return { lower: xs.reduce(min), upper: xs.reduce(max) }; }
function inverse(a: I): I {
  if (a.lower.n <= Z && a.upper.n >= Z) throw new PrecisionLimited('curvature interval contains zero');
  return { lower: div(integer(1), a.upper), upper: div(integer(1), a.lower) };
}
const binary = (x: number): Q => { const a = binaryProduct(x); return a.exponent >= 0 ? q(a.coefficient << BigInt(a.exponent)) : q(a.coefficient, O << BigInt(-a.exponent)); };
const bv = (coefficient: bigint) => ({ coefficient, exponent: 0 });
const observe = (a: I): PartitionObservation => partitionObservationFromRange({ lower: { n: bv(a.lower.n), d: bv(a.lower.d) }, upper: { n: bv(a.upper.n), d: bv(a.upper.d) } });
const serialize = (a: Q) => ({ numerator: { coefficient: a.n.toString(), exponent: 0 }, denominator: { coefficient: a.d.toString(), exponent: 0 } });
const bounds = (a: I) => ({ lower: serialize(a.lower), upper: serialize(a.upper) });
const resolved = (a: PartitionObservation) => a.resolution !== 'unresolved-at-precision-ceiling';
const positiveInfinity = (): PartitionObservation => ({ value: null, sign: 1, logMagnitude: null, logResolution: 'not-applicable', resolution: 'mathematical-infinity', bounds: null });
class PrecisionLimited extends Error {}
function isqrt(n: bigint): bigint {
  if (n < T) return n;
  let x = O << BigInt(Math.ceil(n.toString(2).length / 2));
  for (;;) { const y = (x + n / x) / T; if (y >= x) return x; x = y; }
}
function sqrtRange(a: Q, P: number): I {
  const sn = isqrt(a.n), sd = isqrt(a.d);
  if (sn * sn === a.n && sd * sd === a.d) return point(q(sn, sd));
  const S = O << BigInt(P), n = isqrt(a.n * S * S / a.d);
  return { lower: q(n, S), upper: q(n + O, S) };
}
const precisions = [128, 256, 512, 1024, 2048, 4096, 8192];
type Role = 'global-minimum' | 'metastable-minimum' | 'unstable-maximum' | 'critical-minimum';
export function curieWeissVariationalModel(input: CurieWeissVariationalParameters, options: CurieWeissVariationalOptions = {}) {
  if (!input || !Number.isFinite(input.couplingK) || input.couplingK < 0 || input.couplingK > 8 || !Number.isFinite(input.fieldB) || Math.abs(input.fieldB) > 4)
    throw new RangeError('Expected dimensionless ferromagnetic K=0..8 and b=-4..4');
  const minimum = options.minimumPrecision ?? 128, maximum = options.maximumPrecision ?? 8192;
  if (!precisions.includes(minimum) || !precisions.includes(maximum) || minimum > maximum) throw new RangeError('Invalid precision range');
  const parameters = Object.freeze({ couplingK: input.couplingK, fieldB: input.fieldB }), K = binary(input.couplingK), b = binary(input.fieldB);
  const one = integer(1), half = q(O, T), r = q((O << BigInt(64)) - O, O << BigInt(64));
  // atanh(r)>32 ln2>16>K+|b|, so all stationary roots lie in (-r,r).
  const attempts: { precisionBits: number; reason: string | null }[] = [];
  function calculate(P: number) {
    const ctx = createTranscendentalContext(P), S = O << BigInt(P);
    const log = (a: Q): I => { const v = ctx.logRatio(bv(a.n), bv(a.d)); return { lower: q(v.lower, S), upper: q(v.upper, S) }; };
    const logTwo = log(integer(2));
    // A(t)=sum t^j/(2j+3); B(t)=sum t^j/[(2j+4)(2j+3)].
    // Factoring m^3 or m^4 preserves tiny critical signals without subtracting
    // rounded atanh(m)-m or entropy-ln2. Bounds are relative to the factor.
    function series(t: Q, kind: 'A' | 'B'): I {
      let power = one, total = integer(0);
      for (let j = 0; j <= 2 * P; j++) {
        const den = kind === 'A' ? BigInt(2 * j + 3) : BigInt((2 * j + 4) * (2 * j + 3));
        total = add(total, div(power, q(den))); power = mul(power, t);
        const nextDen = kind === 'A' ? BigInt(2 * j + 5) : BigInt((2 * j + 6) * (2 * j + 5));
        const tail = div(power, mul(q(nextDen), sub(one, t)));
        if (cmp(mul(tail, q(S)), one) <= Z) return { lower: total, upper: add(total, tail) };
      }
      throw new PrecisionLimited('series resource ceiling');
    }
    const small = (m: Q) => cmp(absolute(m), q(O, BigInt(8))) <= Z;
    const atanh = (m: Q): I => times(log(div(add(one, m), sub(one, m))), point(half));
    function f(m: Q): I {
      const linear = sub(mul(sub(one, K), m), b);
      return small(m) ? sum(point(linear), times(point(mul(square(m), m)), series(square(m), 'A'))) : difference(atanh(m), point(add(mul(K, m), b)));
    }
    function entropy(m: Q): I {
      const a = mul(add(one, m), half), c = mul(sub(one, m), half);
      return negative(sum(times(point(a), log(a)), times(point(c), log(c))));
    }
    function delta(m: Q): I {
      const m2 = square(m), linear = sub(mul(mul(sub(one, K), half), m2), mul(b, m));
      return small(m) ? sum(point(linear), times(point(square(m2)), series(m2, 'B')))
        : sum(difference(point(neg(add(mul(mul(K, half), m2), mul(b, m)))), entropy(m)), logTwo);
    }
    function curvature(a: Q, c: Q): I {
      const lowerSquare = a.n <= Z && c.n >= Z ? integer(0) : min(square(a), square(c));
      const upperSquare = max(square(a), square(c));
      return { lower: sub(div(one, sub(one, lowerSquare)), K), upper: sub(div(one, sub(one, upperSquare)), K) };
    }
    function info(a: Q, c: Q, role: Role, iterations: number) {
      const midpoint = mul(add(a, c), half), fa = f(a), fc = f(c), curve = curvature(a, c);
      const maxSlope = [fa.lower, fa.upper, fc.lower, fc.upper].map(absolute).reduce(max);
      const error = mul(mul(sub(c, a), half), maxSlope), center = delta(midpoint);
      const deltaRange = { lower: sub(center.lower, error), upper: add(center.upper, error) };
      const minimumAbs = a.n <= Z && c.n >= Z ? integer(0) : min(absolute(a), absolute(c)), maximumAbs = max(absolute(a), absolute(c));
      const lowEntropy = entropy(maximumAbs), highEntropy = entropy(minimumAbs);
      const critical = role === 'critical-minimum';
      const derivative = critical ? positiveInfinity() : observe(inverse(curve));
      const publicRoot = { magnetization: observe({ lower: a, upper: c }), curvature: observe(curve), branchFieldDerivative: derivative,
        constrainedFreeEnergy: observe(difference(deltaRange, logTwo)), freeEnergyDifferenceToZero: observe(deltaRange),
        entropyNats: observe({ lower: lowEntropy.lower, upper: highEntropy.upper }), role,
        globalMinimum: role === 'global-minimum' || critical, stationaryMultiplicity: critical ? 3 : 1,
        residualEnclosure: bounds({ lower: min(fa.lower, fc.lower), upper: max(fa.upper, fc.upper) }), bisectionIterations: iterations };
      const ready = [publicRoot.magnetization, publicRoot.curvature, derivative, publicRoot.constrainedFreeEnergy, publicRoot.freeEnergyDifferenceToZero, publicRoot.entropyNats].every(resolved);
      return { publicRoot, deltaRange, interval: { lower: a, upper: c }, ready };
    }
    function bisect(a: Q, c: Q, increasing: boolean, role: Role) {
      const fa = f(a), fc = f(c);
      if (!(increasing ? fa.upper.n < Z && fc.lower.n > Z : fa.lower.n > Z && fc.upper.n < Z)) throw new PrecisionLimited('turning-point endpoint sign unresolved');
      for (let j = 0; j < 4096; j++) {
        const near = min(absolute(a), absolute(c));
        if ((a.n > Z || c.n < Z) && cmp(mul(sub(c, a), q(O << BigInt(112))), near) <= Z) {
          try { const result = info(a, c, role, j); if (result.ready) return result; } catch (e) { if (!(e instanceof PrecisionLimited)) throw e; }
        }
        const mid = mul(add(a, c), half), fm = f(mid);
        const sign = fm.lower.n > Z ? 1 : fm.upper.n < Z ? -1 : 0;
        if (!sign) throw new PrecisionLimited('stationary residual sign unresolved');
        if ((sign < 0) === increasing) a = mid; else c = mid;
      }
      throw new PrecisionLimited('root bisection resource ceiling');
    }
    let classifiedRootCount: number | null = null;
    let spinodal: { turningMagnetization: PartitionObservation; fieldMagnitude: PartitionObservation } | null = null;
    try {
      let roots: ReturnType<typeof info>[];
      if (cmp(K, one) <= Z) {
        classifiedRootCount = 1;
        roots = b.n === Z ? [info(integer(0), integer(0), cmp(K, one) === Z ? 'critical-minimum' : 'global-minimum', 0)] : [bisect(neg(r), r, true, 'global-minimum')];
      } else {
        const t = div(sub(K, one), K), s = sqrtRange(t, P);
        const threshold = cmp(s.upper, q(O, BigInt(8))) <= Z
          ? times(times(point(t), s), difference(point(K), series(t, 'A')))
          : difference(times(point(K), s), { lower: atanh(s.lower).lower, upper: atanh(s.upper).upper });
        spinodal = { turningMagnetization: observe(s), fieldMagnitude: observe(threshold) };
        const field = absolute(b), inside = cmp(field, threshold.lower) < Z, outside = cmp(field, threshold.upper) > Z;
        if (!inside && !outside) throw new PrecisionLimited('spinodal comparison unresolved; possible coalescence');
        classifiedRootCount = inside ? 3 : 1;
        if (!b.n) {
          const positive = bisect(s.upper, r, true, 'global-minimum');
          roots = [info(neg(positive.interval.upper), neg(positive.interval.lower), 'global-minimum', positive.publicRoot.bisectionIterations), info(integer(0), integer(0), 'unstable-maximum', 0), positive];
        } else if (inside) {
          roots = [bisect(neg(r), neg(s.upper), true, b.n < Z ? 'global-minimum' : 'metastable-minimum'),
            bisect(neg(s.lower), s.lower, false, 'unstable-maximum'), bisect(s.upper, r, true, b.n > Z ? 'global-minimum' : 'metastable-minimum')];
        } else roots = [b.n > Z ? bisect(s.upper, r, true, 'global-minimum') : bisect(neg(r), neg(s.upper), true, 'global-minimum')];
      }
      const globalMinimumIndices = roots.map((v, i) => v.publicRoot.globalMinimum ? i : -1).filter(i => i >= 0);
      const global = roots[globalMinimumIndices[0]], pressure = observe(negative(difference(global.deltaRange, logTwo)));
      const barriers = roots.length === 3 ? [0, 2].map(i => ({ minimumIndex: i, barrier: observe(difference(roots[1].deltaRange, roots[i].deltaRange)) })) : [];
      let minimumFreeEnergyGap: PartitionObservation | null = null;
      if (roots.length === 3) {
        if (!b.n) minimumFreeEnergyGap = observe(point(integer(0)));
        else {
          const metastable = roots[b.n > Z ? 0 : 2], direct = difference(metastable.deltaRange, global.deltaRange);
          // Even part plus field bias: 2|b|*|m_meta| <= psi_meta-psi_global
          // <= 2|b|*|m_global|. These bounds retain subnormal field splitting.
          const theorem = { lower: mul(mul(integer(2), absolute(b)), min(absolute(metastable.interval.lower), absolute(metastable.interval.upper))),
            upper: mul(mul(integer(2), absolute(b)), max(absolute(global.interval.lower), absolute(global.interval.upper))) };
          minimumFreeEnergyGap = observe({ lower: max(direct.lower, theorem.lower), upper: min(direct.upper, theorem.upper) });
        }
      }
      if (!roots.every(v => v.ready) || !resolved(pressure) || barriers.some(v => !resolved(v.barrier)) || minimumFreeEnergyGap && !resolved(minimumFreeEnergyGap) || spinodal && (!resolved(spinodal.turningMagnetization) || !resolved(spinodal.fieldMagnitude))) throw new PrecisionLimited('derived free-energy or stability enclosure unresolved');
      return { fullyResolved: true as const, precisionBits: P, roots: roots.map(v => v.publicRoot), rootCount: classifiedRootCount, rootCountCandidates: [classifiedRootCount],
        spinodal, globalMinimumIndices, thermodynamicLimitPressure: pressure, barriers, minimumFreeEnergyGap, unresolvedReason: null };
    } catch (e) {
      if (!(e instanceof PrecisionLimited)) throw e;
      return { fullyResolved: false as const, precisionBits: P, roots: [], rootCount: classifiedRootCount, rootCountCandidates: classifiedRootCount === null ? [1, 2, 3] : [classifiedRootCount],
        spinodal, globalMinimumIndices: [], thermodynamicLimitPressure: null, barriers: [], minimumFreeEnergyGap: null, unresolvedReason: e.message };
    }
  }
  let latest: ReturnType<typeof calculate> | undefined;
  for (const P of precisions.filter(v => v >= minimum && v <= maximum)) { latest = calculate(P); attempts.push({ precisionBits: P, reason: latest.unresolvedReason }); if (latest.fullyResolved) break; }
  return { scope: 'curie-weiss-constrained-variational-limit' as const, parameters, ...latest!, attempts,
    approximationContract: 'Constrained psi=-K*m^2/2-b*m-s(m), not the off-shell log-cosh surrogate. Exact binary64 K=0..8,b=-4..4, directed logarithms and relative series tails. Root existence/uniqueness follows monotonic branches; no epsilon merges roots. Observations resolve only at relative enclosure width <=2^-48. Global selection uses field-sign/symmetry theorems as well as free-energy enclosures. Branch derivatives on unstable roots are not equilibrium susceptibilities. Residual ranges are diagnostics, excluded from fullyResolved. Critical K=1,b=0 has a quartic minimum and infinite limiting derivative. Finite-N mixtures, physical relaxation times and nearest-neighbor lattice accuracy are separate.' };
}
