/** Two labeled binary sites. Coupling and a support constraint are distinct
 * causes of dependence. All energies are assembled before binary64 rounding. */
import { binaryProduct as b, binarySum as sum, binaryMultiply as mul, binaryNegate as neg, type BinaryValue } from './learning-binary-products';
import { exactFinitePartition, partitionObservationRange as rangeOf, partitionObservationFromRange as enclose,
  scalePartitionObservation as scaleObservation, differencePartitionObservations as difference, type PartitionObservation } from './learning-partition-exact-core';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
export interface CoupledPartitionParameters {
  couplingEnergy: number;
  fieldEnergy: number;
  temperature: number;
  support: 'full-product' | 'fixed-zero-sum';
}
type Rational = NonNullable<ReturnType<typeof rangeOf>>['lower'];
type Range = NonNullable<ReturnType<typeof rangeOf>>;
const one = b(1), zero = b(0), Z = BigInt(0);
const r = (n: BinaryValue, d = one): Rational => ({ n, d });
const add = (a: Rational, c: Rational): Rational => r(sum([mul(a.n, c.d), mul(c.n, a.d)]), mul(a.d, c.d));
const negative = (a: Rational): Rational => r(neg(a.n), a.d);
const compare = (a: Rational, c: Rational) => sum([mul(a.n, c.d), neg(mul(c.n, a.d))]).coefficient;
const exact = (v: number) => enclose({ lower: r(b(v)), upper: r(b(v)) });
const addRange = (a: Range, c: Range): Range => ({ lower: add(a.lower, c.lower), upper: add(a.upper, c.upper) });
const subtractRange = (a: Range, c: Range): Range => ({ lower: add(a.lower, negative(c.upper)), upper: add(a.upper, negative(c.lower)) });
const sumRanges = (a: Range[]): Range => a.reduce(addRange, { lower: r(zero), upper: r(zero) });
const infinity = (sign: number): PartitionObservation => ({ value: null, sign, logMagnitude: null, logResolution: 'not-applicable', resolution: 'mathematical-infinity', bounds: null });
const positiveProof = (q: PartitionObservation) => q.resolution === 'unresolved-at-precision-ceiling' ? { ...q, sign: 1 } : q;
function logIntegerRatio(n: number, d: number, P: number) {
  if (n === d) return exact(0);
  const interval = createTranscendentalContext(P).logRatio(b(n), b(d));
  return enclose({ lower: r({ coefficient: interval.lower, exponent: -P }), upper: r({ coefficient: interval.upper, exponent: -P }) });
}
/** Directed entropy of a marginal. For a zero lower endpoint, use
 * 0 <= -p ln p <= u(1-ln u), avoiding both ln0 and a false exact zero. */
function entropyRange(p: Range, P: number): Range {
  const lo = compare(p.lower, r(zero)) < Z ? r(zero) : p.lower;
  const hi = compare(p.upper, r(one)) > Z ? r(one) : p.upper;
  if (hi.n.coefficient === Z) return { lower: r(zero), upper: r(zero) };
  const ctx = createTranscendentalContext(P), upperLog = ctx.logRatio(hi.n, hi.d);
  const logU = { lower: r({ coefficient: upperLog.lower, exponent: -P }), upper: r({ coefficient: upperLog.upper, exponent: -P }) };
  if (lo.n.coefficient === Z) {
    const factor = add(r(one), negative(logU.lower));
    return { lower: r(zero), upper: r(mul(hi.n, factor.n), mul(hi.d, factor.d)) };
  }
  const lowerLog = ctx.logRatio(lo.n, lo.d);
  const minusLogL = neg({ coefficient: lowerLog.lower, exponent: -P });
  const minusLogU = neg({ coefficient: upperLog.upper, exponent: -P });
  return { lower: r(mul(lo.n, minusLogU), lo.d), upper: r(mul(hi.n, minusLogL), hi.d) };
}
const configurations = [[-1, -1], [-1, 1], [1, -1], [1, 1]] as const;
export function coupledPartitionModel(parameters: CoupledPartitionParameters) {
  const { couplingEnergy: J, fieldEnergy: h, temperature: T, support } = parameters;
  if (![J, h, T].every(Number.isFinite) || !['full-product', 'fixed-zero-sum'].includes(support)) throw new RangeError('Expected finite coupling, field, temperature and a valid support');
  const allowed = configurations.map((s, index) => ({ spins: s, index, alignment: s[0] * s[1], magnetization: s[0] + s[1] }))
    .filter(s => support === 'full-product' || s.magnetization === 0);
  const energies = allowed.map(s => sum([b(-J, s.alignment), b(-h, s.magnetization)]));
  const features = allowed.map(s => [s.alignment, s.magnetization, ...s.spins]);
  let result: ReturnType<typeof calculate> | undefined;
  const attempts: number[] = [];
  function calculate(P: number) {
    const joint = exactFinitePartition(energies, T, { features, minimumPrecision: P, maximumPrecision: P });
    const independent = exactFinitePartition([b(-h), b(h)], T, { minimumPrecision: P, maximumPrecision: P });
    const moments = joint.features!;
    let logFactorizationDefect: PartitionObservation;
    if (support === 'full-product' && J === 0) logFactorizationDefect = exact(0);
    else if (T === 0) {
      const ground = rangeOf(joint.energyAnchor)!.lower.n;
      const defect = sum([ground, b(2, Math.abs(h))]);
      logFactorizationDefect = defect.coefficient === Z ? logIntegerRatio(joint.anchorMultiplicity, h === 0 ? 4 : 1, P) : infinity(defect.coefficient > Z ? -1 : 1);
    } else logFactorizationDefect = difference(joint.logPartition, independent.logPartition, 2);
    const freeEnergyDefect = support === 'full-product' && J === 0 ? exact(0) : difference(joint.freeEnergy, independent.freeEnergy, 2);
    let mutualInformation: PartitionObservation;
    if (support === 'full-product' && J === 0) mutualInformation = exact(0);
    else if (support === 'fixed-zero-sum') mutualInformation = logIntegerRatio(2, 1, P);
    else if (T === 0 && joint.anchorMultiplicity === 1) mutualInformation = exact(0);
    else {
      const marginals = [0, 1].flatMap(site => [-1, 1].map(spin => sumRanges(joint.states.filter((_, i) => allowed[i].spins[site] === spin).map(s => rangeOf(s.probability)!))));
      const marginalEntropy = sumRanges(marginals.map(p => entropyRange(p, P)));
      const mi = subtractRange(marginalEntropy, rangeOf(joint.entropyInKB)!);
      // KL nonnegativity is a theorem, not a threshold on a displayed number.
      if (mi.lower.n.coefficient < Z) mi.lower = r(zero);
      mutualInformation = enclose(mi);
      if (T !== 0 && J !== 0) mutualInformation = positiveProof(mutualInformation);
    }
    const covariance = moments.covariance;
    const response = (q: PartitionObservation, observable: number, source: number) => {
      if (T !== 0) return { status: 'finite-temperature-derivative' as const, observation: scaleObservation(q, b(T < 0 ? -1 : 1), b(kB, Math.abs(T))) };
      // At a ground crossing, zero covariance need not imply a derivative:
      // J=h=0 has Cov(A,M)=0, but any nonzero field selects A=1.
      const ground = features.filter((_, i) => joint.states[i].probability.sign === 1);
      const minimum = Math.min(...ground.map(f => f[source])), maximum = Math.max(...ground.map(f => f[source]));
      const total = ground.reduce((v, f) => v + f[observable], 0);
      const locallyConstant = [minimum, maximum].every(extreme => {
        const selected = ground.filter(f => f[source] === extreme);
        return selected.reduce((v, f) => v + f[observable], 0) * ground.length === total * selected.length;
      });
      return locallyConstant ? { status: 'locally-constant-zero' as const, observation: exact(0) }
        : { status: 'undefined-at-zero-temperature' as const, observation: null };
    };
    const responses = { alignmentToCoupling: response(covariance[0][0], 0, 0), magnetizationToField: response(covariance[1][1], 1, 1), alignmentToField: response(covariance[0][1], 0, 1) };
    const spinCovariance = covariance[2][3];
    // The odds ratio is exp(4 beta J) on the full product support.
    const signedSpinCovariance = T !== 0 && support === 'full-product' && J !== 0 && spinCovariance.sign === null
      ? { ...spinCovariance, sign: Math.sign(J) * Math.sign(T) } : spinCovariance;
    const quantities = [logFactorizationDefect, freeEnergyDefect, mutualInformation, signedSpinCovariance, ...Object.values(responses).flatMap(v => v.observation ? [v.observation] : [])];
    return { joint, independentSingleSite: independent, logFactorizationDefect, freeEnergyDefect, mutualInformation,
      meanAlignment: moments.means[0], meanMagnetization: moments.means[1], meanSpins: moments.means.slice(2),
      sourceHessian: covariance.slice(0, 2).map(row => row.slice(0, 2)), spinCovariance: signedSpinCovariance, responses,
      fullyResolved: joint.fullyResolved && independent.fullyResolved && quantities.every(q => q.resolution !== 'unresolved-at-precision-ceiling') };
  }
  for (const P of [128, 256, 512, 1024, 2048, 4096, 8192]) {
    attempts.push(P); result = calculate(P);
    if (result.fullyResolved || T === 0) break;
  }
  const model = result!;
  return { scope: 'two-labeled-binary-sites-canonical-fixed-support' as const, parameters: { ...parameters }, ...model, attempts,
    factorizationIdentity: support === 'full-product' && J === 0,
    sourceCoordinates: ['beta-times-coupling', 'beta-times-field'] as const,
    states: configurations.map((spins, index) => {
      const i = allowed.findIndex(s => s.index === index);
      return { spins, index, allowed: i >= 0, energy: i >= 0 ? model.joint.states[i].energy : null,
        probability: i >= 0 ? model.joint.states[i].probability : exact(0),
        logProbability: i >= 0 ? model.joint.states[i].logProbability : infinity(-1) };
    }),
    contract: 'Dimensionless source Hessian = covariance of alignment and total magnetization. Physical fixed-temperature responses multiply by signed beta. T=0 is the positive-side ground distribution, not a finite beta derivative. Mutual information uses both marginals and is distinct from the signed log partition factorization defect. Forbidden configurations have exact zero probability; unresolved finite probabilities are not zero.' as const };
}
