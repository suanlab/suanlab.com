/** Two named spins, explicit support, exact binary Hamiltonian and directed
 * finite Gibbs kernels. These are equilibrium-preserving update rules, not
 * physical transition rates. No random trajectory is generated here.
 */
import { binaryProduct as product, binarySum as sum, binaryNegate as negate, type BinaryValue } from './learning-binary-products';
import { createTranscendentalContext, type FixedInterval } from './learning-interval-transcendentals';
import { partitionObservationFromRange, type PartitionObservation } from './learning-partition-exact-core';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
export type GibbsKernelName = 'first' | 'second' | 'random' | 'systematic' | 'blocked';
export interface FiniteGibbsParameters {
  couplingEnergy: number;
  firstFieldEnergy: number;
  secondFieldEnergy: number;
  energyOffset: number;
  temperature: number;
  support: 'full-product' | 'fixed-zero-sum';
  initialStateIndex?: number;
  observationSteps?: readonly number[];
}
export interface FiniteGibbsOptions { minimumPrecision?: number; maximumPrecision?: number }
type Matrix = FixedInterval[][];
const Z = BigInt(0), O = BigInt(1), TWO = BigInt(2);
const names: GibbsKernelName[] = ['first', 'second', 'random', 'systematic', 'blocked'];
const precisions = [128, 256, 512, 1024, 2048, 4096, 8192];
const dyadic = (coefficient: bigint, exponent = 0): BinaryValue => ({ coefficient, exponent });
const floor = (n: bigint, d: bigint) => n >= Z ? n / d : -((-n + d - O) / d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
const point = (n: bigint): FixedInterval => ({ lower: n, upper: n });
const add = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower + b.lower, upper: a.upper + b.upper });
const subtract = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower - b.upper, upper: a.upper - b.lower });
const reduce = (xs: FixedInterval[]) => xs.reduce(add, point(Z));
function components(graph: boolean[][]) {
  const reach = graph.map(row => [...row]), n = graph.length;
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) reach[i][j] ||= reach[i][k] && reach[k][j];
  const seen = new Set<number>(), result: number[][] = [];
  for (let i = 0; i < n; i++) if (!seen.has(i)) { const c = Array.from({ length: n }, (_, j) => j).filter(j => reach[i][j] && reach[j][i]); c.forEach(j => seen.add(j)); result.push(c); }
  return result;
}
export function finiteGibbsModel(input: FiniteGibbsParameters, options: FiniteGibbsOptions = {}) {
  if (!input || ![input.couplingEnergy, input.firstFieldEnergy, input.secondFieldEnergy, input.energyOffset, input.temperature].every(Number.isFinite)
    || input.temperature === 0 || !['full-product', 'fixed-zero-sum'].includes(input.support))
    throw new RangeError('Expected finite Hamiltonian parameters, nonzero finite temperature and explicit supported state space; zero-temperature conditionals require a separate limiting construction');
  const states = ([[-1, -1], [-1, 1], [1, -1], [1, 1]] as [number, number][]).filter(s => input.support === 'full-product' || s[0] + s[1] === 0);
  if (input.observationSteps !== undefined && !Array.isArray(input.observationSteps)) throw new RangeError('Expected an observation step array');
  const n = states.length, initial = input.initialStateIndex ?? 0, observationSteps = input.observationSteps === undefined ? [1, 2, 4] : Array.from(input.observationSteps);
  if (!Number.isInteger(initial) || initial < 0 || initial >= n || !Array.isArray(input.observationSteps ?? observationSteps)
    || observationSteps.length < 1 || observationSteps.length > 8 || !observationSteps.every((v, i) => Number.isInteger(v) && v >= 1 && v <= 64 && (i === 0 || v > observationSteps[i - 1])))
    throw new RangeError('Expected an allowed initial state and 1..8 increasing observation steps in 1..64');
  const minimum = options.minimumPrecision ?? 128, maximum = options.maximumPrecision ?? 8192;
  if (!precisions.includes(minimum) || !precisions.includes(maximum) || minimum > maximum) throw new RangeError('Invalid precision range');
  const energies = states.map(([a, b]) => sum([product(input.energyOffset), product(-1, input.couplingEnergy, a, b), product(-1, input.firstFieldEnergy, a), product(-1, input.secondFieldEnergy, b)]));
  const active = energies.filter(e => e.coefficient !== Z), unit = active.length ? Math.min(...active.map(e => e.exponent)) : 0;
  const integerEnergies = energies.map(e => e.coefficient === Z ? Z : e.coefficient << BigInt(e.exponent - unit));
  const direction = input.temperature < 0 ? -1 : 1, thermal = product(kB, Math.abs(input.temperature));
  const graph1 = states.map(s => states.map(t => t[1] === s[1])), graph2 = states.map(s => states.map(t => t[0] === s[0]));
  const graphs: Record<GibbsKernelName, boolean[][]> = { first: graph1, second: graph2,
    random: graph1.map((row, i) => row.map((v, j) => v || graph2[i][j])),
    systematic: graph1.map(row => states.map((_, j) => row.some((v, k) => v && graph2[k][j]))),
    blocked: states.map(() => states.map(() => true)) };
  const independent = input.couplingEnergy === 0 && input.support === 'full-product';
  const witnessSign = input.support === 'fixed-zero-sum' || input.couplingEnergy === 0 ? 0 : Math.sign(input.couplingEnergy) * direction;
  const diagnostics = Object.fromEntries(names.map(name => {
    const classes = components(graphs[name]), irreducible = classes.length === 1;
    return [name, { rowStochastic: true, targetStationary: true, proof: 'conditional-normalization-and-kernel-composition' as const,
      stronglyConnectedComponents: classes, irreducible, positiveSelfLoops: true, aperiodicWithinEachClass: true,
      uniqueStationaryDistribution: irreducible, convergesToTargetFromEveryInitialState: irreducible,
      detailedBalance: name !== 'systematic' || witnessSign === 0,
      detailedBalanceProof: name !== 'systematic' ? 'heat-bath-reversibility-or-convex-mixture' : witnessSign === 0 ? 'independent-product-or-identity' : 'witness-sign-is-sign-beta-J',
      coordinatesResampledPerStep: name === 'systematic' || name === 'blocked' ? 2 : 1 }];
  })) as Record<GibbsKernelName, { rowStochastic: boolean; targetStationary: boolean; proof: string; stronglyConnectedComponents: number[][];
    irreducible: boolean; positiveSelfLoops: boolean; aperiodicWithinEachClass: boolean; uniqueStationaryDistribution: boolean;
    convergesToTargetFromEveryInitialState: boolean; detailedBalance: boolean; detailedBalanceProof: string; coordinatesResampledPerStep: number }>;
  function calculate(P: number) {
    const ctx = createTranscendentalContext(P), S = ctx.scale;
    const positiveMultiply = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: floor(a.lower * b.lower, S), upper: ceil(a.upper * b.upper, S) });
    const half = (v: FixedInterval): FixedInterval => ({ lower: floor(v.lower, TWO), upper: ceil(v.upper, TWO) });
    const observe = (v: FixedInterval, knownSign?: number): PartitionObservation => {
      const q = partitionObservationFromRange({ lower: { n: dyadic(v.lower, -P), d: product(1) }, upper: { n: dyadic(v.upper, -P), d: product(1) } });
      if (knownSign !== undefined && q.sign === null) q.sign = knownSign;
      return q;
    };
    const observeMass = (v: FixedInterval) => observe(v, v.upper > Z ? 1 : 0);
    const expCache = new Map<string, FixedInterval>();
    function normalize(indices: number[]) {
      const anchor = indices.map(i => integerEnergies[i]).reduce((a, b) => (direction > 0 ? a <= b : a >= b) ? a : b);
      const costs = indices.map(i => direction > 0 ? integerEnergies[i] - anchor : anchor - integerEnergies[i]);
      const raw = costs.map(cost => { const key = cost.toString(); let q = expCache.get(key); if (!q) { q = ctx.expNegativeRatio(dyadic(cost, unit), thermal); expCache.set(key, q); } return q; });
      const W = reduce(raw);
      const probabilities = raw.map(q => ({ lower: floor(q.lower * S, W.upper), upper: (v => v > S ? S : v)(ceil(q.upper * S, W.lower)) }));
      return { probabilities, raw, W, costs };
    }
    const targetPreparation = normalize(states.map((_, i) => i)), target = targetPreparation.probabilities;
    const logCache = new Map<string, FixedInterval>();
    const logarithm = (v: bigint) => { const key = v.toString(); let q = logCache.get(key); if (!q) { q = ctx.logRatio(dyadic(v, -P), product(1)); logCache.set(key, q); } return q; };
    const logWLower = logarithm(targetPreparation.W.lower), logWUpper = logarithm(targetPreparation.W.upper);
    const logTarget = target.map((p, i) => {
      if (p.lower > Z) return observe({ lower: logarithm(p.lower).lower, upper: logarithm(p.upper).upper }, -1);
      // A probability below the fixed-point quantum still has a finite exact
      // energy exponent. Keep it rational instead of projecting it to -Infinity.
      const cost = dyadic(-targetPreparation.costs[i], unit);
      const lo = sum([cost, negate({ coefficient: logWUpper.upper * thermal.coefficient, exponent: thermal.exponent - P })]);
      const hi = sum([cost, negate({ coefficient: logWLower.lower * thermal.coefficient, exponent: thermal.exponent - P })]);
      const q = partitionObservationFromRange({ lower: { n: lo, d: thermal }, upper: { n: hi, d: thermal } });
      if (q.sign === null) q.sign = -1; return q;
    });
    const coordinateKernel = (site: number): Matrix => {
      const cache = new Map<number, Map<number, FixedInterval>>();
      return states.map(s => {
        const other = s[1 - site]; let conditional = cache.get(other);
        if (!conditional) {
          const indices = states.map((t, j) => t[1 - site] === other ? j : -1).filter(j => j >= 0);
          const prepared = normalize(indices);
          conditional = new Map(indices.map((j, k) => [j, prepared.probabilities[k]])); cache.set(other, conditional);
        }
        return states.map((_, j) => conditional!.get(j) ?? point(Z));
      });
    };
    const matrixMultiply = (A: Matrix, B: Matrix): Matrix => A.map(row => states.map((_, j) => reduce(row.map((a, k) => positiveMultiply(a, B[k][j])))));
    const evolve = (q: FixedInterval[], K: Matrix) => states.map((_, j) => reduce(q.map((a, i) => positiveMultiply(a, K[i][j]))));
    const K1 = coordinateKernel(0), K2 = coordinateKernel(1);
    const kernels: Record<GibbsKernelName, Matrix> = { first: K1, second: K2,
      random: K1.map((row, i) => row.map((q, j) => half(add(q, K2[i][j])))),
      systematic: independent ? target.map(() => target) : matrixMultiply(K1, K2), blocked: target.map(() => target) };
    const flows = Object.fromEntries(names.map(name => [name, kernels[name].map((row, i) => row.map(q => positiveMultiply(target[i], q)))])) as Record<GibbsKernelName, Matrix>;
    const distributions = {} as Record<GibbsKernelName, Matrix>, distances = {} as Record<GibbsKernelName, FixedInterval[]>;
    const absolute = (v: FixedInterval): FixedInterval => v.lower >= Z ? v : v.upper <= Z ? { lower: -v.upper, upper: -v.lower }
      : { lower: Z, upper: -v.lower > v.upper ? -v.lower : v.upper };
    for (const name of names) {
      let q = states.map((_, i) => point(i === initial ? S : Z)); const rows: FixedInterval[][] = [], tv: FixedInterval[] = [];
      const oneStepExactTarget = name === 'blocked' || name === 'systematic' && independent;
      for (let step = 1; step <= observationSteps[observationSteps.length - 1]; step++) {
        q = oneStepExactTarget ? target : evolve(q, kernels[name]);
        if (observationSteps.includes(step)) {
          rows.push(q); tv.push(oneStepExactTarget ? point(Z) : half(reduce(q.map((p, i) => absolute(subtract(p, target[i]))))));
        }
      }
      distributions[name] = rows; distances[name] = tv;
    }
    // For two binary variables, the last-coordinate marginal chain has one
    // nontrivial eigenvalue: the product of the two conditional contrasts.
    // Therefore q_t-pi=lambda^(t-1)(q_1-pi). Multiply dyadic intervals BEFORE
    // projecting; tiny TV distances must not be erased by fixed-point rounding.
    const signedPositive = (v: FixedInterval): FixedInterval => {
      const a = witnessSign < 0 ? { lower: -v.upper, upper: -v.lower } : v;
      return { lower: a.lower < Z ? Z : a.lower, upper: a.upper };
    };
    const contrastA = input.support === 'fixed-zero-sum' ? point(S) : independent ? point(Z)
      : signedPositive(subtract(K1[1][3], K1[0][2]));
    const contrastB = input.support === 'fixed-zero-sum' ? point(S) : independent ? point(Z)
      : signedPositive(subtract(K2[2][3], K2[0][1]));
    const rawMultiply = (a: BinaryValue, b: BinaryValue): BinaryValue => {
      const coefficient = a.coefficient * b.coefficient;
      if (coefficient === Z) return dyadic(Z);
      const twos = (coefficient & -coefficient).toString(2).length - 1;
      return { coefficient: coefficient >> BigInt(twos), exponent: a.exponent + b.exponent + twos };
    };
    const rawObserve = (lower: BinaryValue, upper: BinaryValue, knownSign: number): PartitionObservation => {
      const q = partitionObservationFromRange({ lower: { n: lower, d: product(1) }, upper: { n: upper, d: product(1) } });
      if (q.sign === null) q.sign = knownSign; return q;
    };
    const contractionLower = rawMultiply(dyadic(contrastA.lower, -P), dyadic(contrastB.lower, -P));
    const contractionUpper = rawMultiply(dyadic(contrastA.upper, -P), dyadic(contrastB.upper, -P));
    const contraction = rawObserve(contractionLower, contractionUpper, independent ? 0 : 1);
    const initialDistance = independent ? point(Z) : half(reduce(kernels.systematic[initial].map((p, i) => absolute(subtract(p, target[i])))));
    const systematicDistances = observationSteps.map(step => {
      let lower = dyadic(initialDistance.lower, -P), upper = dyadic(initialDistance.upper, -P);
      for (let j = 1; j < step; j++) { lower = rawMultiply(lower, contractionLower); upper = rawMultiply(upper, contractionUpper); }
      return rawObserve(lower, upper, independent ? 0 : 1);
    });
    const systematicWitness = witnessSign === 0 ? point(Z) : subtract(flows.systematic[0][1], flows.systematic[1][0]);
    const mapMatrix = (matrix: Matrix) => matrix.map(row => row.map(observeMass));
    const targetObservations = target.map(observeMass);
    targetObservations.forEach((p, i) => { p.logMagnitude = logTarget[i].value; p.logResolution = logTarget[i].resolution === 'unresolved-at-precision-ceiling' ? 'unresolved' : logTarget[i].value === null ? 'outside-number-range' : 'finite'; });
    const kernelObservations = Object.fromEntries(names.map(name => [name, mapMatrix(kernels[name])])) as Record<GibbsKernelName, PartitionObservation[][]>;
    const flowObservations = Object.fromEntries(names.map(name => [name, mapMatrix(flows[name])])) as Record<GibbsKernelName, PartitionObservation[][]>;
    const distributionObservations = Object.fromEntries(names.map(name => [name, mapMatrix(distributions[name])])) as Record<GibbsKernelName, PartitionObservation[][]>;
    const totalVariation = Object.fromEntries(names.map(name => [name, distances[name].map(q => observe(q, name === 'blocked' || name === 'systematic' && independent ? 0 : 1))])) as Record<GibbsKernelName, PartitionObservation[]>;
    totalVariation.systematic = systematicDistances;
    const witness = observe(systematicWitness, witnessSign);
    const all = [...targetObservations, ...logTarget, witness, contraction,
      ...names.flatMap(name => [...kernelObservations[name].flat(), ...flowObservations[name].flat(), ...distributionObservations[name].flat(), ...totalVariation[name]])];
    const numericalChecks = Object.fromEntries(names.map(name => {
      const rowSums = kernels[name].map(reduce), stationarityResidual = evolve(target, kernels[name]).map((q, i) => subtract(q, target[i]));
      return [name, { rowSumsContainOne: rowSums.every(q => q.lower <= S && q.upper >= S),
        stationarityResidualsContainZero: stationarityResidual.every(q => q.lower <= Z && q.upper >= Z),
        rowSums: rowSums.map(q => observe(q)), stationarityResiduals: stationarityResidual.map(q => observe(q)) }];
    })) as Record<GibbsKernelName, { rowSumsContainOne: boolean; stationarityResidualsContainZero: boolean; rowSums: PartitionObservation[]; stationarityResiduals: PartitionObservation[] }>;
    return { target: targetObservations, logTarget, kernels: kernelObservations, flows: flowObservations,
      distributions: distributionObservations, totalVariationToTarget: totalVariation, systematicContraction: contraction,
      systematicDistanceIdentity: 'TV(q_t,pi)=lambda^(t-1) TV(q_1,pi); lambda is the product of conditional contrasts, or 1 on the fixed-zero-sum identity chain', numericalChecks,
      systematicDetailedBalanceWitness: { sourceIndex: 0, destinationIndex: 1, forwardFlow: flowObservations.systematic[0][1], reverseFlow: flowObservations.systematic[1][0],
        difference: witness, provenSign: witnessSign, proof: input.support === 'fixed-zero-sum' ? 'single-coordinate kernels are identity on this support' : 'pi0*pi1*(pi0*pi3-pi1*pi2)/((pi0+pi1)*(pi0+pi2)*(pi1+pi3)); cross-product odds=exp(4 beta J)' },
      precisionBits: P, fullyResolved: all.every(q => q.resolution !== 'unresolved-at-precision-ceiling') };
  }
  const attempts: number[] = []; let latest: ReturnType<typeof calculate> | undefined;
  for (const P of precisions.filter(p => p >= minimum && p <= maximum)) { attempts.push(P); latest = calculate(P); if (latest.fullyResolved) break; }
  const exactObservation = (v: BinaryValue) => partitionObservationFromRange({ lower: { n: v, d: product(1) }, upper: { n: v, d: product(1) } });
  return { scope: 'finite-two-spin-conditional-gibbs-kernels' as const,
    parameters: { couplingEnergy: input.couplingEnergy, firstFieldEnergy: input.firstFieldEnergy, secondFieldEnergy: input.secondFieldEnergy,
      energyOffset: input.energyOffset, temperature: input.temperature, support: input.support, initialStateIndex: initial, observationSteps: [...observationSteps] },
    states: states.map((spins, i) => ({ index: i, spins, energy: exactObservation(energies[i]) })),
    initialStateIndex: initial, observationSteps, boltzmannConstant: kB, diagnostics, ...latest!, attempts,
    resolutionScope: 'Target, log target, kernels, flows, evolved laws, total variation, contraction and detailed-balance witness. Numerical residuals are diagnostic enclosures and do not participate in fullyResolved.',
    approximationContract: 'Exact binary Hamiltonian and locally anchored directed exponential/logarithm enclosures, 128..8192 bits. Resolved nonzero observations have relative enclosure width <=2^-48; unresolved is not zero. Kernel graphs and structural reversibility proofs do not depend on displayed floating zeros. Numerical residuals containing zero are not proofs of stationarity. Systematic steps update both coordinates; random steps update one. No PRNG path or physical clock is defined. Temperature zero is explicitly unsupported by this finite nonzero-temperature conditional model.' };
}
