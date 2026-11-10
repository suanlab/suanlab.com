/** Stationary autonomous three-state jump entropy, measured in nats.
 * Exact traffic/path ratios plus directed logs and tilted moment enclosures.
 * Worker transport is prepared; this is not yet a registered reader or heat model. */
import { fraction as f, zero, one, add, sub, mul, div, sum, neg, equal, exact, point, logarithm, plus, minus, negative, scale, square, observe, buildEntropyTilt, entropyMoment, type F, type I } from './learning-entropy-production-math';

import { isThreeStateEntropyParameters, snapshotEntropyParameters, type ThreeStateEntropyParameters } from './learning-entropy-production-input';
export { isThreeStateEntropyParameters };
export type { ThreeStateEntropyParameters };
const Z = BigInt(0);
const positiveRateProduct = (xs: F[]) => xs.reduce(mul, one);
const productSign = (x: F) => x.n === x.d ? 0 : x.n > x.d ? 1 : -1;
const logSum = (xs: I[]) => xs.reduce(plus, point(zero));

export function threeStateEntropyProductionModel(input: ThreeStateEntropyParameters) {
  if (!isThreeStateEntropyParameters(input)) throw new RangeError('Fifteen owned integer fields, positive rates/denominators, magnitudes<=1e9, time0..4s, tilt-1..2,1..64 intervals,0..16 jumps,0..2 initial state, no unused direction bits; zero time permits no jumps');
  const parameters = snapshotEntropyParameters(input);
  const p = parameters, rates = [p.a, p.b, p.c, p.d, p.e, p.f].map(n => f(BigInt(n), BigInt(p.rateDenominator)));
  const [a, b, c, d, e, rateF] = rates, exits = [add(a, rateF), add(d, b), add(c, e)];
  const Q = [[neg(exits[0]), a, rateF], [d, neg(exits[1]), b], [c, e, neg(exits[2])]];
  const weights = [sum([mul(c, d), mul(d, e), mul(b, c)]), sum([mul(a, e), mul(a, c), mul(e, rateF)]), sum([mul(b, rateF), mul(a, b), mul(d, rateF)])];
  const total = sum(weights), pi = weights.map(x => div(x, total));
  const flows = Q.map((row, i) => row.map((x, j) => i === j ? zero : mul(pi[i], x)));
  const current = sub(flows[0][1], flows[1][0]), ratio = div(positiveRateProduct([a, b, c]), positiveRateProduct([d, e, rateF]));
  const affinity = logarithm(ratio), rate = scale(affinity, current), detailedBalance = ratio.n === ratio.d;
  const pairs = [[0, 1], [1, 2], [2, 0]];
  const edges = pairs.map(([i, j]) => {
    const net = sub(flows[i][j], flows[j][i]), fluxRatio = div(flows[i][j], flows[j][i]), increment = logarithm(fluxRatio);
    return { from: i, to: j, forwardFlow: exact(flows[i][j]), reverseFlow: exact(flows[j][i]), current: exact(net), logFluxRatio: observe(increment), entropyRate: observe(scale(increment, net)) };
  });
  const pairedRate = logSum(pairs.map(([i, j]) => scale(logarithm(div(flows[i][j], flows[j][i])), sub(flows[i][j], flows[j][i]))));
  const adjoint = Q.map((row, i) => row.map((x, j) => i === j ? x : div(mul(pi[j], Q[j][i]), pi[i])));
  const states = [p.initialState];
  for (let k = 0; k < p.jumpCount; k++) states.push((states[k] + ((p.pathCode >> k) & 1 ? 2 : 1)) % 3);
  const boundaryRatio = div(pi[states[0]], pi[states[states.length - 1]]);
  let rateRatio = one, pathRatio = one;
  const jumps = states.slice(1).map((to, index) => {
    const from = states[index], qRatio = div(Q[from][to], Q[to][from]), fluxRatio = div(flows[from][to], flows[to][from]);
    rateRatio = mul(rateRatio, qRatio); pathRatio = mul(pathRatio, fluxRatio);
    return { index, from, to, clockwise: to === (from + 1) % 3, rateRatio: exact(qRatio), fluxRatio: exact(fluxRatio), rateOnlyIncrement: observe(logarithm(qRatio)), entropyIncrement: observe(logarithm(fluxRatio)) };
  });
  const combinedRatio = mul(boundaryRatio, rateRatio);
  if (!equal(combinedRatio, pathRatio)) throw new RangeError('Exact path boundary telescope failed');
  const time = f(BigInt(p.timeNumerator), BigInt(p.timeDenominator)), dwell = div(time, f(BigInt(p.jumpCount + 1)));
  const action = mul(dwell, sum(states.map(i => exits[i])));
  const forwardLog = minus(plus(logarithm(pi[states[0]]), logSum(states.slice(1).map((to, index) => logarithm(Q[states[index]][to])))), point(action));
  const reversedStates = [...states].reverse();
  const reversedLog = minus(plus(logarithm(pi[reversedStates[0]]), logSum(reversedStates.slice(1).map((to, index) => logarithm(Q[reversedStates[index]][to])))), point(action));
  const pathEntropy = logarithm(pathRatio), rateOnly = logarithm(rateRatio), boundary = logarithm(boundaryRatio);
  const tilt = f(BigInt(p.tiltNumerator), BigInt(p.tiltDenominator)), mirror = sub(one, tilt);
  const operator = buildEntropyTilt(rates, pi, tilt), mirrored = equal(tilt, mirror) ? operator : buildEntropyTilt(rates, pi, mirror);
  const homogeneous = equal(a, b) && equal(b, c) && equal(d, e) && equal(e, rateF);
  const homogeneousLogRatio = homogeneous ? logarithm(div(a, d)) : null;
  // For the homogeneous family all tilted rows have the same sum. The
  // independent marked-Poisson expression uses those exact/interval entries.
  const homogeneousExponent = homogeneous ? plus(plus(operator.shifted[0][1], operator.shifted[0][2]), point(neg(exits[0]))) : null;
  const H = negative(logSum(pi.map(x => scale(logarithm(x), x))));
  const history = Array.from({ length: p.sampleSteps + 1 }, (_, step) => {
    const t = f(BigInt(p.timeNumerator) * BigInt(step), BigInt(p.timeDenominator) * BigInt(p.sampleSteps));
    const main = entropyMoment(operator, pi, exits, t), reverse = equal(tilt, mirror) ? main : entropyMoment(mirrored, pi, exits, t);
    const variance = homogeneousLogRatio ? square(homogeneousLogRatio) : null;
    const symmetryDifference = equal(tilt, mirror) ? point(zero) : minus(main.logRange, reverse.logRange);
    return {
      step, time: exact(t), stationaryShannonEntropy: observe(H), meanEntropy: observe(scale(rate, t)),
      moment: main.moment, logMoment: main.logMoment, momentStatus: main.linearStatus, momentProof: main.proof,
      mirrorMoment: reverse.moment, mirrorLogMoment: reverse.logMoment, mirrorMomentStatus: reverse.linearStatus,
      logSymmetryDifference: observe(symmetryDifference), symmetryEnclosuresConsistent: symmetryDifference.lower <= Z && symmetryDifference.upper >= Z,
      expectedLogMomentSign: t.n === Z || detailedBalance || tilt.n === Z || tilt.n === tilt.d ? 0 : tilt.n > Z && tilt.n < tilt.d ? -1 : 1,
      homogeneousClosedLogMoment: homogeneousExponent ? main.proof.method === 'normalization-identity' ? exact(zero) : observe(scale(homogeneousExponent, t)) : null,
      homogeneousEntropyVariance: variance ? observe(scale(variance, mul(add(a, d), t))) : null,
      homogeneousClockwiseCountMean: homogeneous ? exact(mul(a, t)) : null, homogeneousReverseCountMean: homogeneous ? exact(mul(d, t)) : null,
      homogeneousCountReferenceStatus: homogeneous ? 'homogeneous-marked-poisson-reference' as const : 'not-applicable-state-dependent-rates' as const,
    };
  });
  return {
    parameters, precisionBits: 256, generator: Q.map(row => row.map(exact)), stationaryLaw: pi.map(exact), stationaryFlows: flows.map(row => row.map(exact)),
    stationaryCurrent: exact(current), cycleRateRatio: exact(ratio), cycleAffinity: observe(affinity), entropyRate: observe(rate), pairedEntropyRate: observe(pairedRate), edges,
    stationaryShannonEntropy: observe(H), stationaryShannonEntropyRate: exact(zero), adjointGenerator: adjoint.map(row => row.map(exact)),
    adjointMassResidual: adjoint.map(row => exact(sum(row))), stationaryMassResidual: exact(sub(sum(pi), one)),
    path: { states, reversedStates, jumpTimes: jumps.map((_, i) => exact(mul(dwell, f(BigInt(i + 1))))), equalDwellTime: exact(dwell), jumps,
      initialWeight: exact(pi[states[0]]), finalStationaryWeight: exact(pi[states[states.length - 1]]), survivalAction: exact(action),
      rateRatio: exact(rateRatio), boundaryRatio: exact(boundaryRatio), totalRatio: exact(pathRatio), rateOnlyEntropy: observe(rateOnly), systemBoundaryEntropy: observe(boundary),
      totalEntropy: observe(pathEntropy), totalEntropyExactSign: productSign(pathRatio), totalViaBoundarySum: observe(plus(boundary, rateOnly)),
      reversedTotalEntropy: observe(negative(pathEntropy)), logForwardDensityInSeconds: observe(forwardLog), logReversedDensityInSeconds: observe(reversedLog),
      logDensityDifference: observe(minus(forwardLog, reversedLog)), densityReference: 'Lebesgue measure on ordered jump times expressed in seconds; density unit s^-jumpCount, not probability of an exact path',
      construction: 'Deterministic direction word with equal dwells, positive when T>0; not a sampled trajectory. initialState selects this example only; ensemble moments always start from pi.',
    },
    tiltedGenerator: operator.tiltedGenerator, mirrorTiltedGenerator: mirrored.tiltedGenerator, history, current: history[history.length - 1],
    classification: { detailedBalance, entropyRateExactSign: detailedBalance ? 0 : 1, homogeneousRates: homogeneous, evenTimeReversalStates: true, autonomousRates: true, stationaryInitialization: true,
      allReverseEdgesPositive: true, entropyMeasuredInNats: true, thermalBathSpecified: false, sampledPaths: false, pathLawOrTiltRenormalized: false,
      momentNormalizationIdentityAtZeroAndOne: true, finiteTimeSymmetryAppliesToSpecifiedExperiment: true, productionUIImplemented: true },
    approximationContract: 'Autonomous stationary positive three-state CTMC, row-law convention, even states. Exact rational traffic/currents, cycle and path ratios, adjoint and conservation checks; directed256-bit logs. Entropy is in nats, rate in nats/s; no physical heat or bath is inferred. The deterministic direction word has equal dwells over the requested final time and is not a sampled path. Density logs use ordered jump times in seconds and are not exact-path probabilities. Tilted operators keep Q diagonals; they are not made stochastic. Positive shifted Taylor65term enclosures with an explicit norm tail, matrix squaring, and accounted log-scale/mantissa normalization enclose finite-time moments. This arithmetic normalization does not renormalize any path law. Analytic identities give M=1 at time0, tilt0/1 or detailed balance, not an empirical fluctuation test. Computed log moments have a no-jump lower bound and convexity bounds. Positive linear moments use adaptive256..2048bit exponential projections; unavailable upper bounds and unresolved enclosures are distinct from mathematical zero/infinity. Homogeneous count/variance references do not extend to state-dependent rates. Dedicated Worker transport and an interactive reader are implemented. Their build, native lifecycle, accessibility and integration verification are recorded separately.',
  };
}
