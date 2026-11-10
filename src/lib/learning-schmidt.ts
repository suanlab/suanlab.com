/** Exact density algebra and certified spectra; no arbitrary degenerate vectors. */
import { c, cs, cc, cm, scaleC, norm2, f, q, cq, normalized, zero, one, sub, div, sqrtRange, type C, type F } from './learning-quantum-hilbert-core';
import { product, adjoint, matrixScale, matrixSub, matrixSquaredNorm, matrixObservation as mq, trace, type Matrix as M } from './learning-observable-matrix';
import { variationalCharacteristic } from './learning-quantum-variational-math';
import { isolateTensorSpectrum } from './learning-tensor-roots';
import { createRitzIntervalContext, type RitzInterval as I, type RitzMatrix as IM } from './learning-ritz-interval';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { snapshotSchmidtParameters, type SchmidtParameters } from './learning-schmidt-input';
export { createSchmidtParameters, isSchmidtParameters, snapshotSchmidtParameters, schmidtParameterKeys, type SchmidtParameters } from './learning-schmidt-input';
const int = (n: number) => f(BigInt(n));
const transpose = (a: M): M => a[0].map((_, j) => a.map(row => row[j]));
const conjugate = (a: M): M => a.map(row => row.map(cc));
const isZero = (z: C) => z.re.n === BigInt(0) && z.im.n === BigInt(0);
const divideC = (a: C, b: C) => scaleC(cm(a, cc(b)), div(one, norm2([b])));
function rank(value: M): number {
  const a = value.map(row => [...row]); let row = 0;
  for (let col = 0; col < a[0].length && row < a.length; col++) {
    const pivot = a.findIndex((v, i) => i >= row && !isZero(v[col])); if (pivot < 0) continue;
    [a[row], a[pivot]] = [a[pivot], a[row]];
    for (let i = row + 1; i < a.length; i++) if (!isZero(a[i][col])) {
      const factor = divideC(a[i][col], a[row][col]); for (let j = col; j < a[0].length; j++) a[i][j] = cs(a[i][j], cm(factor, a[row][j]));
    }
    row++;
  }
  return row;
}
export function schmidtModel(value: SchmidtParameters) {
  const parameters = snapshotSchmidtParameters(value), p = parameters, state = Array.from({ length: 2 ** p.sites }, (_, i) => c(int(p[`stateReal${i}`]), int(p[`stateImag${i}`]))), N = norm2(state);
  const ctx = createRitzIntervalContext(), transcendental = createTranscendentalContext(768);
  const convert = (a: M): IM => a.map(row => row.map(ctx.complex));
  const plus = (a: IM, b: IM): IM => a.map((row, i) => row.map((v, j) => ctx.complexAdd(v, b[i][j])));
  const scale = (a: IM, x: I): IM => a.map(row => row.map(v => ctx.complexScale(v, x)));
  const zeros = (n: number, m = n): IM => Array.from({ length: n }, () => Array.from({ length: m }, () => ctx.complex(c(zero))));
  const iNorm2 = (a: IM) => a.reduce((s, row) => ctx.add(s, ctx.normSquared(row)), ctx.zero);
  const rootInterval = (x: I): I => {
    const scaleFactor = BigInt(1) << BigInt(ctx.workBits);
    if (x.upper < BigInt(0)) throw new RangeError('Nonnegative root interval required');
    return ctx.range({ lower: sqrtRange(f(x.lower < BigInt(0) ? BigInt(0) : x.lower, scaleFactor)).lower, upper: sqrtRange(f(x.upper, scaleFactor)).upper });
  };
  const logarithm = (x: F) => transcendental.logRatio({ coefficient: x.n, exponent: 0 }, { coefficient: x.d, exponent: 0 });
  const ln2 = logarithm(int(2)), iLn2 = ctx.range({ lower: f(ln2.lower, transcendental.scale), upper: f(ln2.upper, transcendental.scale) });
  const isolationCache = new Map<string, ReturnType<typeof isolateTensorSpectrum>>();
  const cuts = Array.from({ length: p.sites - 1 }, (_, cutIndex) => {
    const cut = cutIndex + 1, leftDimension = 2 ** cut, rightDimension = state.length / leftDimension;
    const coefficient: M = Array.from({ length: leftDimension }, (_, i) => state.slice(i * rightDimension, (i + 1) * rightDimension));
    const rhoLeft = matrixScale(product(coefficient, adjoint(coefficient)), div(one, N));
    const rhoRight = matrixScale(product(transpose(coefficient), conjugate(coefficient)), div(one, N));
    const smallerSide = leftDimension <= rightDimension ? 'left' : 'right', density = smallerSide === 'left' ? rhoLeft : rhoRight, size = density.length, schmidtRank = rank(coefficient);
    const polynomial = variationalCharacteristic(density), key = polynomial.map(x => `${x.n}/${x.d}`).join(',');
    let roots = isolationCache.get(key); if (!roots) { roots = isolateTensorSpectrum(polynomial, 4096); isolationCache.set(key, roots); }
    const idensity = convert(density), iCoefficient = convert(coefficient);
    const spectrum = roots.roots.slice().reverse().map((root, index, descending) => {
      if (root.upper.n < BigInt(0) || sub(root.lower, one).n > BigInt(0)) throw new RangeError('Density spectrum outside probability range');
      const weight = ctx.range(root), zeroWeight = root.exact && root.lower.n === BigInt(0);
      let P = ctx.identity(size);
      descending.forEach((other, j) => { if (j === index) return;
        const otherWeight = ctx.range(other), gap = ctx.sub(weight, otherWeight);
        const factor = idensity.map((row, i) => row.map((v, k) => i === k ? ctx.complexSubtract(v, { real: otherWeight, imaginary: ctx.zero }) : v));
        P = scale(ctx.product(P, factor), ctx.divide(ctx.one, gap));
      });
      let entropy = ctx.zero;
      if (!zeroWeight) {
        if (root.lower.n <= BigInt(0)) throw new RangeError('Positive root must be separated from zero');
        const logLower = logarithm(root.lower), logUpper = logarithm(root.upper);
        const log = ctx.range({ lower: f(logLower.lower, transcendental.scale), upper: f(logUpper.upper, transcendental.scale) });
        entropy = ctx.neg(ctx.multiply(weight, log));
      }
      return { root, weight, zeroWeight, projector: P, entropy };
    });
    if (spectrum.filter(v => !v.zeroWeight).reduce((n, v) => n + v.root.multiplicity, 0) !== schmidtRank) throw new RangeError('Exact rank and positive root multiplicities disagree');
    let remaining = Math.min(p.retainedRank, schmidtRank), keptWeight = ctx.zero, retainedProjector = zeros(size), boundaryDegenerate = false;
    const allocation = spectrum.map((v, groupIndex) => {
      const kept = v.zeroWeight ? 0 : Math.min(remaining, v.root.multiplicity); remaining -= kept;
      keptWeight = ctx.add(keptWeight, ctx.multiply(ctx.from(int(kept)), v.weight));
      if (kept > 0 && kept < v.root.multiplicity) boundaryDegenerate = true;
      if (kept === v.root.multiplicity && !v.zeroWeight) retainedProjector = plus(retainedProjector, v.projector);
      return { groupIndex, keptMultiplicity: kept, discardedMultiplicity: v.zeroWeight ? 0 : v.root.multiplicity - kept };
    });
    // Exact trace1 fixes the mass when the entire positive support is retained.
    if (p.retainedRank >= schmidtRank) keptWeight = ctx.one;
    const discardedWeight = ctx.sub(ctx.one, keptWeight), entropy = spectrum.reduce((s, v) => ctx.add(s, ctx.multiply(ctx.from(int(v.root.multiplicity)), v.entropy)), ctx.zero);
    const retainedPositiveRank = Math.min(p.retainedRank, schmidtRank), normalizedDefined = retainedPositiveRank > 0;
    const normalizedDistance = normalizedDefined ? ctx.multiply(ctx.from(int(2)), ctx.sub(ctx.one, rootInterval(keptWeight))) : null;
    let chosen = null;
    if (!boundaryDegenerate) {
      const truncated = smallerSide === 'left' ? ctx.product(retainedProjector, iCoefficient) : ctx.product(iCoefficient, retainedProjector[0].map((_, j) => retainedProjector.map(row => row[j])));
      const rest = ctx.matrixSubtract(iCoefficient, truncated), normInput = rootInterval(ctx.from(N));
      const normalizedTruncatedNorm = normalizedDefined ? rootInterval(ctx.multiply(ctx.from(N), keptWeight)) : null;
      const overlap = ctx.dot(truncated.flat(), rest.flat());
      chosen = {
        retainedPositiveSupportProjector: ctx.observeMatrix(retainedProjector), truncatedRawCoefficientMatrix: ctx.observeMatrix(truncated),
        amplitudesOnNormalizedInput: ctx.observeMatrix(scale(truncated, ctx.divide(ctx.one, normInput))).flat(),
        normalizedAmplitudes: normalizedTruncatedNorm ? ctx.observeMatrix(scale(truncated, ctx.divide(ctx.one, normalizedTruncatedNorm))).flat() : null,
        directSquaredError: ctx.observation(ctx.divide(iNorm2(rest), ctx.from(N))), directRetainedNormSquared: ctx.observation(ctx.divide(iNorm2(truncated), ctx.from(N))),
        directedResiduals: {
          errorAccount: ctx.observation(ctx.sub(ctx.divide(iNorm2(rest), ctx.from(N)), discardedWeight)),
          retainedNormAccount: ctx.observation(ctx.sub(ctx.divide(iNorm2(truncated), ctx.from(N)), keptWeight)),
          orthogonality: ctx.observeComplex(ctx.complexScale(overlap, ctx.divide(ctx.one, ctx.from(N)))),
          projectorIdempotence: ctx.observeMatrix(ctx.matrixSubtract(ctx.product(retainedProjector, retainedProjector), retainedProjector)),
          projectorHermiticity: ctx.observeMatrix(ctx.matrixSubtract(retainedProjector, ctx.adjoint(retainedProjector))),
        },
      };
    }
    const sumP = spectrum.reduce((s, v) => plus(s, v.projector), zeros(size));
    const reconstructed = spectrum.reduce((s, v) => plus(s, scale(v.projector, v.weight)), zeros(size));
    const sumWeight = spectrum.reduce((s, v) => ctx.add(s, ctx.multiply(ctx.from(int(v.root.multiplicity)), v.weight)), ctx.zero);
    const spectralPurity = spectrum.reduce((s, v) => ctx.add(s, ctx.multiply(ctx.from(int(v.root.multiplicity)), ctx.square(v.weight))), ctx.zero);
    const purity = matrixSquaredNorm(density);
    return {
      cut, leftDimension, rightDimension, smallerSide, smallerDimension: size, coefficientMatrix: mq(coefficient), rhoLeft: mq(rhoLeft), rhoRight: mq(rhoRight), schmidtRank,
      leftZeroMultiplicity: leftDimension - schmidtRank, rightZeroMultiplicity: rightDimension - schmidtRank,
      characteristicPolynomial: polynomial.map(q), purity: q(purity), linearEntropy: q(sub(one, purity)), entropyNats: ctx.observation(entropy), entropyBits: ctx.observation(ctx.divide(entropy, iLn2)),
      spectrum: spectrum.map(v => ({ weight: ctx.probability(v.weight), singularValue: ctx.observation(rootInterval(v.weight)), multiplicity: v.root.multiplicity, zeroWeight: v.zeroWeight, projectorOnSmallerPhysicalSide: ctx.observeMatrix(v.projector) })),
      truncation: {
        rankBudget: p.retainedRank, retainedPositiveRank, allocation, boundaryDegenerate, optimalUnnormalizedStateUnique: !boundaryDegenerate,
        normalizedStateDefinedForEveryOptimum: normalizedDefined, chosenStateAvailable: chosen !== null,
        retainedWeight: ctx.probability(keptWeight), discardedWeight: ctx.probability(discardedWeight), optimalUnnormalizedSquaredError: ctx.probability(discardedWeight),
        optimalNormalizedFidelitySquared: normalizedDefined ? ctx.probability(keptWeight) : null,
        optimalNormalizedDistanceSquared: normalizedDistance ? ctx.observation(normalizedDistance) : null,
        chosen, noArbitraryDegenerateEigenvectorsChosen: true,
      },
      exactResiduals: {
        leftHermiticity: q(matrixSquaredNorm(matrixSub(rhoLeft, adjoint(rhoLeft)))), rightHermiticity: q(matrixSquaredNorm(matrixSub(rhoRight, adjoint(rhoRight)))),
        leftTrace: cq(cs(trace(rhoLeft), c(one))), rightTrace: cq(cs(trace(rhoRight), c(one))), purityAgreement: q(sub(matrixSquaredNorm(rhoLeft), matrixSquaredNorm(rhoRight))),
      },
      directedResiduals: {
        projectorCompleteness: ctx.observeMatrix(ctx.matrixSubtract(sumP, ctx.identity(size))), densityReconstruction: ctx.observeMatrix(ctx.matrixSubtract(reconstructed, idensity)), probabilityMass: ctx.observation(ctx.sub(sumWeight, ctx.one)), purityAccount: ctx.observation(ctx.sub(spectralPurity, ctx.from(purity))),
        spectralGroups: spectrum.map(v => ({ hermiticity: ctx.observeMatrix(ctx.matrixSubtract(v.projector, ctx.adjoint(v.projector))), idempotence: ctx.observeMatrix(ctx.matrixSubtract(ctx.product(v.projector, v.projector), v.projector)), eigenEquation: ctx.observeMatrix(ctx.matrixSubtract(ctx.product(idensity, v.projector), scale(v.projector, v.weight))), traceMultiplicity: ctx.observeComplex(ctx.complexSubtract(ctx.trace(v.projector), ctx.complex(c(int(v.root.multiplicity))))) })),
      },
    };
  });
  return {
    parameters, sites: p.sites, dimension: state.length, state: state.map(cq), normalizedState: state.map(v => normalized(v, N)), rawNormSquared: q(N), selectedCut: p.selectedCut, selectedCutIndex: p.selectedCut - 1, cuts,
    arithmetic: 'exact-density-algebra-and-certified-spectral-projectors', rootIsolationBits: 4096, intervalWorkBits: 2048, logarithmPrecisionBits: 768, outputPrecisionBits: 512,
    approximationContract: '2..4 ordered qubits,35 owned bounded integer fields and nonzero Gaussian-integer raw state. All contiguous cuts use physical rhoLeft=C C†/N and rhoRight=Cᵀ C*/N;the smaller dimension<=4 is explicitly identified. Exact rank/density/purity;4096-bit square-free Sturm spectra with exact multiplicities,2048-bit spectral projectors and768-bit directed logarithms,outward512-bit observations. Rank budget0..4 means at most that rank;positive support is retained. Degenerate positive cutoff splits have a well-defined optimal error/fidelity but no chosen projector or state. Zero-budget normalized outputs are null. Right-side truncation uses C Pᵀ. No arbitrary eigenvectors,automatic MPS compression,DMRG,multicut-global-optimality or infinite-size claim is supplied.',
  };
}
