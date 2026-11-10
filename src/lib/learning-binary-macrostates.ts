/** Distinguishable, noninteracting binary sites. Full-space probabilities use
 * a specified uniform microstate preparation; fixed-K block probabilities use
 * a separate uniform shell preparation. Neither specifies time evolution.
 */
import { binaryProduct as product, binarySum as sum, binaryMultiply as multiply,
  binaryNegate as negate, binaryObservation as observe, binaryRatioObservation as ratio,
  type BinaryValue } from './learning-binary-products';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
const Z = BigInt(0), O = BigInt(1);
const integer = (n: bigint): BinaryValue => ({ coefficient: n, exponent: 0 });
const difference = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
function natural(n: number, max: number, label: string) {
  if (!Number.isInteger(n) || n < 0 || n > max) throw new RangeError(`Invalid ${label}`);
}
/** Recurrence computes integer counts, not probabilities or a Stirling fit. */
export function binaryMultiplicityRow(sites: number): bigint[] {
  natural(sites, 4096, 'site count');
  const row = [O];
  for (let k = 0; k < sites; k++) row.push(row[k] * BigInt(sites - k) / BigInt(k + 1));
  return row;
}
/** The 4096-site bound is a resource limit of the finite teaching calculator,
 * not a thermodynamic limit. All observations project only after raw products.
 */
export function binaryMacrostateModel(sites: number, excitations: number, blockASites: number, energyGap = kB * 300) {
  natural(sites, 4096, 'site count'); natural(excitations, sites, 'excitation count'); natural(blockASites, sites, 'block size');
  if (typeof energyGap !== 'number' || !Number.isFinite(energyGap) || energyGap <= 0)
    throw new RangeError('Expected a finite positive energy gap');
  const counts = binaryMultiplicityRow(sites), total = O << BigInt(sites), denominator = integer(total);
  const countSum = counts.reduce((a, b) => a + b, Z);
  if (countSum !== total) throw new RangeError('Failed exact full-space normalization');
  const totalLog = product(sites, Math.LN2);
  let macroEntropyNumerator = product(0), conditionalEntropyNumerator = product(0);
  let peak = Z; const macroModes: number[] = [];
  const shells = counts.map((count, k) => {
    if (count > peak) { peak = count; macroModes.length = 0; macroModes.push(k); }
    else if (count === peak) macroModes.push(k);
    const log = observe(integer(count)).logMagnitude!;
    const logRaw = product(log), logInverseProbability = difference(totalLog, logRaw);
    const h = multiply(integer(count), logInverseProbability), conditional = multiply(integer(count), logRaw);
    macroEntropyNumerator = sum([macroEntropyNumerator, h]);
    conditionalEntropyNumerator = sum([conditionalEntropyNumerator, conditional]);
    return { excitations: k, multiplicity: count.toString(), multiplicityObservation: observe(integer(count)),
      probabilityUnderUniformFullSpace: ratio(integer(count), denominator),
      logMultiplicity: observe(logRaw), boltzmannEntropy: observe(multiply(product(kB), logRaw)),
      macroEntropyContribution: ratio(h, denominator),
      conditionalEntropyContribution: ratio(conditional, denominator) };
  });
  const blockBSites = sites - blockASites;
  const aCounts = binaryMultiplicityRow(blockASites), bCounts = binaryMultiplicityRow(blockBSites);
  const shellCount = counts[excitations], minimumA = Math.max(0, excitations - blockBSites), maximumA = Math.min(excitations, blockASites);
  let blockSum = Z, firstMoment = Z, secondMoment = Z, blockPeak = Z;
  const conditionalBlockModes: number[] = [];
  const blockDistribution = Array.from({ length: maximumA - minimumA + 1 }, (_, i) => {
    const a = minimumA + i, b = excitations - a, count = aCounts[a] * bCounts[b];
    blockSum += count; firstMoment += BigInt(a) * count; secondMoment += BigInt(a * a) * count;
    if (count > blockPeak) { blockPeak = count; conditionalBlockModes.length = 0; conditionalBlockModes.push(a); }
    else if (count === blockPeak) conditionalBlockModes.push(a);
    return { excitationsA: a, excitationsB: b, compatibleMicrostates: count.toString(),
      probabilityGivenFixedTotal: ratio(integer(count), integer(shellCount)) };
  });
  if (blockSum !== shellCount) throw new RangeError('Failed exact fixed-shell normalization');
  const varianceNumerator = secondMoment * shellCount - firstMoment * firstMoment;
  if (varianceNumerator < Z) throw new RangeError('Failed exact conditional variance');
  return { scope: 'finite-distinguishable-binary-sites' as const,
    parameters: { sites, excitations, blockASites, blockBSites, energyGap },
    fullSpaceMeasure: 'uniform-over-all-2^N-microstates' as const,
    conditionalMeasure: 'uniform-over-the-selected-fixed-K-shell' as const,
    totalMicrostates: total.toString(), totalMicrostatesObservation: observe(denominator),
    uniformFullSpaceMicrostateProbability: ratio(integer(O), denominator),
    fullSpaceCountSum: countSum.toString(), fullSpaceNormalizedExactly: countSum === total,
    selectedShell: shells[excitations], shells, macroModes,
    uniformWithinSelectedShellProbability: ratio(integer(O), integer(shellCount)),
    selectedEnergy: observe(product(excitations, energyGap)),
    fullSpaceInformationEntropy: observe(totalLog),
    macroInformationEntropy: ratio(macroEntropyNumerator, denominator),
    meanWithinShellInformationEntropy: ratio(conditionalEntropyNumerator, denominator),
    unconditionalMeanExcitations: observe(product(sites, .5)), unconditionalExcitationVariance: observe(product(sites, .25)),
    blockDistribution, conditionalBlockModes, fixedShellCountSum: blockSum.toString(), fixedShellNormalizedExactly: blockSum === shellCount,
    conditionalBlockMean: ratio(integer(firstMoment), integer(shellCount)),
    conditionalBlockVariance: ratio(integer(varianceNumerator), integer(shellCount * shellCount)),
    approximationContract: 'Counts, normalization and moments use exact integers; logarithms and entropies are approximations. Full-space and fixed-shell measures are distinct; no dynamics or thermodynamic-limit assertion.' as const };
}
