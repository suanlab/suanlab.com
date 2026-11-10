/** Exact finite binary reservoir versus a matching canonical preparation.
 * Distinguishable noninteracting sites; no dynamics or ensemble equivalence
 * assertion. Probabilities/counts/moments remain exact before observation.
 */
import { binaryMultiplicityRow } from './learning-binary-macrostates';
import { binaryProduct, binaryMultiply, binarySum, binaryNegate, binaryObservation,
  binaryRatioObservation, type BinaryValue } from './learning-binary-products';
import { roundInformationTerm } from './learning-information-arithmetic';
const Z = BigInt(0), O = BigInt(1), T = BigInt(2);
const integer = (coefficient: bigint): BinaryValue => ({ coefficient, exponent: 0 });
const absolute = (n: bigint) => n < Z ? -n : n;
const probability = (n: bigint, d: bigint) => binaryRatioObservation(integer(n), integer(d));
function gcd(a: number, b: number): number { while (b) { const r = a % b; a = b; b = r; } return a; }
/** Information terms contain approximate transcendental factors. Keep their
 * unrestricted exponent; the probabilities themselves are never rounded here.
 */
function informationTerm(n: BinaryValue, denominator: bigint): BinaryValue {
  const rawN = n.exponent >= 0 ? n.coefficient << BigInt(n.exponent) : n.coefficient;
  const rawD = n.exponent < 0 ? denominator << BigInt(-n.exponent) : denominator;
  const r = roundInformationTerm(rawN, rawD);
  return { coefficient: r.n, exponent: -(r.d.toString(2).length - 1) };
}
function entropy(weights: bigint[], total: bigint) {
  const terms = weights.filter(w => w > Z && w < total).map(w => {
    const gap = total - w;
    if (gap * BigInt(8) <= total) {
      const delta = probability(gap, total).value ?? 0;
      const gain = delta === 0 ? 1 : -Math.log1p(-delta) / delta;
      return informationTerm(binaryMultiply(integer(w * gap), binaryProduct(gain)), total * total);
    }
    const p = probability(w, total);
    const log = p.value === null || p.value < 2 ** -1022 ? p.logMagnitude! : Math.log(p.value);
    return informationTerm(binaryMultiply(integer(w), binaryProduct(-log)), total);
  });
  return binaryObservation(binarySum(terms));
}
function divergence(u: bigint[], U: bigint, v: bigint[], V: bigint) {
  const terms: BinaryValue[] = [], D = U * V;
  for (let i = 0; i < u.length; i++) {
    const a = u[i] * V, b = v[i] * U, gap = a - b;
    if (b === Z) { if (a > Z) return { value: null, sign: 1, logMagnitude: null, status: 'infinite' as const }; continue; }
    if (gap === Z) continue;
    if (a === Z) { terms.push(informationTerm(integer(b), D)); continue; }
    if (absolute(gap) * BigInt(8) <= b) {
      const delta = probability(gap, b).value ?? 0;
      let gain = 0;
      for (let j = 32; j >= 0; j--) gain = 1 / ((j + 1) * (j + 2)) - delta * gain;
      terms.push(informationTerm(binaryMultiply(integer(gap * gap), binaryProduct(gain)), b * D));
    } else {
      const r = probability(a, b);
      const log = r.value === null || r.value < 2 ** -1022 ? r.logMagnitude! : Math.log(r.value);
      const numerator = binarySum([binaryMultiply(integer(a), binaryProduct(log)), binaryNegate(integer(gap))]);
      if (numerator.coefficient <= Z) throw new RangeError('Failed positive reservoir divergence');
      terms.push(informationTerm(numerator, D));
    }
  }
  return { ...binaryObservation(binarySum(terms)), status: 'finite' as const };
}
export function finiteReservoirModel(subsystemSites: number, reservoirSites: number, totalExcitations: number) {
  for (const [label, value] of [['subsystem', subsystemSites], ['reservoir', reservoirSites]] as const)
    if (!Number.isInteger(value) || value < 0 || value > 4096) throw new RangeError(`Invalid ${label} size`);
  const N = subsystemSites + reservoirSites;
  if (N < 1 || N > 4096) throw new RangeError('Expected 1 to 4096 total sites');
  if (!Number.isInteger(totalExcitations) || totalExcitations < 0 || totalExcitations > N)
    throw new RangeError('Invalid total excitation count');
  const n = subsystemSites, b = reservoirSites, K = totalExcitations;
  const whole = binaryMultiplicityRow(N), sub = binaryMultiplicityRow(n), bath = binaryMultiplicityRow(b);
  const denominator = whole[K];
  const counts = sub.map((count, a) => count * (bath[K - a] ?? Z));
  const common = gcd(K, N), k = BigInt(K / common), d = BigInt(N / common), rest = d - k;
  const canonicalDenominator = d ** BigInt(n);
  const canonicalCounts: bigint[] = [];
  if (k === Z || rest === Z) for (let a = 0; a <= n; a++) canonicalCounts.push(a === (k === Z ? 0 : n) ? O : Z);
  else {
    let count = rest ** BigInt(n); canonicalCounts.push(count);
    for (let a = 0; a < n; a++) {
      const numerator = count * BigInt(n - a) * k, divisor = BigInt(a + 1) * rest;
      if (numerator % divisor !== Z) throw new RangeError('Failed exact canonical recurrence');
      count = numerator / divisor; canonicalCounts.push(count);
    }
  }
  if (counts.reduce((a, x) => a + x, Z) !== denominator || canonicalCounts.reduce((a, x) => a + x, Z) !== canonicalDenominator)
    throw new RangeError('Failed exact reservoir normalization');
  const first = counts.reduce((s, count, a) => s + BigInt(a) * count, Z);
  const second = counts.reduce((s, count, a) => s + BigInt(a * a) * count, Z);
  const varianceNumerator = second * denominator - first * first;
  const meanN = BigInt(n) * k, canonicalVarN = BigInt(n) * k * rest;
  if (first * d !== meanN * denominator || varianceNumerator < Z)
    throw new RangeError('Failed exact reservoir moments');
  if (N > 1 && varianceNumerator * d * d * BigInt(N - 1) !== canonicalVarN * BigInt(b) * denominator * denominator)
    throw new RangeError('Failed finite population correction');
  const tvN = counts.reduce((s, count, a) => s + absolute(count * canonicalDenominator - canonicalCounts[a] * denominator), Z);
  return {
    scope: 'finite-uniform-binary-shell-versus-matching-canonical' as const,
    parameters: { subsystemSites: n, reservoirSites: b, totalExcitations: K, totalSites: N },
    wholeShellCount: denominator.toString(), canonicalCommonDenominator: canonicalDenominator.toString(),
    distribution: counts.map((count, a) => ({ excitations: a, compatibleMicrostates: count.toString(),
      microcanonical: probability(count, denominator), canonical: probability(canonicalCounts[a], canonicalDenominator),
      canonicalNumerator: canonicalCounts[a].toString() })),
    meanExcitations: probability(first, denominator), varianceExcitations: probability(varianceNumerator, denominator * denominator),
    canonicalVariance: probability(canonicalVarN, d * d), totalVariation: probability(tvN, T * denominator * canonicalDenominator),
    entropyNats: entropy(counts, denominator),
    microToCanonicalKlNats: divergence(counts, denominator, canonicalCounts, canonicalDenominator),
    canonicalToMicroKlNats: divergence(canonicalCounts, canonicalDenominator, counts, denominator),
    exactlyEqual: tvN === Z,
    approximationContract: 'Exact counts, normalization, moments and total variation; approximate logarithms and 192-significant-bit information terms. The 4096-site resource bound is not a thermodynamic limit.' as const,
  };
}
