/** Finite IID qubit-source spectral weight, without measuring the source.
 * This computes a projector weight, not a codec's entanglement fidelity. */
import { f, q, zero, one, add, sub, mul, le, type F } from './learning-quantum-hilbert-core';
const pow = (a: F, exponent: number) => f(a.n ** BigInt(exponent), a.d ** BigInt(exponent));
export function binarySourceBlockWeight(probabilityOne: F, copies: number, retainedRank: bigint) {
  if (!probabilityOne || typeof probabilityOne.n !== 'bigint' || typeof probabilityOne.d !== 'bigint'
    || probabilityOne.d <= BigInt(0) || probabilityOne.n.toString().length > 32 || probabilityOne.d.toString().length > 32
    || !le(zero, probabilityOne) || !le(probabilityOne, one)) throw new RangeError('A rational probability in [0,1] with at most32 digits is required');
  if (!Number.isSafeInteger(copies) || copies < 1 || copies > 64) throw new RangeError('Use between1 and64 independent copies');
  const dimension = BigInt(1) << BigInt(copies);
  if (typeof retainedRank !== 'bigint' || retainedRank < BigInt(0) || retainedRank > dimension) throw new RangeError('Retained rank must lie between0 and2^copies');
  const probabilityZero = sub(one, probabilityOne);
  let multiplicity = BigInt(1);
  const groups = Array.from({ length: copies + 1 }, (_, ones) => {
    if (ones) multiplicity = multiplicity * BigInt(copies - ones + 1) / BigInt(ones);
    return { ones, multiplicity, eigenvalue: mul(pow(probabilityOne, ones), pow(probabilityZero, copies - ones)) };
  }).sort((a, b) => { const difference = a.eigenvalue.n * b.eigenvalue.d - b.eigenvalue.n * a.eigenvalue.d;
    return difference > BigInt(0) ? -1 : difference < BigInt(0) ? 1 : a.ones - b.ones; });
  let remaining = retainedRank, weight = zero;
  const selections = groups.map(group => {
    const kept = remaining < group.multiplicity ? remaining : group.multiplicity;
    remaining -= kept; weight = add(weight, mul(f(kept), group.eigenvalue));
    return { ones: group.ones, multiplicity: group.multiplicity.toString(), retained: kept.toString(), eigenvalue: q(group.eigenvalue) };
  });
  let requiredQubits: number | null = null;
  if (retainedRank > BigInt(0)) {
    requiredQubits = 0;
    while ((BigInt(1) << BigInt(requiredQubits)) < retainedRank) requiredQubits++;
  }
  return { copies, dimension: dimension.toString(), retainedRank: retainedRank.toString(), requiredQubits,
    retainedWeight: q(weight), discardedWeight: q(sub(one, weight)), groups: selections,
    interpretation: 'Largest-eigenvalue rank-constrained projector weight for a binary IID source. Equal eigenvalues use increasing Hamming weight as a canonical tie order. A zero rank has no nonzero encoding space. No encoder, decoder, entanglement-fidelity guarantee, typicality cutoff or asymptotic converse is computed.' };
}
