/** Information measures already contain approximate transcendental factors.
 * Round a nonnegative rational term to 192 significant binary digits, ties to
 * even, with unrestricted exponent. Relative error <=2^-192 for a nonzero
 * term. A positive term never becomes zero; probabilities are not rounded here.
 */
export const informationArithmeticBits = 192;
export function roundInformationTerm(n: bigint, d: bigint): { n: bigint; d: bigint } {
  const Z = BigInt(0), O = BigInt(1), T = BigInt(2);
  if (n < Z || d <= Z) throw new RangeError('Invalid nonnegative information fraction');
  if (n === Z) return { n: Z, d: O };
  let leading = n.toString(2).length - d.toString(2).length;
  if (leading >= 0 ? n < (d << BigInt(leading)) : (n << BigInt(-leading)) < d) leading--;
  const quantum = leading - (informationArithmeticBits - 1);
  const numerator = quantum < 0 ? n << BigInt(-quantum) : n;
  const denominator = quantum > 0 ? d << BigInt(quantum) : d;
  let mantissa = numerator / denominator; const remainder = numerator % denominator;
  if (T * remainder > denominator || T * remainder === denominator && mantissa % T !== Z) mantissa++;
  if (quantum >= 0) return { n: mantissa << BigInt(quantum), d: O };
  let power = -quantum;
  while (power > 0 && mantissa % T === Z) { mantissa /= T; power--; }
  return { n: mantissa, d: O << BigInt(power) };
}
