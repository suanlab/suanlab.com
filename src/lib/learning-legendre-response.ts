import { calculateScalarGibbs } from './learning-scalar-gibbs';
import { createRealTimeCheckerInterval, type CheckerInterval } from './learning-real-time-checker-interval';
import { CheckerRational } from './learning-real-time-validation-rational';

export type LegendreResponseModel = 'gaussian' | 'spin' | 'quartic';

const interval = createRealTimeCheckerInterval(256);
const { S, one, zero, integer, from, add, sub, multiply, divide, scale, square, log, expNegative } = interval;
const logTwo = log(integer(2));
const toText = (n: bigint) => CheckerRational.of(n, S).text();
const encode = (x: CheckerInterval) => ({ lower: toText(x.lower), upper: toText(x.upper) });
const decode = (x: ReturnType<typeof encode>) => ({
  lower: from(CheckerRational.input(x.lower)).lower,
  upper: from(CheckerRational.input(x.upper)).upper,
});

const quartic = { quartic: .25, quadratic: -.5, coupling: 0, radius: 4, panels: 128 };
const positivePart = (x: CheckerInterval) => ({ lower: x.lower < zero.lower ? zero.lower : x.lower, upper: x.upper });

export function calculateLegendreResponse(model: LegendreResponseModel) {
  if (model !== 'gaussian' && model !== 'spin' && model !== 'quartic') {
    throw new RangeError('Unknown Legendre response model');
  }
  const maxTicks = model === 'quartic' ? 8 : 32;
  const base = model === 'quartic' ? calculateScalarGibbs({ ...quartic, sources: [0] }) : undefined;

  return Array.from({ length: 2 * maxTicks + 1 }, (_, index) => {
    const tick = index - maxTicks;
    const source = from(CheckerRational.of(BigInt(tick), BigInt(8)));
    let generator: CheckerInterval;
    let mean: CheckerInterval;
    let susceptibility: CheckerInterval;
    if (base) {
      const result = tick === 0 ? base : calculateScalarGibbs({ ...quartic, sources: [tick / 8] });
      mean = decode(result.means[0]);
      susceptibility = decode(result.covariance[0][0]);
      generator = tick === 0 ? zero : sub(decode(result.logPartition), decode(base.logPartition));
    } else if (model === 'gaussian') {
      // Precision 2 yields W(h) = h²/4.
      mean = divide(source, integer(2));
      generator = divide(square(source), integer(4));
      susceptibility = divide(one, integer(2));
    } else {
      const magnitude = from(CheckerRational.of(BigInt(Math.abs(tick)), BigInt(8)));
      const q = expNegative(scale(magnitude, 2));
      const total = add(one, q);
      mean = scale(divide(sub(one, q), total), Math.sign(tick));
      generator = tick === 0 ? zero : sub(add(magnitude, log(total)), logTwo);
      susceptibility = divide(scale(q, 4), square(total));
    }
    // For the even base measure W(0)=0 is its minimum and Γ(m) ≥ 0.
    const effectiveAction = positivePart(sub(multiply(source, mean), generator));
    return {
      source: tick / 8,
      mean: encode(mean),
      generator: encode(positivePart(generator)),
      susceptibility: encode(susceptibility),
      effectiveAction: encode(effectiveAction),
      effectiveCurvature: susceptibility.lower > zero.lower ? encode(divide(one, susceptibility)) : null,
    };
  });
}
