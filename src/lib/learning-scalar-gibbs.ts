/** Finite lattice quartic Gibbs integrals with directed enclosures. */
import { createRealTimeCheckerInterval, type CheckerInterval } from './learning-real-time-checker-interval';
import { CheckerRational } from './learning-real-time-validation-rational';

import { readScalarGibbsInput } from './learning-scalar-gibbs-input';

const arithmetic = createRealTimeCheckerInterval(256);
const { S, zero, one, integer, point, from, add, sub, multiply, divide, divideInteger, square, scale, expNegative } = arithmetic;
const ZERO = BigInt(0);
const max = (a: bigint, b: bigint) => a > b ? a : b;
const min = (a: bigint, b: bigint) => a < b ? a : b;
const absolute = (x: bigint) => x < ZERO ? -x : x;
const bound = (x: CheckerInterval) => max(absolute(x.lower), absolute(x.upper));
const fraction = (x: number) => from(CheckerRational.of(BigInt(16 * x), BigInt(16)));
const text = (x: bigint) => CheckerRational.of(x, S).text();
const wire = (x: CheckerInterval) => ({ lower: text(x.lower), upper: text(x.upper) });
const sum = (xs: CheckerInterval[]) => xs.reduce(add, zero);
const product = (xs: CheckerInterval[]) => xs.reduce(multiply, one);

function powers(x: CheckerInterval) {
  const x2 = square(x);
  return [one, x, x2, multiply(x2, x), square(x2)];
}

/** Absolute moment of exp(-alpha*x^4) outside [-R,R]. */
function tailMoment(alpha: CheckerInterval, radius: number, order: number) {
  const rate = multiply(alpha, integer(4 * radius ** 3));
  // Integrate the expansion of (R+u)^order against exp(-rate*u).
  // binom(order,k) k! can be accumulated as a falling factorial.
  let falling = 1, inversePower = divide(one, rate), integral = zero;
  for (let k = 0; k <= order; k++) {
    integral = add(integral, scale(inversePower, falling * radius ** (order-k)));
    falling *= order-k;
    inversePower = divide(inversePower, rate);
  }
  return scale(multiply(expNegative(scale(alpha, radius ** 4)), integral), 2);
}

export function calculateScalarGibbs(value: unknown) {
  const input = readScalarGibbsInput(value);
  if (!input) throw new RangeError('Expected bounded one- or two-site scalar Gibbs input');
  const { quartic, quadratic, coupling, sources, panels, radius } = input;
  const count = sources.length, a = fraction(quartic), b = fraction(quadratic), c = fraction(coupling);
  const alpha = divideInteger(a, 2), halfCoupling = divideInteger(c, 2);
  const shiftedSites = sources.map(h => {
    const halfAbs = fraction(Math.abs(h)/2);
    const d = fraction(Math.max(0,-quadratic) + Math.abs(h)/2);
    return { h: fraction(h), sign: integer(Math.sign(h)), halfAbs,
      centerSquare: divide(d,a), offset: add(divide(square(d),scale(a,2)),halfAbs) };
  });
  const positiveB = fraction(Math.max(0,quadratic)), offset = sum(shiftedSites.map(site => site.offset));
  const width = divideInteger(integer(2 * radius),panels);
  const cellVolume = count === 1 ? width : square(width);
  const midpointFactor = divideInteger(square(width),24);
  const grid = Array.from({ length: panels }, (_,i) => {
    const lo = add(integer(-radius),scale(width,i));
    const hi = add(lo,width);
    return { interval: powers({ lower: lo.lower,upper: hi.upper }), midpoint: powers(divideInteger(add(lo,hi),2)) };
  });
  // Constant, marginal first/second/fourth moments, and the mixed moment.
  const exponents = count === 1 ? [[0],[1],[2],[4]] : [[0,0],[1,0],[0,1],[2,0],[0,2],[4,0],[0,4],[1,1]];
  const symmetric = sources.every(h => h === 0);
  const totals = exponents.map(orders => ({
    orders, odd: symmetric && orders.reduce((x,y) => x+y,0)%2 === 1,
    positive: orders.every(p => p%2 === 0), midpoint: zero, box: zero,
  }));
  function evaluate(p: CheckerInterval[][]) {
    let action = sum(p.map((xs,i) => {
      const site = shiftedSites[i];
      return sum([multiply(alpha,xs[4]),multiply(alpha,square(sub(xs[2],site.centerSquare))),
        multiply(positiveB,xs[2]),multiply(site.halfAbs,square(sub(xs[1],site.sign)))]);
    }));
    if (count === 2) action = add(action,multiply(halfCoupling,square(sub(p[0][1],p[1][1]))));
    return expNegative(action);
  }
  function accumulate(cells: typeof grid) {
    const boxPowers = cells.map(x => x.interval);
    const midPowers = cells.map(x => x.midpoint);
    const boxWeight = evaluate(boxPowers), midWeight = evaluate(midPowers);
    const gradients = boxPowers.map((xs,i) => {
      const local = sub(add(multiply(scale(a,4),xs[3]),multiply(scale(b,2),xs[1])),shiftedSites[i].h);
      return count === 1 ? local : add(local,multiply(c,sub(xs[1],boxPowers[1-i][1])));
    });
    const curvatures = boxPowers.map(xs => sum([multiply(scale(a,12),xs[2]),scale(b,2),c]));
    for (const total of totals) {
      if (total.odd) continue;
      const monomial = (values: CheckerInterval[][], axis = -1, derivative = 0) => product(total.orders.map((order,i) => {
        if (i !== axis) return values[i][order];
        if (order < derivative) return zero;
        return scale(values[i][order-derivative],derivative === 1 ? order : order*(order-1));
      }));
      const p = monomial(boxPowers), mid = multiply(monomial(midPowers),midWeight);
      const derivatives = total.orders.map((_,i) => multiply(boxWeight,
        add(sub(monomial(boxPowers,i,2),scale(multiply(monomial(boxPowers,i,1),gradients[i]),2)),
          multiply(p,sub(square(gradients[i]),curvatures[i])))));
      const error = multiply(midpointFactor,point(derivatives.reduce((s,x) => s+bound(x),ZERO))).upper;
      const rectangle = multiply(p,boxWeight);
      total.midpoint = add(total.midpoint,mid);
      total.box = add(total.box,{ lower: max(rectangle.lower,mid.lower-error),upper: min(rectangle.upper,mid.upper+error) });
    }
  }
  for (const x of grid) {
    if (count === 1) accumulate([x]);
    else for (const y of grid) accumulate([x,y]);
  }
  const tails = [0,1,2,3,4].map(p => tailMoment(alpha,radius,p));
  const envelope = tails.map((tail,p) => add(tail,divideInteger(integer(2*radius ** (p+1)),p+1)));
  const integrals = totals.map(({ orders,midpoint,box,odd,positive }) => {
    const inside = multiply(box,cellVolume);
    const tail = odd ? ZERO : sum(orders.map((p,i) => product(orders.map((q,j) => i === j ? tails[p] : envelope[q])))).upper;
    const full = { lower: positive ? inside.lower : inside.lower-tail,upper: inside.upper+tail };
    return { orders, midpoint: multiply(midpoint,cellVolume), inside, tail, full };
  });
  const mass = integrals[0].full;
  if (mass.lower <= ZERO) throw Error('Positive mass enclosure required');
  const moment = (orders: number[]) => divide(integrals.find(x => x.orders.every((p,i) => p === orders[i]))!.full,mass);
  const marginal = (site: number,power: number) => moment(sources.map((_,i) => i === site ? power : 0));
  const means = sources.map((_,i) => marginal(i,1));
  const covariance = means.map((mean,i) => means.map((other,j) => {
    if (i !== j) return sub(moment([1,1]),multiply(mean,other));
    const variance = sub(marginal(i,2),square(mean));
    return { lower: max(ZERO,variance.lower),upper: variance.upper };
  }));
  const logPartition = add(offset,arithmetic.log(mass));
  return {
    input, shift: wire(offset), scaledPartition: wire(mass), logPartition: wire(logPartition),
    freeEnergyOverTemperature: wire(arithmetic.neg(logPartition)), means: means.map(wire),
    secondMoments: sources.map((_,i) => wire(marginal(i,2))),
    fourthMoments: sources.map((_,i) => wire(marginal(i,4))),
    covariance: covariance.map(row => row.map(wire)),
    originHessian: sources.map((_,i) => sources.map((__,j) => i === j ? 2*quadratic+coupling : -coupling)),
    integrals: integrals.map(({ orders,midpoint,inside,tail,full }) => ({
      orders, midpoint: wire(midpoint), boxIntegral: wire(inside), outsideAbsoluteUpper: text(tail), fullIntegral: wire(full),
    })),
  };
}

export type ScalarGibbsResult = ReturnType<typeof calculateScalarGibbs>;
