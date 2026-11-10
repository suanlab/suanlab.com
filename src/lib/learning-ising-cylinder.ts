export interface IsingCylinderInput {
  length: number;
  horizontalCoupling: number;
  verticalCoupling: number;
}

/** Square cylinder with periodic horizontal bonds and open vertical ends. */
export function calculateIsingCylinder(input: IsingCylinderInput) {
  const { length, horizontalCoupling, verticalCoupling } = input;
  if (!Number.isInteger(length) || length < 3 || length > 8 ||
      ![horizontalCoupling, verticalCoupling].every(k => Number.isFinite(k) && k >= 0 && k <= 1.25)) {
    throw new RangeError('Expected side length 3 to 8 and couplings between 0 and 1.25');
  }

  const states = 1 << length;
  const mask = states - 1;
  const orderCount = 5;
  const popcounts = new Uint8Array(states);
  for (let row = 1; row < states; row++) {
    popcounts[row] = popcounts[row >> 1] + (row & 1);
  }
  const magnetizations = Array.from(popcounts, count => 2 * count - length);
  const verticalWeights = Array.from({ length: length + 1 }, (_, k) => Math.exp(-2 * verticalCoupling * k));
  const horizontalWeights = Array.from(popcounts, (_, row) => {
    const rotated = ((row << 1) & mask) | (row >> (length - 1));
    return Math.exp(-2 * horizontalCoupling * popcounts[row ^ rotated]);
  });

  let moments = new Float64Array(states * orderCount);
  let next = new Float64Array(moments.length);
  for (let row = 0; row < states; row++) {
    for (let order = 0; order < orderCount; order++) {
      moments[row * orderCount + order] = horizontalWeights[row] * magnetizations[row] ** order;
    }
  }

  for (let layer = 1; layer < length; layer++) {
    for (let row = 0; row < states; row++) {
      let mass = 0, first = 0, second = 0, third = 0, fourth = 0;
      for (let previous = 0; previous < states; previous++) {
        const weight = verticalWeights[popcounts[previous ^ row]];
        const offset = previous * orderCount;
        mass += weight * moments[offset];
        first += weight * moments[offset + 1];
        second += weight * moments[offset + 2];
        third += weight * moments[offset + 3];
        fourth += weight * moments[offset + 4];
      }
      // Expand powers of the magnetization after adding the new row.
      const offset = row * orderCount, m = magnetizations[row], m2 = m * m, w = horizontalWeights[row];
      next[offset] = w * mass;
      next[offset + 1] = w * (first + m * mass);
      next[offset + 2] = w * (second + 2 * m * first + m2 * mass);
      next[offset + 3] = w * (third + 3 * m * second + 3 * m2 * first + m2 * m * mass);
      next[offset + 4] = w * (fourth + 4 * m * third + 6 * m2 * second + 4 * m2 * m * first + m2 * m2 * mass);
    }
    [moments, next] = [next, moments];
  }

  let mass = 0, second = 0, fourth = 0;
  for (let row = 0; row < states; row++) {
    const offset = row * orderCount;
    mass += moments[offset];
    second += moments[offset + 2];
    fourth += moments[offset + 4];
  }
  const sites = length ** 2;
  return {
    meanMagnetization: 0,
    secondMoment: second / mass,
    fourthMoment: fourth / mass,
    susceptibilityPerSite: second / (mass * sites),
    binderRatio: 1 - fourth * mass / (3 * second ** 2),
    logPartition: Math.log(mass) + horizontalCoupling * sites + verticalCoupling * length * (length - 1),
  };
}

const lengths = [3, 4, 6, 8] as const;

export function calculateIsingSweep(ratio: number) {
  if (ratio !== .5 && ratio !== 1 && ratio !== 1.5) throw new RangeError('Unsupported coupling ratio');

  const criticalEquation = (kx: number) => Math.sinh(2 * kx) * Math.sinh(2 * ratio * kx) - 1;
  const critical = bisectSignChange(criticalEquation, 0, .8);
  const curves = lengths.map(length => {
    const observe = (kx: number) => calculateIsingCylinder({ length, horizontalCoupling: kx, verticalCoupling: ratio * kx });
    const samples = Array.from({ length: 41 }, (_, index) => {
      const kx = index / 50;
      const { binderRatio, susceptibilityPerSite } = observe(kx);
      return { kx, binderRatio, susceptibilityPerSite };
    });
    return { length, samples, observe };
  });

  const crossings = ([0, 2] as const).map(index => {
    const left = curves[index], right = curves[index + 1];
    const bracketIndex = left.samples.findIndex((sample, i) =>
      i > 0 && sample.binderRatio <= right.samples[i].binderRatio &&
      left.samples[i - 1].binderRatio > right.samples[i - 1].binderRatio);
    const kx = bracketIndex < 0 ? null : bisectSignChange(
      k => left.observe(k).binderRatio - right.observe(k).binderRatio,
      left.samples[bracketIndex - 1].kx, left.samples[bracketIndex].kx,
    );
    return { lengths: [left.length, right.length], kx };
  });

  return { ratio, critical, curves: curves.map(({ length, samples }) => ({ length, samples })), crossings };
}

function bisectSignChange(evaluate: (x: number) => number, lower: number, upper: number) {
  const sign = Math.sign(evaluate(lower));
  for (let i = 0; i < 25; i++) {
    const mid = (lower + upper) / 2;
    if (Math.sign(evaluate(mid)) === sign) lower = mid;
    else upper = mid;
  }
  return (lower + upper) / 2;
}
