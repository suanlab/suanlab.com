const logWeight = (x: number) => Math.abs(x) + Math.log1p(Math.exp(-2 * Math.abs(x)));

function logRingPartition(sites: number, coupling: number) {
  return sites * logWeight(coupling) + Math.log1p(Math.tanh(coupling) ** sites);
}

export function calculateIsingDecimation(initial: number, length: 8 | 16 | 32) {
  if (!Number.isFinite(initial) || initial < 0 || initial > 4 || ![8, 16, 32].includes(length)) {
    throw new RangeError('Expected coupling between 0 and 4 and ring size 8, 16 or 32');
  }
  let coupling = initial;
  let normalization = 0;
  let spacing = 1;
  const steps = [];
  for (let sites: number = length; sites >= 2; sites /= 2) {
    const correlationLength = coupling === 0 ? 0 : -1 / Math.log(Math.tanh(coupling));
    steps.push({
      sites, coupling, spacing, normalization, correlationLength,
      physicalCorrelationLength: spacing * correlationLength,
      reconstructedLogPartition: normalization + logRingPartition(sites, coupling),
    });
    const nextCoupling = Math.log1p(2 * Math.sinh(coupling) ** 2) / 2;
    normalization += (sites / 2) * (Math.LN2 + nextCoupling);
    coupling = nextCoupling;
    spacing *= 2;
  }
  return steps;
}

export function calculateFourSpinMarginal(coupling: number) {
  if (!Number.isFinite(coupling) || coupling < 0 || coupling > 4) throw new RangeError('Unsupported star coupling');
  const states = Array.from({ length: 16 }, (_, code) => {
    const spins: number[] = Array.from({ length: 4 }, (_, i) => code & (1 << i) ? 1 : -1);
    return {
      spins,
      product: spins.reduce((total, spin) => total * spin, 1),
      logWeight: logWeight(coupling * spins.reduce((total, spin) => total + spin, 0)),
    };
  });
  const average = (select: (state: typeof states[number]) => number) =>
    states.reduce((total, state) => total + select(state), 0) / states.length;
  const fourSpin = average(state => state.product * state.logWeight);
  const pair = average(state => state.spins[0] * state.spins[1] * state.logWeight);
  const constant = average(state => state.logWeight);
  const weights = states.map(state => Math.exp(state.logWeight));
  const truncated = states.map(state => Math.exp(state.logWeight - state.product * fourSpin));
  const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
  const partition = sum(weights), truncatedPartition = sum(truncated);
  const distribution = states.map(({ spins }, i) => ({
    spins,
    exact: weights[i] / partition,
    truncated: truncated[i] / truncatedPartition,
  }));
  const totalVariation = sum(distribution.map(row => Math.abs(row.exact - row.truncated))) / 2;
  return { constant, pair, fourSpin, distribution, totalVariation };
}
