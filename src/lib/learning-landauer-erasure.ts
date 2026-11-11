const boltzmannConstant = 1.380649e-23;

function binaryEntropy(p: number) {
  return p === 0 || p === 1 ? 0 : -p * Math.log(p) - (1 - p) * Math.log1p(-p);
}

/** Entropy balance for a degenerate bit, cyclic control and one equilibrium bath. */
export function calculateLandauerErasure(p: number, error: number, temperature: number) {
  if (![p, error, temperature].every(Number.isFinite) || p < 0 || p > 1 || error < 0 || error > .5 || temperature <= 0) {
    throw new RangeError('Expected an input probability, error in [0, 0.5], and positive temperature');
  }
  const initialEntropy = binaryEntropy(p);
  const finalEntropy = binaryEntropy(error);
  return {
    initialEntropy,
    finalEntropy,
    entropyDecrease: initialEntropy - finalEntropy,
    heatFloor: boltzmannConstant * temperature * (initialEntropy - finalEntropy),
  };
}
