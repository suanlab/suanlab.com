/** Radii in units of central mass, and outward velocity in units of light speed. */
export function calculateSchwarzschildRedshift(emitter: number, receiver: number, beta: number) {
  if (![emitter, receiver, beta].every(Number.isFinite) || emitter <= 2 || receiver < emitter || Math.abs(beta) >= 1) {
    throw new RangeError('Expected ordered exterior radii and subluminal emitter speed');
  }
  const emissionLapse = 1 - 2 / emitter;
  const receptionLapse = 1 - 2 / receiver;
  const gravitationalFactor = Math.sqrt(emissionLapse / receptionLapse);
  const dopplerFactor = Math.sqrt((1 + beta) / (1 - beta));
  const ratio = gravitationalFactor * dopplerFactor;
  return {
    gravitationalFactor, dopplerFactor, ratio, redshift: 1 / ratio - 1,
    cancellationSpeed: (receptionLapse - emissionLapse) / (receptionLapse + emissionLapse),
  };
}
