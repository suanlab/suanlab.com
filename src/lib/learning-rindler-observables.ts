/** Accelerated coordinates on the right wedge of flat spacetime, in units with c=1. */
export function calculateRindlerReadings(rho: number, eta: number) {
  if (!Number.isFinite(rho) || rho <= 0 || !Number.isFinite(eta) || eta < 0 || eta > 6) {
    throw new RangeError('Expected a positive radius and nonnegative elapsed accelerated time');
  }
  return {
    time: rho * Math.sinh(eta),
    position: rho * Math.cosh(eta),
    acceleratedProperTime: rho * eta,
    inertialProperTime: 2 * rho * Math.sinh(eta / 2),
    acceleration: 1 / rho,
    coordinateLightSpeed: rho,
    measuredLightSpeed: 1,
  };
}

export function readRindlerCoordinates(time: number, position: number) {
  if (!Number.isFinite(time) || !Number.isFinite(position) || position <= Math.abs(time)) {
    throw new RangeError('Expected an event inside the right wedge');
  }
  return { rho: Math.sqrt((position - time) * (position + time)), eta: Math.atanh(time / position) };
}
