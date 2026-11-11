export function greatCircleGeodesic(radius: number, separation: number, tilt: number, longArc: boolean, fraction: number) {
  if (![radius, separation, tilt, fraction].every(Number.isFinite) || radius <= 0 ||
      separation <= 0 || separation > Math.PI || fraction < 0 || fraction > 1) {
    throw new RangeError('Expected a positive radius, separation up to π, and a fraction between zero and one');
  }

  const signedSweep = longArc ? separation - 2 * Math.PI : separation;
  const angle = signedSweep * fraction;
  const transverse = Math.sin(angle);
  const point: [number, number, number] = [
    radius * Math.cos(angle),
    radius * transverse * Math.cos(tilt),
    radius * transverse * Math.sin(tilt),
  ];
  return {
    point,
    length: Math.abs(signedSweep) * radius,
    shortestLength: separation * radius,
    antipodal: separation === Math.PI,
  };
}

export function polarStraightLine(parameter: number) {
  const radius = Math.hypot(1, parameter);
  const angularVelocity = 1 / radius ** 2;
  const radialAcceleration = 1 / radius ** 3;
  return {
    radius,
    angle: Math.atan(parameter),
    angularVelocity,
    radialAcceleration,
    connectionCorrection: radius * angularVelocity ** 2,
  };
}
