/** Ideal real-amplitude rotation, attenuation, and inverse rotation. */
export function calculatePhotonicProduct(x1: number, x2: number, theta: number, transmission: number) {
  if (![x1, x2, theta, transmission].every(Number.isFinite) || transmission < 0 || transmission > 1) {
    throw new RangeError('Expected finite amplitudes and angle, and a passive transmission in [0, 1]');
  }
  const c = Math.cos(theta), s = Math.sin(theta);
  const internal1 = c * x1 + s * x2;
  const internal2 = -s * x1 + c * x2;
  const y1 = c * internal1 - s * transmission * internal2;
  const y2 = s * internal1 + c * transmission * internal2;
  return {
    y1,
    y2,
    power1: y1 ** 2,
    power2: y2 ** 2,
    inputPower: x1 ** 2 + x2 ** 2,
    dissipatedPower: (1 - transmission ** 2) * internal2 ** 2,
  };
}
