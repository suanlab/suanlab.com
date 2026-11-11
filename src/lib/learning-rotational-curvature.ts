export type RotationalSurface = 'plane' | 'sphere' | 'hyperbolic';

export function calculateRotationalCurvature(surface: RotationalSurface, scale: number, radialRatio: number) {
  if (!['plane', 'sphere', 'hyperbolic'].includes(surface) || !Number.isFinite(scale) || scale <= 0 ||
      !Number.isFinite(radialRatio) || radialRatio <= 0 || radialRatio > Math.PI / 2) {
    throw new RangeError('Expected a supported surface and positive radius up to πR/2');
  }
  const radius = scale * radialRatio;
  const f = surface === 'plane' ? radius
    : scale * (surface === 'sphere' ? Math.sin(radialRatio) : Math.sinh(radialRatio));
  const derivative = surface === 'plane' ? 1 : surface === 'sphere' ? Math.cos(radialRatio) : Math.cosh(radialRatio);
  const curvature = surface === 'plane' ? 0 : (surface === 'sphere' ? 1 : -1) / scale ** 2;
  const area = surface === 'plane' ? Math.PI * radius ** 2
    : 4 * Math.PI * scale ** 2 * (surface === 'sphere' ? Math.sin(radialRatio / 2) : Math.sinh(radialRatio / 2)) ** 2;
  return {
    radius, circumference: 2 * Math.PI * f, area, curvature,
    radialConnection: -f * derivative, angularConnection: derivative / f,
    frameAngleChange: -2 * Math.PI * derivative,
    holonomy: curvature * area,
  };
}
