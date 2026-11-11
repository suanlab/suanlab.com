import assert from 'node:assert/strict';
import { test } from 'node:test';
import { greatCircleGeodesic, polarStraightLine } from '../src/lib/learning-great-circle-geodesic';

const close = (a: number, b: number, tolerance = 1e-10) => assert(Math.abs(a - b) <= tolerance, `${a} vs ${b}`);

test('two great-circle arcs share endpoints but have different lengths', () => {
  const radius = 2, theta = .9, tilt = .7;
  const short = greatCircleGeodesic(radius, theta, tilt, false, 1);
  const long = greatCircleGeodesic(radius, theta, tilt, true, 1);
  short.point.forEach((p, i) => close(p, long.point[i]));
  close(short.length, 1.8);
  close(short.length + long.length, 2 * Math.PI * radius);
  assert(long.length > short.length);
});

test('antipodal endpoints allow distinct paths with the same length', () => {
  const a = greatCircleGeodesic(1, Math.PI, 0, false, .5);
  const b = greatCircleGeodesic(1, Math.PI, Math.PI / 2, false, .5);
  close(a.point[1], 1);
  close(b.point[2], 1);
  close(a.length, b.length);
  assert(a.antipodal && b.antipodal);
});

test('a Cartesian straight line has nonzero polar acceleration and zero covariant acceleration', () => {
  const point = polarStraightLine(1);
  close(point.radialAcceleration, 1 / (2 * Math.sqrt(2)));
  close(point.radialAcceleration, point.connectionCorrection);
  close(point.radius * Math.cos(point.angle), 1);
  close(point.radius * Math.sin(point.angle), 1);
});
