import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateMemoryRoofline } from '../src/lib/learning-memory-roofline';

test('reduced memory traffic moves a fixed workload across the ridge', () => {
  const model = (bytes: number) => calculateMemoryRoofline(1e9, bytes, 8e12, 200e9);
  const original = model(100e6), tiled = model(20e6);
  assert.equal(original.ridge, 40);
  assert.equal(original.bottleneck, 'memory');
  assert.equal(original.throughputCeiling, 2e12);
  assert.equal(original.runtimeFloor, .5e-3);
  assert.equal(tiled.bottleneck, 'compute');
  assert.equal(tiled.runtimeFloor, .125e-3);
  assert.equal(model(25e6).bottleneck, 'balanced');
});

test('runtime and throughput bounds agree across several workloads', () => {
  for (const bytes of [1, 40, 200]) {
    const operations = 1280;
    const result = calculateMemoryRoofline(operations, bytes, 12e12, 90e9);
    assert(Math.abs((operations / result.runtimeFloor) / result.throughputCeiling - 1) < 1e-12);
    assert(result.runtimeFloor >= result.computeTime);
    assert(result.runtimeFloor >= result.transferTime);
  }
  assert.throws(() => calculateMemoryRoofline(1280, 0, 12e12, 90e9), RangeError);
});
