import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fitOneCoolingReading, fitTwoCoolingReadings, predictCooling } from '../src/lib/learning-cooling-inverse';

const close = (a: number, b: number) => assert(Math.abs(a - b) < 1e-10, `${a} vs ${b}`);

test('a family fits one observation and two noiseless readings recover the true parameters', () => {
  const first = predictCooling(40, .2, 2), second = predictCooling(40, .2, 6);
  for (const rate of [.05, .2, .4]) {
    const fit = fitOneCoolingReading(first, 2, rate);
    close(predictCooling(fit.initial, fit.rate, 2), first);
  }
  const fit = fitTwoCoolingReadings(first, second, 2, 6);
  assert(fit);
  close(fit.initial, 40);
  close(fit.rate, .2);
});

test('nearby observation times amplify perturbations, and incompatible readings are reported', () => {
  const first = predictCooling(40, .2, 2);
  const early = fitTwoCoolingReadings(first, predictCooling(40, .2, 2.25) + .5, 2, 2.25);
  const late = fitTwoCoolingReadings(first, predictCooling(40, .2, 6) + .5, 2, 6);
  assert(early && late);
  assert(Math.abs(early.rate - .2) > Math.abs(late.rate - .2));
  assert.equal(fitTwoCoolingReadings(first, 0, 2, 6), null);
  assert.equal(fitTwoCoolingReadings(first, first, 2, 6), null);
  assert.throws(() => fitTwoCoolingReadings(first, first, 2, 2), RangeError);
});
