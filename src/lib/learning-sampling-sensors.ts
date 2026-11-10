/** Calibrated rectangular-pixel sampling and an explicit Poisson/Gaussian camera.
 * Pure browser-safe models. SI inputs; no import of optical relative intensity.
 * ADC distributions are finite Poisson sums with a disclosed Chernoff tail bound,
 * not Monte Carlo samples or a silently renormalized probability window.
 */
export const sensorPlanckConstant = 6.62607015e-34;
export const sensorLightSpeed = 299792458;
const tau = 2 * Math.PI;
function bound(name: string, value: number, lo: number, hi: number) {
  if (!Number.isFinite(value) || value < lo || value > hi) throw new RangeError(`${name} must be finite in [${lo}, ${hi}]`);
}
function integer(name: string, value: number, lo: number, hi: number) {
  bound(name, value, lo, hi);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer`);
}
/** sin(pi*x)/(pi*x); integer zeros are exact, without snapping neighbours. */
export function pixelSinc(x: number) {
  if (!Number.isFinite(x)) throw new RangeError('sinc argument must be finite');
  if (x === 0) return 1;
  if (Number.isInteger(x)) return 0;
  const z = Math.PI * x;
  return Math.abs(z) < 1e-4 ? 1 - z * z / 6 + z ** 4 / 120 : Math.sin(z) / z;
}
function cosineCycles(cycles: number) {
  const q = 4 * cycles;
  if (Number.isInteger(q)) return [1, 0, -1, 0][((q % 4) + 4) % 4];
  return Math.cos(tau * cycles);
}
export interface PixelSamplingInput {
  pitch: number; apertureWidth: number; pixelHeight: number; wavelength: number;
  irradiance: number; modulation: number; frequency: number; phase: number;
  velocity: number; exposure: number; centerTime: number; origin: number;
  radius: number; queryCoordinate: number;
}
export function calibratedPhotonExposure(irradiance: number, area: number, exposure: number, wavelength: number) {
  bound('irradiance W/m²', irradiance, 0, 1e6); bound('area m²', area, 0, 1);
  bound('exposure s', exposure, 0, 100); bound('wavelength m', wavelength, 200e-9, 2e-6);
  const energy = irradiance * area * exposure;
  const photonEnergy = sensorPlanckConstant * sensorLightSpeed / wavelength;
  return { energy, photonEnergy, incidentPhotonMean: energy / photonEnergy };
}
export function pixelApertureSampling(input: PixelSamplingInput) {
  const s = { ...input };
  bound('pitch m', s.pitch, 1e-6, 100e-6); bound('apertureWidth m', s.apertureWidth, 0, s.pitch);
  bound('pixelHeight m', s.pixelHeight, 1e-6, 100e-6); bound('wavelength m', s.wavelength, 200e-9, 2e-6);
  bound('irradiance W/m²', s.irradiance, 0, 10); bound('modulation', s.modulation, 0, 1);
  bound('frequency cycles/m', s.frequency, 0, 4 / s.pitch); bound('phase rad', s.phase, -Math.PI, Math.PI);
  bound('velocity m/s', s.velocity, -.1, .1); bound('exposure s', s.exposure, 0, 1);
  bound('centerTime s', s.centerTime, 0, 1); bound('origin m', s.origin, -s.pitch, s.pitch);
  integer('radius', s.radius, 1, 32); bound('queryCoordinate m', s.queryCoordinate, -40 * s.pitch, 40 * s.pitch);
  const spatialTransfer = pixelSinc(s.frequency * s.apertureWidth);
  const temporalTransfer = pixelSinc(s.frequency * s.velocity * s.exposure);
  const transfer = spatialTransfer * temporalTransfer;
  const samplingFrequency = 1 / s.pitch;
  const aliasOrder = Math.floor(s.frequency * s.pitch + .5);
  const aliasFrequency = (s.frequency * s.pitch - aliasOrder) / s.pitch;
  const basePhaseCycles = s.phase / tau - s.frequency * s.velocity * s.centerTime;
  const aliasPhaseCycles = basePhaseCycles + aliasOrder * s.origin / s.pitch;
  const pointLimit = (x: number) => s.irradiance * (1 + s.modulation * cosineCycles(s.frequency * x + basePhaseCycles));
  const boxLimit = (x: number) => s.irradiance * (1 + s.modulation * transfer * cosineCycles(s.frequency * x + basePhaseCycles));
  const aliasLimit = (x: number) => s.irradiance * (1 + s.modulation * transfer * cosineCycles(aliasFrequency * x + aliasPhaseCycles));
  const area = s.apertureWidth * s.pixelHeight;
  const samples = Array.from({ length: 2 * s.radius + 1 }, (_, i) => {
    const index = i - s.radius, coordinate = s.origin + index * s.pitch;
    const limitAverage = boxLimit(coordinate);
    const physical = calibratedPhotonExposure(limitAverage, area, s.exposure, s.wavelength);
    return { index, coordinate, pointIrradiance: pointLimit(coordinate), limitAverage,
      averageIrradiance: area > 0 && s.exposure > 0 ? limitAverage : null,
      aliasAverage: aliasLimit(coordinate), ...physical };
  });
  function observe(coordinate: number) {
    bound('observation coordinate m', coordinate, -40 * s.pitch, 40 * s.pitch);
    const finiteReconstruction = samples.reduce((sum, q) => sum + q.limitAverage * pixelSinc((coordinate - s.origin) / s.pitch - q.index), 0);
    return { coordinate, pointIrradiance: pointLimit(coordinate), limitAverage: boxLimit(coordinate),
      aliasAverage: aliasLimit(coordinate), finiteReconstruction };
  }
  const span = (s.radius + 1) * s.pitch;
  // At least 24 plotted samples per highest original-frequency cycle.
  const intervals = Math.max(256, Math.ceil(48 * span * s.frequency));
  const profile = Array.from({ length: intervals + 1 }, (_, i) => observe(-span + 2 * span * i / intervals));
  return { spatialTransfer, temporalTransfer, transfer, area, samplingFrequency, nyquistFrequency: samplingFrequency / 2,
    aliasFrequency, aliasOrder, aliasPhaseCycles, strictlyBelowNyquist: s.frequency < samplingFrequency / 2,
    apertureAverageDefined: area > 0 && s.exposure > 0, samples: samples.map(q => ({ ...q })), profile, query: observe(s.queryCoordinate), observe };
}
export interface PhotonSensorInput {
  incidentPhotonMean: number; quantumEfficiency: number; exposure: number; darkRate: number;
  readSigma: number; gain: number; offset: number; bits: number; fullWell: number;
}
const nodes = [.09501250983763744, .2816035507792589, .4580167776572274, .6178762444026438, .755404408355003, .8656312023878318, .9445750230732326, .9894009349916499];
const weights = [.1894506104550685, .1826034150449236, .16915651939500254, .14959598881657673, .12462897125553387, .09515851168249278, .06225352393864789, .027152459411754095];
/** Standard normal CDF via 16-node Gaussian quadrature on panels <=2.
 * Beyond ±9 the absolute discarded Gaussian tail is below 1.13e-19.
 */
export function sensorNormalCdf(z: number) {
  if (typeof z !== 'number' || Number.isNaN(z)) throw new RangeError('normal argument cannot be NaN');
  if (z >= 9) return 1;
  if (z <= -9) return 0;
  const a = Math.abs(z), lower = a < 1 ? 0 : a, upper = a < 1 ? a : 9;
  const panels = Math.max(1, Math.ceil((upper - lower) / 2)); let integral = 0;
  for (let j = 0; j < panels; j++) {
    const half = (upper - lower) / (2 * panels), mid = lower + (j + .5) * (upper - lower) / panels;
    for (let i = 0; i < nodes.length; i++) integral += half * weights[i] * (Math.exp(-.5 * (mid - half * nodes[i]) ** 2) + Math.exp(-.5 * (mid + half * nodes[i]) ** 2)) / Math.sqrt(tau);
  }
  if (a < 1) return .5 + (z < 0 ? -integral : integral);
  return z < 0 ? integral : 1 - integral;
}
function logFactorial(n: number) {
  if (n < 16) { let v = 0; for (let k = 2; k <= n; k++) v += Math.log(k); return v; }
  return (n + .5) * Math.log(n) - n + .5 * Math.log(tau) + 1 / (12 * n) - 1 / (360 * n ** 3) + 1 / (1260 * n ** 5) - 1 / (1680 * n ** 7);
}
function chernoff(mean: number, k: number) { if (k === 0) return Math.exp(-mean); return Math.exp(-mean + k * (1 + Math.log(mean / k))); }
export function sensorPoissonDistribution(mean: number) {
  bound('Poisson mean electrons', mean, 0, 4224);
  if (mean === 0) return { values: [{ count: 0, probability: 1 }], lower: 0, upper: 0, omittedProbabilityBound: 0, coveredProbability: 1 };
  const radius = 12 * Math.sqrt(mean) + 32;
  const lower = Math.max(0, Math.floor(mean - radius)), upper = Math.ceil(mean + radius), mode = Math.floor(mean);
  const probabilities = new Array<number>(upper - lower + 1);
  probabilities[mode - lower] = Math.exp(-mean + mode * Math.log(mean) - logFactorial(mode));
  for (let n = mode; n > lower; n--) probabilities[n - 1 - lower] = probabilities[n - lower] * n / mean;
  for (let n = mode; n < upper; n++) probabilities[n + 1 - lower] = probabilities[n - lower] * mean / (n + 1);
  const omittedProbabilityBound = (lower ? chernoff(mean, lower - 1) : 0) + chernoff(mean, upper + 1);
  const values = probabilities.map((probability, i) => ({ count: i + lower, probability }));
  return { values, lower, upper, omittedProbabilityBound, coveredProbability: probabilities.reduce((a, b) => a + b, 0) };
}
/** floor(y+.5) followed by clipping to unsigned [0,2^bits-1]. */
export function sensorQuantize(value: number, bits: number) {
  bound('analog value DN', value, -1e12, 1e12); integer('bits', bits, 2, 8);
  return Math.min(2 ** bits - 1, Math.max(0, Math.floor(value + .5)));
}
export function photonCountingSensor(input: PhotonSensorInput) {
  const s = { ...input };
  bound('incidentPhotonMean', s.incidentPhotonMean, 0, 4096); bound('quantumEfficiency', s.quantumEfficiency, 0, 1);
  bound('exposure s', s.exposure, 0, 1); bound('darkRate electrons/s', s.darkRate, 0, 128);
  bound('readSigma electrons', s.readSigma, 0, 16); bound('gain DN/electron', s.gain, .05, 8);
  bound('offset DN', s.offset, -16, 256); integer('bits', s.bits, 2, 8); integer('fullWell electrons', s.fullWell, 1, 8192);
  if (s.exposure === 0 && s.incidentPhotonMean !== 0) throw new RangeError('Zero exposure requires zero incidentPhotonMean');
  const photoMean = s.quantumEfficiency * s.incidentPhotonMean, darkMean = s.darkRate * s.exposure;
  const electronMean = photoMean + darkMean, electronVariance = electronMean + s.readSigma ** 2;
  const poisson = sensorPoissonDistribution(electronMean), maxCode = 2 ** s.bits - 1, sigmaDN = s.gain * s.readSigma;
  const bins = Array.from({ length: maxCode + 1 }, (_, code) => ({ code, probability: 0 }));
  let fullWellOverflowProbability = 0, adcLowerClippingProbability = 0, adcUpperClippingProbability = 0, storedMean = 0;
  for (const q of poisson.values) {
    const stored = Math.min(q.count, s.fullWell), mu = s.offset + s.gain * stored;
    storedMean += q.probability * stored;
    if (q.count > s.fullWell) fullWellOverflowProbability += q.probability;
    if (sigmaDN === 0) {
      bins[sensorQuantize(mu, s.bits)].probability += q.probability;
      if (mu < -.5) adcLowerClippingProbability += q.probability;
      if (mu >= maxCode + .5) adcUpperClippingProbability += q.probability;
    } else {
      const cdf = (v: number) => sensorNormalCdf((v - mu) / sigmaDN);
      let previous = 0;
      for (let code = 0; code <= maxCode; code++) {
        const next = code === maxCode ? 1 : cdf(code + .5);
        const probability = next - previous;
        if (probability < -1e-14) throw new Error('Nonmonotone numerical Gaussian CDF');
        bins[code].probability += q.probability * probability;
        previous = next;
      }
      adcLowerClippingProbability += q.probability * cdf(-.5);
      adcUpperClippingProbability += q.probability * (1 - cdf(maxCode + .5));
    }
  }
  const storedVariance = poisson.values.reduce((sum, q) => sum + q.probability * (Math.min(q.count, s.fullWell) - storedMean) ** 2, 0);
  const adcMean = bins.reduce((sum, q) => sum + q.code * q.probability, 0);
  const adcVariance = bins.reduce((sum, q) => sum + (q.code - adcMean) ** 2 * q.probability, 0);
  return { photoMean, darkMean, electronMean, electronVariance, poisson, maxCode,
    unsaturatedAnalogMean: s.offset + s.gain * electronMean,
    unsaturatedAnalogVariance: s.gain ** 2 * electronVariance,
    unsaturatedSignalToNoise: electronVariance > 0 ? photoMean / Math.sqrt(electronVariance) : null,
    storedMean, storedVariance, fullWellAnalogMean: s.offset + s.gain * storedMean,
    fullWellAnalogVariance: s.gain ** 2 * (storedVariance + s.readSigma ** 2),
    fullWellOverflowProbability, adcLowerClippingProbability, adcUpperClippingProbability,
    lowerRailOccupancy: bins[0].probability, upperRailOccupancy: bins[maxCode].probability,
    bins, adcMean, adcVariance, coveredProbability: bins.reduce((sum, q) => sum + q.probability, 0),
    omittedProbabilityBound: poisson.omittedProbabilityBound,
    adcMeanTailErrorBound: maxCode * poisson.omittedProbabilityBound,
    adcSecondMomentTailErrorBound: maxCode ** 2 * poisson.omittedProbabilityBound,
    probabilityMassResidual: poisson.coveredProbability - 1,
    quantizationVarianceApproximation: 1 / 12 };
}
