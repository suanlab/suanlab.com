/** Dispersive packets: exact quadratic-envelope evolution and a real periodic
 * nearest-neighbor chain. SI lengths/times; squared envelope norm is not energy.
 * Pure browser-safe mathematics: no registry, filesystem or plotting imports.
 */
export type PacketComplex = [number, number];
export interface GaussianPacket {
  amplitude: number; sigma: number; center: number; carrierWaveNumber: number;
  carrierFrequency: number; groupVelocity: number; dispersion: number; phase: number;
}
export interface LatticePacket {
  particles: 32 | 64; length: number; density: number; tension: number;
  amplitude: number; modeCenter: number; bandwidth: number; center: number; phase: number;
}
function bounded(x: number, min: number, max: number, name: string) {
  if (!Number.isFinite(x) || x < min || x > max) throw new RangeError(`${name} outside supported range.`);
}
function gaussianData(s: GaussianPacket) {
  bounded(s.amplitude, 0, 2, 'Amplitude'); bounded(s.sigma, .25, 4, 'Initial envelope width');
  bounded(s.center, -8, 8, 'Initial center'); bounded(s.carrierWaveNumber, -16, 16, 'Carrier wave number');
  bounded(s.carrierFrequency, -32, 32, 'Carrier angular frequency');
  bounded(s.groupVelocity, -4, 4, 'Group velocity'); bounded(s.dispersion, -4, 4, 'Quadratic dispersion');
  bounded(s.phase, -4 * Math.PI, 4 * Math.PI, 'Initial phase');
}

/** Width parameter sigma describes the AMPLITUDE Gaussian, not intensity SD.
 * k0=0 has no carrier phase velocity; a zero field has no observed local phase.
 */
export function gaussianDispersivePacket(s: GaussianPacket, x: number, t: number) {
  gaussianData(s); bounded(x, -128, 128, 'Observation position'); bounded(t, -4, 4, 'Time');
  const r = s.sigma ** 2, b = s.dispersion * t, denominator = r * r + b * b;
  const center = s.center + s.groupVelocity * t, y = x - center, chirp = b / denominator;
  const spread = Math.hypot(1, b / r), width = s.sigma * spread;
  const envelope = s.amplitude / Math.sqrt(spread) * Math.exp(-y * y * r / (2 * denominator));
  const envelopePhase = -Math.atan2(b, r) / 2 + y * y * chirp / 2;
  const phase = s.carrierWaveNumber * x - s.carrierFrequency * t + s.phase + envelopePhase;
  const field: PacketComplex = [envelope * Math.cos(phase), envelope * Math.sin(phase)];
  const localWaveNumber = s.carrierWaveNumber + y * chirp;
  const localAngularFrequency = s.carrierFrequency + s.dispersion * r / (2 * denominator)
    + s.groupVelocity * y * chirp - s.dispersion * y * y * (r * r - b * b) / (2 * denominator ** 2);
  const carrierRatio = s.carrierWaveNumber === 0 ? null : s.carrierFrequency / s.carrierWaveNumber;
  const carrierPhaseVelocity = carrierRatio !== null && Number.isFinite(carrierRatio) ? carrierRatio : null;
  const carrierPhaseVelocityStatus = s.carrierWaveNumber === 0 ? 'zero-carrier-wave-number' : carrierPhaseVelocity === null ? 'unrepresentable-ratio' : 'finite';
  const crestRatio = envelope === 0 || localWaveNumber === 0 ? null : localAngularFrequency / localWaveNumber;
  const localCrestVelocity = crestRatio !== null && Number.isFinite(crestRatio) ? crestRatio : null;
  const localCrestVelocityStatus = envelope === 0 ? 'zero-field' : localWaveNumber === 0 ? 'zero-phase-gradient' : localCrestVelocity === null ? 'unrepresentable-ratio' : 'finite';
  // Logarithmic envelope derivatives, with the carrier phase factored out.
  const invD: PacketComplex = [r / denominator, -b / denominator];
  const invD2: PacketComplex = [invD[0] ** 2 - invD[1] ** 2, 2 * invD[0] * invD[1]];
  const logX: PacketComplex = [-y * invD[0], -y * invD[1]];
  const logXX: PacketComplex = [y * y * invD2[0] - invD[0], y * y * invD2[1] - invD[1]];
  const logT: PacketComplex = [s.dispersion * invD[1] / 2 + s.groupVelocity * y * invD[0] - s.dispersion * y * y * invD2[1] / 2,
    -s.dispersion * invD[0] / 2 + s.groupVelocity * y * invD[1] + s.dispersion * y * y * invD2[0] / 2];
  const multiply = (v: PacketComplex): PacketComplex => [field[0] * v[0] - field[1] * v[1], field[0] * v[1] + field[1] * v[0]];
  return { x, time: t, center, envelope, field, realField: field[0], intensity: envelope ** 2,
    amplitudeWidth: width, intensityVariance: width ** 2 / 2,
    intensityStandardDeviation: width / Math.sqrt(2), integratedNorm: s.amplitude ** 2 * s.sigma * Math.sqrt(Math.PI),
    intensityCenter: s.amplitude === 0 ? null : center,
    carrierPhaseVelocity, carrierPhaseVelocityStatus,
    groupVelocity: s.groupVelocity, chirp, phase,
    observedPhase: envelope === 0 ? null : phase,
    localWaveNumber: envelope === 0 ? null : localWaveNumber,
    localAngularFrequency: envelope === 0 ? null : localAngularFrequency,
    localCrestVelocity, localCrestVelocityStatus,
    // All three derivatives have the SAME full carrier factor; they are not
    // derivatives of the full field, only of its envelope times that factor.
    envelopeDerivativeX: multiply(logX), envelopeDerivativeXX: multiply(logXX), envelopeDerivativeT: multiply(logT),
    model: 'exact-quadratic-envelope' as const, normMeaning: 'integrated-squared-envelope-not-mechanical-energy' as const };
}

/** First circular moment of nonnegative weights on a periodic domain. No
 * arbitrary angle is returned for a uniform/zero distribution. Not an unwrap.
 */
export function periodicPacketCenter(weights: number[], positions: number[], length: number) {
  bounded(length, 4, 16, 'Length');
  if (!weights.length || weights.length !== positions.length) throw new RangeError('Matching nonempty center arrays required.');
  let sum = 0, re = 0, im = 0;
  for (let i = 0; i < weights.length; i++) {
    bounded(weights[i], 0, 1e12, 'Center weight'); bounded(positions[i], 0, length, 'Center position');
    const angle = 2 * Math.PI * positions[i] / length;
    sum += weights[i]; re += weights[i] * Math.cos(angle); im += weights[i] * Math.sin(angle);
  }
  const resultant = sum === 0 ? 0 : Math.hypot(re, im) / sum;
  const position = resultant < 1e-10 ? null : ((Math.atan2(im, re) / (2 * Math.PI) + 1) % 1) * length;
  return { position, resultant, totalWeight: sum, undefinedThreshold: 1e-10 };
}

/** Positive-frequency, positive-k modes below Nyquist; real displacements and
 * velocities obey m q''=kappa*(q_next-2q+q_prev) exactly, without time stepping.
 */
export function latticeDispersivePacket(s: LatticePacket, t: number) {
  if (![32, 64].includes(s.particles)) throw new RangeError('Use32 or64 particles.');
  bounded(s.length, 4, 16, 'Length'); bounded(s.density, .25, 4, 'Line density'); bounded(s.tension, .25, 4, 'Tension');
  bounded(s.amplitude, 0, .2, 'Peak amplitude bound'); bounded(s.modeCenter, 1, s.particles / 2 - 1, 'Carrier mode');
  bounded(s.bandwidth, .2, 8, 'Mode bandwidth'); bounded(s.center, 0, s.length, 'Initial center');
  bounded(s.phase, -4 * Math.PI, 4 * Math.PI, 'Initial phase'); bounded(t, 0, 16, 'Time');
  const n = s.particles, spacing = s.length / n, mass = s.density * spacing, stiffness = s.tension / spacing;
  const c = Math.sqrt(s.tension / s.density);
  const raw = Array.from({ length: n / 2 - 1 }, (_, i) => Math.exp(-.5 * ((i + 1 - s.modeCenter) / s.bandwidth) ** 2));
  const total = raw.reduce((a, b) => a + b, 0);
  const modes = raw.map((weight, i) => {
    const mode = i + 1, waveNumber = 2 * Math.PI * mode / s.length;
    const frequency = 2 * c / spacing * Math.sin(waveNumber * spacing / 2), amplitude = s.amplitude * weight / total;
    return { mode, waveNumber, frequency, amplitude, phase: s.phase - waveNumber * s.center,
      phaseVelocity: frequency / waveNumber, groupVelocity: c * Math.cos(waveNumber * spacing / 2),
      dispersion: -c * spacing / 2 * Math.sin(waveNumber * spacing / 2),
      energy: mass * n * frequency ** 2 * amplitude ** 2 / 2 };
  });
  const at = (x: number) => {
    let real = 0, imaginary = 0, velocity = 0, acceleration = 0;
    for (const m of modes) {
      const phase = m.waveNumber * x - m.frequency * t + m.phase;
      real += m.amplitude * Math.cos(phase); imaginary += m.amplitude * Math.sin(phase);
      velocity += m.frequency * m.amplitude * Math.sin(phase);
      acceleration -= m.frequency ** 2 * m.amplitude * Math.cos(phase);
    }
    return { x, displacement: real, velocity, acceleration, analyticField: [real, imaginary] as PacketComplex, analyticIntensity: real ** 2 + imaginary ** 2 };
  };
  const x = Array.from({ length: n }, (_, i) => i * spacing), points = x.map(at);
  const position = points.map(p => p.displacement), velocity = points.map(p => p.velocity);
  const force = position.map((q, i) => stiffness * (position[(i + 1) % n] - 2 * q + position[(i + n - 1) % n]));
  const kinetic = velocity.map(v => mass * v * v / 2);
  const linkEnergy = position.map((q, i) => stiffness * (position[(i + 1) % n] - q) ** 2 / 2);
  const linkFlux = position.map((q, i) => -stiffness * (position[(i + 1) % n] - q) * (velocity[(i + 1) % n] + velocity[i]) / 2);
  const assignedEnergy = kinetic.map((e, i) => e + (linkEnergy[i] + linkEnergy[(i + n - 1) % n]) / 2);
  const linkRate = position.map((q, i) => stiffness * (position[(i + 1) % n] - q) * (velocity[(i + 1) % n] - velocity[i]));
  const assignedEnergyRate = velocity.map((v, i) => v * force[i] + (linkRate[i] + linkRate[(i + n - 1) % n]) / 2);
  const energy = assignedEnergy.reduce((a, b) => a + b, 0), modalEnergy = modes.reduce((sum, m) => sum + m.energy, 0);
  const carrierWaveNumber = 2 * Math.PI * s.modeCenter / s.length;
  const carrierFrequency = 2 * c / spacing * Math.sin(carrierWaveNumber * spacing / 2);
  const analyticIntensity = points.map(p => p.analyticIntensity);
  return { time: t, length: s.length, spacing, mass, stiffness, c, modes, x, points, position, velocity, force,
    initialSlopeBound: modes.reduce((sum, m) => sum + m.waveNumber * m.amplitude, 0),
    kinetic, linkEnergy, assignedEnergy, assignedEnergyRate, linkFlux, energy, modalEnergy,
    energyTransportVelocity: energy === 0 ? null : spacing * linkFlux.reduce((a, b) => a + b, 0) / energy,
    energyWeightedGroupVelocity: modalEnergy === 0 ? null : modes.reduce((sum, m) => sum + m.energy * m.groupVelocity, 0) / modalEnergy,
    carrierWaveNumber, carrierFrequency, carrierPhaseVelocity: carrierFrequency / carrierWaveNumber,
    carrierGroupVelocity: c * Math.cos(carrierWaveNumber * spacing / 2),
    carrierDispersion: -c * spacing / 2 * Math.sin(carrierWaveNumber * spacing / 2),
    analyticNorm: spacing * analyticIntensity.reduce((a, b) => a + b, 0),
    expectedAnalyticNorm: s.length * modes.reduce((sum, m) => sum + m.amplitude ** 2, 0),
    analyticCenter: periodicPacketCenter(analyticIntensity, x, s.length),
    energyCenter: periodicPacketCenter(assignedEnergy, x, s.length),
    normalization: 'sum-of-modal-amplitudes-equals-prescribed-peak-bound' as const,
    model: 'real-periodic-nearest-neighbor-chain-positive-resolved-modes' as const };
}
