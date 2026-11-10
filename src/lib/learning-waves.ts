import {samplingSensorKinds,evaluateSamplingSensors,type SamplingSensorKind,type SamplingSensorResult} from './learning-sampling-sensors-lab';
import {fourierOpticsKinds,evaluateFourierOptics,type FourierOpticsKind,type FourierOpticsResult} from './learning-fourier-optics-lab';
import {rayWaveOpticsKinds,evaluateRayWaveOptics,type RayWaveOpticsKind,type RayWaveOpticsResult} from './learning-ray-wave-optics-lab';
import {polarizationKinds,evaluatePolarization,type PolarizationKind,type PolarizationResult} from './learning-polarization-lab';
import {dispersionKinds,evaluateDispersion,type DispersionKind,type DispersionResult} from './learning-dispersion-lab';
import {interferenceKinds,evaluateInterference,type InterferenceKind,type InterferenceResult} from './learning-interference-lab';
import {waveEquationKinds,evaluateWaveEquation,type WaveEquationKind,type WaveEquationResult} from './learning-wave-equation-lab';
import type {FoundationResult} from './learning-foundations';
import type {Vec2} from './learning-simulations';
import {forcedOscillatorState, oscillatorFrequencyResponse, sinusoidalForcingTransient, type ForcedOscillator} from './learning-waves-oscillations';
import {massWeightedCoupledNormalModes,coupledModeBeatingEnergyTransfer} from './learning-coupled-modes';

export const wavesKinds = ['sinusoidal-forcing-transient-response', 'oscillator-frequency-response-power-resonance','mass-weighted-coupled-normal-modes','coupled-mode-beating-energy-transfer',...waveEquationKinds,...interferenceKinds,...dispersionKinds,...polarizationKinds,...rayWaveOpticsKinds,...fourierOpticsKinds,...samplingSensorKinds] as const;
export type WavesKind = typeof wavesKinds[number];
export type OscillatorWavesKind = typeof wavesKinds[0] | typeof wavesKinds[1];
export type CoupledWavesKind = typeof wavesKinds[2] | typeof wavesKinds[3];
export const isWavesKind = (kind: string): kind is WavesKind => (wavesKinds as readonly string[]).includes(kind);
export interface OscillatorWavesResult extends FoundationResult {
  kind: OscillatorWavesKind;
  system: ForcedOscillator;
  response: ReturnType<typeof oscillatorFrequencyResponse>;
  transient?: ReturnType<typeof sinusoidalForcingTransient>;
  query?: ReturnType<typeof forcedOscillatorState>;
  sweep?: {frequency: number; response: ReturnType<typeof oscillatorFrequencyResponse>}[];
  periodic?: {time: number; position: number; velocity: number; force: number; power: number; heat: number; energy: number}[];
}
export interface CoupledWavesResult extends FoundationResult {
  kind: CoupledWavesKind;
  coupledNormal?: ReturnType<typeof massWeightedCoupledNormalModes>;
  coupledBeating?: ReturnType<typeof coupledModeBeatingEnergyTransfer>;
}
export type WavesResult = SamplingSensorResult | FourierOpticsResult | RayWaveOpticsResult | OscillatorWavesResult | CoupledWavesResult | WaveEquationResult | InterferenceResult | DispersionResult | PolarizationResult;
function validate(values: number[], bounds: number[][]) {
  if (values.length !== bounds.length) throw new RangeError('Wrong waves control count.');
  for (let i = 0; i < bounds.length; i++) if (!Number.isInteger(values[i]) || values[i] < bounds[i][0] || values[i] > bounds[i][1]) throw new RangeError(`Invalid waves control ${i}.`);
}
/** Integer controls: [m,k,4c,F0,4Omega,2x0,2v0,2T,128t/T]. */
export function evaluateWaves(kind: OscillatorWavesKind, values: number[]): OscillatorWavesResult;
export function evaluateWaves(kind: CoupledWavesKind, values: number[]): CoupledWavesResult;
export function evaluateWaves(kind: WaveEquationKind, values: number[]): WaveEquationResult;
export function evaluateWaves(kind: InterferenceKind, values: number[]): InterferenceResult;
export function evaluateWaves(kind: DispersionKind, values: number[]): DispersionResult;
export function evaluateWaves(kind: PolarizationKind, values: number[]): PolarizationResult;
export function evaluateWaves(kind: RayWaveOpticsKind, values: number[]): RayWaveOpticsResult;
export function evaluateWaves(kind: SamplingSensorKind, values: number[]): SamplingSensorResult;
export function evaluateWaves(kind: FourierOpticsKind, values: number[]): FourierOpticsResult;
export function evaluateWaves(kind: WavesKind, values: number[]): WavesResult;
export function evaluateWaves(kind: WavesKind, values: number[]): WavesResult {
  if (!isWavesKind(kind)) throw new Error('Unknown waves unit.');
  if(kind===samplingSensorKinds[0]||kind===samplingSensorKinds[1])return evaluateSamplingSensors(kind,values);
  if(kind===fourierOpticsKinds[0]||kind===fourierOpticsKinds[1])return evaluateFourierOptics(kind,values);
  if(kind===rayWaveOpticsKinds[0]||kind===rayWaveOpticsKinds[1])return evaluateRayWaveOptics(kind,values);
  if(kind===polarizationKinds[0]||kind===polarizationKinds[1])return evaluatePolarization(kind,values);
  if(kind===dispersionKinds[0]||kind===dispersionKinds[1])return evaluateDispersion(kind,values);
  if(kind===interferenceKinds[0]||kind===interferenceKinds[1])return evaluateInterference(kind,values);
  if(kind===waveEquationKinds[0]||kind===waveEquationKinds[1])return evaluateWaveEquation(kind,values);
  if(kind===wavesKinds[2]||kind===wavesKinds[3]){
    const normal=kind===wavesKinds[2],ranges=normal?[[1,16],[1,16],[0,64],[0,64],[0,64],[-8,8],[-8,8],[-8,8],[-8,8],[0,128]]:[[1,16],[0,64],[0,64],[0,8],[0,128]];
    if(values.length!==ranges.length+1)throw new RangeError('Wrong coupled control count.');
    validate(values.slice(0,-1),ranges);
    const fraction=values[values.length-1];
    if(!Number.isFinite(fraction)||fraction<0||fraction>1)throw new RangeError('Query fraction must lie between zero and one.');
    if(normal){
      const s={mass1:values[0]/4,mass2:values[1]/4,stiffness1:values[2]/16,stiffness2:values[3]/16,coupling:values[4]/16};
      const r=massWeightedCoupledNormalModes(s,{position:[values[5]/2,values[6]/2],velocity:[values[7]/2,values[8]/2]},values[9],values[9]*fraction),q=r.query;
      return {kind,coupledNormal:r,curves:[r.states.map(p=>[p.time,p.position[0]]as Vec2)],stats:{'t · s':q.time,'q₁ · m':q.position[0],'q₂ · m':q.position[1],'v₁ · m/s':q.velocity[0],'v₂ · m/s':q.velocity[1],'ω₁ · rad/s':r.spectrum.frequencies[0],'ω₂ · rad/s':r.spectrum.frequencies[1],'E · J':q.energy,'E mode 1 · J':q.modalEnergies[0],'E mode 2 · J':q.modalEnergies[1],'ΣE mode − E · J':q.modalPhysicalResidual}};
    }
    const r=coupledModeBeatingEnergyTransfer({mass:values[0]/4,stiffness:values[1]/16,coupling:values[2]/16,amplitude:values[3]/2},values[4],values[4]*fraction),q=r.query;
    return {kind,coupledBeating:r,curves:[r.states.map(p=>[p.time,p.position[0]]as Vec2)],stats:{'t · s':q.time,'q₁ · m':q.position[0],'q₂ · m':q.position[1],'v₁ · m/s':q.velocity[0],'v₂ · m/s':q.velocity[1],'E · J':q.energy,'Ec · J':q.couplingEnergy,'E₁ bare · J':q.bareEnergy[0],'J₁→₂ · W':q.flux12,'W₁→₂ · J':q.netTransfer,'ΔE₂* − W₁→₂ · J':q.transferResidual,'Δω · rad/s':r.spectralSplit,'t swap · s':r.firstEnvelopeSwap?.time??null}};
  }
  const transientKind = kind === wavesKinds[0];
  validate(values, transientKind ? [[1,4],[1,4],[0,16],[0,4],[0,16],[-4,4],[-4,4],[0,24],[0,128]] : [[1,4],[1,4],[0,16],[0,4],[0,32]]);
  const [mass, stiffness, dampingCode, forceAmplitude, frequencyCode] = values;
  const system = {mass, stiffness, damping: dampingCode / 4, forceAmplitude, frequency: frequencyCode / 4};
  const response = oscillatorFrequencyResponse(system);
  if (transientKind) {
    const initial = {position: values[5] / 2, velocity: values[6] / 2}, duration = values[7] / 2;
    const transient = sinusoidalForcingTransient(system, initial, duration);
    const query = forcedOscillatorState(system, initial, duration * values[8] / 128);
    return {kind, system, response, transient, query,
      curves: [transient.states.map(s => [s.time, s.position] as Vec2)],
      stats: {'t · s': query.time, 'x(t) · m': query.position, 'v(t) · m/s': query.velocity, 'F(t) · N': query.force,
        'E(t) · J': query.energy, 'W(0,T) · J': transient.work, 'Q(0,T) · J': transient.heat,
        'ΔE(0,T) · J': transient.energyChange, 'ΔE−W+Q · J': transient.balanceResidual,
        'W/T · W': transient.finiteWindowMeanInputPower, '⟨P⟩ periodic · W': response.meanPower}};
  }
  const frequencies = Array.from({length: 257}, (_, i) => i / 32);
  frequencies.push(system.frequency, response.naturalFrequency);
  if (response.displacementPeakFrequency !== null) frequencies.push(response.displacementPeakFrequency);
  if (response.halfPower) frequencies.push(response.halfPower.lower, response.halfPower.upper);
  // Every characteristic marker lies within [0,8] under these controls.
  const sweep = [...new Set(frequencies)].sort((a,b) => a-b).map(frequency => ({frequency, response: oscillatorFrequencyResponse({...system, frequency})}));
  let periodic: OscillatorWavesResult['periodic'];
  const displacementCoefficients = forceAmplitude === 0 ? {real: 0, imaginary: 0} : response.susceptibility === null ? null : {
    real: forceAmplitude * response.susceptibility.real,
    imaginary: forceAmplitude * response.susceptibility.imaginary};
  if (displacementCoefficients !== null && response.period !== null) {
    const coefficient = displacementCoefficients;
    periodic = Array.from({length: 129}, (_, i) => {
      const time = response.period! * i / 128, angle = system.frequency * time;
      const position = coefficient.real * Math.cos(angle) - coefficient.imaginary * Math.sin(angle);
      const velocity = -system.frequency * (coefficient.real * Math.sin(angle) + coefficient.imaginary * Math.cos(angle));
      const force = forceAmplitude * Math.cos(angle);
      return {time, position, velocity, force, power: force * velocity, heat: system.damping * velocity * velocity, energy: (mass * velocity * velocity + stiffness * position * position) / 2};
    });
  }
  return {kind, system, response, sweep, periodic, curves: response.meanPower === null ? [] : [[[system.frequency, response.meanPower]]],
    stats: {'Ω · rad/s': system.frequency, 'ω₀ · rad/s': response.naturalFrequency,
      'ωd · rad/s': response.dampedFreeFrequency, 'Ωx · rad/s': response.displacementPeakFrequency,
      'ΩP · rad/s': response.powerPeakFrequency, 'G · m/N': response.gain, 'A · m': response.amplitude,
      'φ · rad': response.phaseLag, '⟨P⟩ · W': response.meanPower, 'Q · 1': response.qualityFactor,
      'Ω− · rad/s': response.halfPower?.lower ?? null, 'Ω+ · rad/s': response.halfPower?.upper ?? null,
      'ΔΩ · rad/s': response.halfPower?.width ?? null}};
}
