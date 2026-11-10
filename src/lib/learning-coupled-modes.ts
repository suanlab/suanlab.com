import type {Vec2} from './learning-simulations';

/** SI units. This educational grid uses quarter-kg masses and 1/16-N/m
 * stiffnesses to make structural zero/degenerate cases unambiguous.
 * Initial data and query times need not lie on the grid.
 */
export interface CoupledSystem {mass1: number; mass2: number; stiffness1: number; stiffness2: number; coupling: number;}
export interface CoupledInitialState {position: Vec2; velocity: Vec2;}
export interface BeatingSystem {mass: number; stiffness: number; coupling: number; amplitude: number;}
type Matrix2 = [Vec2, Vec2];
function grid(value: number, scale: number, min: number, max: number) {
  if (!Number.isFinite(value) || value < min || value > max || !Number.isInteger(value * scale)) throw new RangeError('Parameter outside the supported physical grid.');
}
function validate(s: CoupledSystem) {
  grid(s.mass1, 4, .25, 4); grid(s.mass2, 4, .25, 4);
  [s.stiffness1, s.stiffness2, s.coupling].forEach(v => grid(v, 16, 0, 4));
}
function initialData(initial: CoupledInitialState) {
  for (const v of [initial.position, initial.velocity]) {
    if (!Array.isArray(v) || v.length !== 2) throw new RangeError('Two initial coordinates are required.');
    for (let i = 0; i < 2; i++) if (!Number.isFinite(v[i]) || Math.abs(v[i]) > 4) throw new RangeError('Initial coordinates must be finite and between -4 and 4.');
  }
}
function time(value: number, max = 512) {
  if (!Number.isFinite(value) || value < 0 || value > max) throw new RangeError(`Time must be finite and between 0 and ${max} seconds.`);
}
const sinc = (value: number) => value === 0 ? 1 : Math.sin(value) / value;
const dot = (a: Vec2, b: Vec2) => a[0]*b[0]+a[1]*b[1];
const matrixVector = (m: Matrix2, v: Vec2): Vec2 => [dot(m[0],v),dot(m[1],v)];
function mechanics(s: CoupledSystem, position: Vec2, velocity: Vec2) {
  const {mass1:m, mass2:n, stiffness1:k, stiffness2:l, coupling:c} = s;
  const springForce = c*(position[1]-position[0]);
  const force: Vec2 = [-k*position[0]+springForce,-l*position[1]-springForce];
  const kinetic: Vec2 = [m*velocity[0]**2/2,n*velocity[1]**2/2];
  const anchorEnergy: Vec2 = [k*position[0]**2/2,l*position[1]**2/2];
  const couplingEnergy = c*(position[1]-position[0])**2/2;
  const bareEnergy: Vec2 = [kinetic[0]+anchorEnergy[0],kinetic[1]+anchorEnergy[1]];
  const assignedEnergy: Vec2 = [bareEnergy[0]+couplingEnergy/2,bareEnergy[1]+couplingEnergy/2];
  const flux = c*(position[0]-position[1])*(velocity[0]+velocity[1])/2;
  return {position,velocity,force,acceleration:[force[0]/m,force[1]/n] as Vec2,kinetic,anchorEnergy,couplingEnergy,bareEnergy,assignedEnergy,
    energy:bareEnergy[0]+bareEnergy[1]+couplingEnergy,flux12:flux,
    // These are actual derivatives from forces, not definitions from the flux.
    assignedEnergyRates:[velocity[0]*force[0]+k*position[0]*velocity[0]+c*(position[1]-position[0])*(velocity[1]-velocity[0])/2,
      velocity[1]*force[1]+l*position[1]*velocity[1]+c*(position[1]-position[0])*(velocity[1]-velocity[0])/2] as Vec2};
}

/** Ordered mass-normalized modes, with a reproducible basis in degenerate cases.
 * A chosen basis in a repeated eigenspace is not a unique physical mode label.
 */
export function coupledModeSpectrum(s: CoupledSystem) {
  validate(s);
  const {mass1:m,mass2:n,stiffness1:k,stiffness2:l,coupling:c} = s;
  const mass: Matrix2 = [[m,0],[0,n]],stiffness: Matrix2 = [[k+c,-c],[-c,l+c]];
  const a=(k+c)/m,d=(l+c)/n,b=-c/Math.sqrt(m*n),gap=Math.hypot(a-d,2*b),high=(a+d+gap)/2;
  // Avoid subtracting nearly equal trace/gap values. det(K) is a sum of
  // nonnegative grid-exact products, so structural zero is not a tolerance test.
  const degenerate=c===0 && (16*k)*(4*n)===(16*l)*(4*m);
  const determinant=k*l+c*(k+l),low=degenerate?high:high===0?0:determinant/(m*n*high);
  const squaredFrequencies: Vec2 = [low,high],frequencies: Vec2 = [Math.sqrt(low),Math.sqrt(high)];
  let whitenedModes: Matrix2;
  if(c===0) whitenedModes=a<=d?[[1,0],[0,1]]:[[0,1],[1,0]];
  else {const angle=Math.atan2(2*b,a-d)/2;whitenedModes=[[-Math.sin(angle),Math.cos(angle)],[Math.cos(angle),Math.sin(angle)]];}
  // Arrays contain mode vectors, not rows of Phi.
  const modes=whitenedModes.map(u=>[u[0]/Math.sqrt(m),u[1]/Math.sqrt(n)] as Vec2) as Matrix2;
  const massGram=modes.map(v=>modes.map(w=>m*v[0]*w[0]+n*v[1]*w[1])) as Matrix2;
  const eigenResiduals=modes.map((v,j)=>{const kv=matrixVector(stiffness,v);return[kv[0]-squaredFrequencies[j]*m*v[0],kv[1]-squaredFrequencies[j]*n*v[1]] as Vec2;}) as Matrix2;
  const projectors=modes.map(v=>[[v[0]*v[0]*m,v[0]*v[1]*n],[v[1]*v[0]*m,v[1]*v[1]*n]] as Matrix2);
  return {system:{...s},mass,stiffness,whitenedStiffness:[[a,b],[b,d]] as Matrix2,squaredFrequencies,frequencies,
    modes,whitenedModes,massGram,eigenResiduals,projectors,degenerate,
    rank:high===0?0:determinant===0?1:2,zeroModeCount:high===0?2:determinant===0?1:0,
    periods:frequencies.map(w=>w>0?2*Math.PI/w:null),basisStatus:degenerate?'chosen-basis-in-repeated-eigenspace':'simple-modes'};
}
type Spectrum=ReturnType<typeof coupledModeSpectrum>;
function normalState(s: CoupledSystem, initial: CoupledInitialState, t: number, spectrum: Spectrum) {
  const m=s.mass1,n=s.mass2;
  const initialCoordinates=spectrum.modes.map(v=>m*v[0]*initial.position[0]+n*v[1]*initial.position[1]) as Vec2;
  const initialRates=spectrum.modes.map(v=>m*v[0]*initial.velocity[0]+n*v[1]*initial.velocity[1]) as Vec2;
  const coordinates=spectrum.frequencies.map((w,j)=>initialCoordinates[j]*Math.cos(w*t)+initialRates[j]*t*sinc(w*t)) as Vec2;
  const rates=spectrum.frequencies.map((w,j)=>-initialCoordinates[j]*w*Math.sin(w*t)+initialRates[j]*Math.cos(w*t)) as Vec2;
  const contributions=spectrum.modes.map((v,j)=>v.map(a=>a*coordinates[j]) as Vec2);
  const velocityContributions=spectrum.modes.map((v,j)=>v.map(a=>a*rates[j]) as Vec2);
  const position: Vec2=[contributions[0][0]+contributions[1][0],contributions[0][1]+contributions[1][1]];
  const velocity: Vec2=[velocityContributions[0][0]+velocityContributions[1][0],velocityContributions[0][1]+velocityContributions[1][1]];
  const modalEnergies=coordinates.map((a,j)=>(rates[j]**2+spectrum.squaredFrequencies[j]*a*a)/2) as Vec2;
  const physical=mechanics(s,position,velocity);
  return {time:t,...physical,coordinates,rates,contributions,velocityContributions,modalEnergies,
    modalEnergySum:modalEnergies[0]+modalEnergies[1],modalPhysicalResidual:modalEnergies[0]+modalEnergies[1]-physical.energy};
}
export function coupledModeState(s: CoupledSystem, initial: CoupledInitialState, t: number) {
  initialData(initial);time(t);
  return normalState(s,initial,t,coupledModeSpectrum(s));
}
function drawingIntervals(w: number, duration: number) {return Math.max(64,2*Math.ceil(w*duration/.25));}
export function massWeightedCoupledNormalModes(s: CoupledSystem, initial: CoupledInitialState, duration: number, queryTime: number) {
  initialData(initial);time(duration,128);time(queryTime,128);
  if(queryTime>duration)throw new RangeError('Query time is outside the observation window.');
  const spectrum=coupledModeSpectrum(s),intervals=drawingIntervals(spectrum.frequencies[1],duration);
  const states=Array.from({length:intervals+1},(_,i)=>normalState(s,initial,duration*i/intervals,spectrum));
  const energy0=mechanics(s,initial.position,initial.velocity).energy;
  return {spectrum,initial:{position:[...initial.position] as Vec2,velocity:[...initial.velocity] as Vec2},duration,query:normalState(s,initial,queryTime,spectrum),states,
    energy0,maximumEnergyDrift:Math.max(...states.map(p=>Math.abs(p.energy-energy0))),
    modalEnergyFractions:energy0>0?states[0].modalEnergies.map(e=>e/energy0):null,
    sampling:{method:'analytic-normal-flow',intervals,maximumPhaseAdvance:duration*spectrum.frequencies[1]/intervals}};
}

function beatingParameters(s: BeatingSystem) {
  const physical={mass1:s.mass,mass2:s.mass,stiffness1:s.stiffness,stiffness2:s.stiffness,coupling:s.coupling};
  validate(physical);
  if(!Number.isFinite(s.amplitude)||s.amplitude<0||s.amplitude>4)throw new RangeError('Initial displacement amplitude must be between 0 and 4 m.');
  const low=Math.sqrt(s.stiffness/s.mass),high=Math.sqrt((s.stiffness+2*s.coupling)/s.mass);
  const split=s.coupling===0?0:(2*s.coupling/s.mass)/(high+low);
  return {physical,low,high,split,carrier:(low+high)/2};
}
function envelopeState(amplitude: number, carrier: number, split: number, t: number) {
  const C=Math.cos(carrier*t),S=Math.sin(carrier*t),c=Math.cos(split*t/2),s=Math.sin(split*t/2);
  return {position:[amplitude*C*c,amplitude*S*s] as Vec2,
    velocity:[-amplitude*(carrier*S*c+split*C*s/2),amplitude*(carrier*C*s+split*S*c/2)] as Vec2,
    signedEnvelopes:[amplitude*c,amplitude*s] as Vec2};
}
const sineIntegral=(frequency: number,t: number)=>frequency*t*t*sinc(frequency*t/2)**2/2;
function beatingState(s: BeatingSystem, t: number, p: ReturnType<typeof beatingParameters>) {
  const state=envelopeState(s.amplitude,p.carrier,p.split,t),physical=mechanics(p.physical,state.position,state.velocity);
  // Integrate the explicit trig flux, independently of endpoint energies.
  const netTransfer=s.coupling*s.amplitude**2*p.low*(sineIntegral(p.split,t)-sineIntegral(p.low+p.high,t))/4;
  const assignedEnergy0: Vec2=[s.stiffness*s.amplitude**2/2+s.coupling*s.amplitude**2/4,s.coupling*s.amplitude**2/4];
  const approximateSplit=p.low>0?s.coupling/(s.mass*p.low):null;
  const approximation=approximateSplit===null?null:envelopeState(s.amplitude,p.low+approximateSplit/2,approximateSplit,t);
  return {time:t,...physical,signedEnvelopes:state.signedEnvelopes,netTransfer,assignedEnergy0,
    transferResidual:physical.assignedEnergy[1]-assignedEnergy0[1]-netTransfer,
    analyticFlux:-s.coupling*s.amplitude**2*p.low*Math.cos(p.high*t)*Math.sin(p.low*t)/2,
    approximation,positionApproximationError:approximation?Math.hypot(approximation.position[0]-state.position[0],approximation.position[1]-state.position[1]):null};
}
export function coupledBeatingState(s: BeatingSystem, t: number) {time(t);return beatingState(s,t,beatingParameters(s));}
export function coupledModeBeatingEnergyTransfer(s: BeatingSystem, duration: number, queryTime: number) {
  time(duration,128);time(queryTime,128);if(queryTime>duration)throw new RangeError('Query time is outside the observation window.');
  const p=beatingParameters(s),intervals=drawingIntervals(p.high,duration),states=Array.from({length:intervals+1},(_,i)=>beatingState(s,duration*i/intervals,p));
  const swapTime=s.amplitude>0&&p.split>0?Math.PI/p.split:null;
  const firstEnvelopeSwap=swapTime===null?null:{time:swapTime,withinWindow:swapTime<=duration,state:beatingState(s,swapTime,p)};
  const energy0=(s.stiffness+s.coupling)*s.amplitude**2/2;
  return {system:{...s},duration,query:beatingState(s,queryTime,p),states,frequencies:[p.low,p.high] as Vec2,spectralSplit:p.split,
    carrierFrequency:p.carrier,spectralEnvelopePeriod:p.split>0?2*Math.PI/p.split:null,
    signedEnvelopePeriod:p.split>0?4*Math.PI/p.split:null,observedEnvelopePeriod:s.amplitude>0&&p.split>0?2*Math.PI/p.split:null,
    weakCouplingRatio:s.stiffness>0?s.coupling/s.stiffness:null,firstEnvelopeSwap,energy0,
    assignedEnergyFractions:energy0>0?states.map(p=>p.assignedEnergy.map(e=>e/energy0) as Vec2):null,
    maximumEnergyDrift:Math.max(...states.map(p=>Math.abs(p.energy-energy0))),
    maximumTransferResidual:Math.max(...states.map(p=>Math.abs(p.transferResidual))),
    maximumPositionApproximationError:s.stiffness>0?Math.max(...states.map(p=>p.positionApproximationError!)):null,
    interpretation:s.stiffness===0?'zero-mode-and-relative-motion':s.coupling===0?'uncoupled':'two-frequency-superposition',
    sampling:{method:'analytic-two-frequency-flow',intervals,maximumPhaseAdvance:p.high*duration/intervals}};
}
