import{continuumConservativeKinds,isContinuumConservativeKind,volumeDiscretizationLab,godunovDiscretizationLab}from './learning-continuum-conservative-lab';
import{continuumTurbulenceKinds,isContinuumTurbulenceKind,reynoldsLab,gaussianFilterLab}from './learning-continuum-turbulence-lab';
import{continuumMultiphysicsKinds,isContinuumMultiphysicsKind,thermoelasticLab,thermoelectricLab}from './learning-continuum-multiphysics-lab';
import{continuumDimensionlessKinds,isContinuumDimensionlessKind,pecletLab,damkohlerLab}from './learning-continuum-dimensionless-lab';
import{continuumConstitutiveKinds,isContinuumConstitutiveKind,viscoelasticLab,neoHookeanLab}from './learning-continuum-constitutive-lab';
import {continuumReactionKinds,isContinuumReactionKind,reversibleTransportLab,fisherTransportLab}from './learning-continuum-reaction-transport-lab';
import {continuumDiffusionKinds,isContinuumDiffusionKind,heatDiffusionLab,gaussianDiffusionLab}from './learning-continuum-diffusion-lab';
import {continuumFlowKinds,isContinuumFlowKind,acousticFlowLab,taylorGreenFlowLab}from './learning-continuum-euler-navier-stokes-lab';
import {continuumStressStrainKinds,isContinuumStressStrainKind,tractionStressLab,finiteStrainLab}from './learning-continuum-stress-strain-lab';
import {continuumAssumptionKinds,isContinuumAssumptionKind,particleAssumptionLab,scaleAssumptionLab} from './learning-continuum-assumptions-lab';
import type {FoundationResult} from './learning-foundations';
import {
  transportedDensityState, movingMassWindowState, movingMassWindowBalance,
  affineGasState, affineGasWindowState, affineGasWindowBalance, type AffineControlWindow,
} from './learning-continuum-conservation';
export const continuumKinds = ['moving-control-volume-mass-transport', 'compressible-flow-momentum-energy-balance'] as const;
export type ContinuumKind = typeof continuumConservativeKinds[number]|typeof continuumTurbulenceKinds[number]|typeof continuumKinds[number]|typeof continuumAssumptionKinds[number]|typeof continuumStressStrainKinds[number]|typeof continuumFlowKinds[number]|typeof continuumDiffusionKinds[number]|typeof continuumReactionKinds[number]|typeof continuumConstitutiveKinds[number]|typeof continuumDimensionlessKinds[number]|typeof continuumMultiphysicsKinds[number];
export const isContinuumKind = (id: string): id is ContinuumKind => (continuumKinds as readonly string[]).includes(id)||isContinuumAssumptionKind(id)||isContinuumStressStrainKind(id)||isContinuumFlowKind(id)||isContinuumDiffusionKind(id)||isContinuumReactionKind(id)||isContinuumConstitutiveKind(id)||isContinuumDimensionlessKind(id)||isContinuumMultiphysicsKind(id)||isContinuumTurbulenceKind(id)||isContinuumConservativeKind(id);
function validate(v: number[], mass: boolean) {
  const bounds = mass ? [[.25,4],[-.8,.8],[0,8],[-2,2],[.25,4],[-1,1],[0,1],[-2,2],[-1,1],[-1,1]]
    : [[.25,4],[0,4],[-.5,.5],[-2,2],[1.1,2],[.25,4],[-1,1],[0,1],[-2,2],[0,2],[-1,1],[-1,1]];
  if (v.length !== bounds.length) throw new RangeError('Wrong continuum control count');
  for (let i = 0; i < v.length; i++) {
    if (!Number.isFinite(v[i]) || v[i] < bounds[i][0] || v[i] > bounds[i][1] || !mass && i === 9 && !Number.isInteger(v[i])) throw new RangeError(`Invalid continuum control ${i}`);
  }
}
function massLab(v: number[]) {
  const system = {meanDensity:v[0],contrast:v[1],wavenumber:v[2],velocity:v[3],area:v[4]};
  const window = {center:v[5],halfWidth:v[6],speed:v[7]}, start=v[8], time=v[9];
  const query = movingMassWindowState(system,window,time), balance=movingMassWindowBalance(system,window,start,time);
  const segments=Math.max(160,2*Math.ceil(10*system.wavenumber*16/(4*Math.PI)));
  const spatial=Array.from({length:segments+1},(_,i)=>{const x=-5+10*i/segments;return{x,value:transportedDensityState(system,x,time),initial:transportedDensityState(system,x,start)};});
  const temporal=Array.from({length:81},(_,i)=>{const time=-1+i/40;return{time,value:movingMassWindowState(system,window,time)};});
  const particles=Array.from({length:9},(_,i)=>{const initial=-2+i/2;return{initial,current:initial+system.velocity*time};});
  return {system,window,start,time,query,balance,spatial,temporal,particles};
}
function gasLab(v: number[]) {
  const system={initialDensity:v[0],initialPressure:v[1],expansionRate:v[2],drift:v[3],gamma:v[4],area:v[5]};
  const window:AffineControlWindow={center:v[6],halfWidth:v[7],speed:v[8],mode:(['fixed','translating','material'] as const)[v[9]]};
  const start=v[10],time=v[11],query=affineGasWindowState(system,window,time),balance=affineGasWindowBalance(system,window,start,time);
  const spatial=Array.from({length:161},(_,i)=>{const x=-5+i/16;return{x,value:affineGasState(system,x,time)};});
  const temporal=Array.from({length:81},(_,i)=>{const time=-1+i/40;return{time,value:affineGasWindowState(system,window,time)};});
  const particles=Array.from({length:9},(_,i)=>{const initial=-2+i/2;return{initial,current:(1+system.expansionRate*time)*initial+system.drift*time};});
  return {system,window,start,time,query,balance,spatial,temporal,particles};
}
export interface ContinuumResult extends FoundationResult {
  kind:ContinuumKind;volumeDiscretization?:ReturnType<typeof volumeDiscretizationLab>;godunovDiscretization?:ReturnType<typeof godunovDiscretizationLab>;reynolds?:ReturnType<typeof reynoldsLab>;gaussianFilter?:ReturnType<typeof gaussianFilterLab>;thermoelastic?:ReturnType<typeof thermoelasticLab>;thermoelectric?:ReturnType<typeof thermoelectricLab>;peclet?:ReturnType<typeof pecletLab>;damkohler?:ReturnType<typeof damkohlerLab>;viscoelastic?:ReturnType<typeof viscoelasticLab>;neoHookean?:ReturnType<typeof neoHookeanLab>;reversibleTransport?:ReturnType<typeof reversibleTransportLab>;fisherTransport?:ReturnType<typeof fisherTransportLab>;heatDiffusion?:ReturnType<typeof heatDiffusionLab>;gaussianDiffusion?:ReturnType<typeof gaussianDiffusionLab>;acousticFlow?:ReturnType<typeof acousticFlowLab>;taylorGreenFlow?:ReturnType<typeof taylorGreenFlowLab>;stressTraction?:ReturnType<typeof tractionStressLab>;finiteDeformation?:ReturnType<typeof finiteStrainLab>;particleAveraging?:ReturnType<typeof particleAssumptionLab>;scaleSeparation?:ReturnType<typeof scaleAssumptionLab>;massTransport?:ReturnType<typeof massLab>;gasConservation?:ReturnType<typeof gasLab>;
}
export function evaluateContinuum(kind:ContinuumKind,values:number[]):ContinuumResult {
  if (!isContinuumKind(kind)) throw new RangeError('Unknown continuum lesson');
  if(kind===continuumConservativeKinds[0]){const r=volumeDiscretizationLab(values);return{kind,volumeDiscretization:r,curves:[],stats:{'physical mass ledger change · kg':r.correct.massChangeFromFaces.value,'wrong storage mass change · kg':r.wrong.massChangeFromFaces.value,'stable quadratic norm change · kg²/m³':r.correct.normChangeUsingMassLedger.value,'cell-average L1 error · kg':r.l1CellAverageError.value}};}
  if(kind===continuumConservativeKinds[1]){const r=godunovDiscretizationLab(values);return{kind,godunovDiscretization:r,curves:[],stats:{'Godunov profile L1 error':r.godunov.final.l1ReconstructedProfileError.value,'Roe profile L1 error':r.roe.final.l1ReconstructedProfileError.value,'actual boundary mass supply':r.godunov.final.massChangeFromBoundary.value,'positive Euler entropy correction':r.godunov.final.positiveEulerLedger.value}};}
  if(kind===continuumTurbulenceKinds[0]){const r=reynoldsLab(values),q=r.query;return{kind,reynolds:r,curves:[],stats:{'covariance UV · m²/s²':q.covarianceUV.value,'mean force X · N/m³':q.meanForceX.value,'inferred viscosity · m²/s':q.inferredEddyViscosity?.value??null,'mean transfer · W':r.budget.meanToFluctuationTransfer.value}};}
  if(kind===continuumTurbulenceKinds[1]){const r=gaussianFilterLab(values),q=r.query;return{kind,gaussianFilter:r,curves:[],stats:{'residual variance · m²/s²':q.variance.value,'filtered residual · m/s':q.filteredResidualVelocity.value,'signed local transfer · m²/s³':q.transferPerMass.value,'resolved energy rate · W':r.budget.resolvedEnergyRate.value}};}
  if(kind===continuumMultiphysicsKinds[0]){const r=thermoelasticLab(values),q=r.query;return{kind,thermoelastic:r,curves:[],stats:{'temperature rate amplitude · K/s':q.temperatureRateAmplitude.value,'total availability · J':q.totalAvailability.value,'conduction availability loss · W':q.conductionAvailabilityLoss.value,'one-way availability rate · W':r.oneWay.totalAvailabilityRate.value}};}
  if(kind===continuumMultiphysicsKinds[1]){const r=thermoelectricLab(values),q=r.query;return{kind,thermoelectric:r,curves:[],stats:{'terminal voltage drop · V':q.terminalVoltageDrop.value,'left heat input · W':q.leftHeatInput.value,'local electrical work · W/m³':q.localElectricalWork.value,'electrical power input · W':q.electricalPowerInput.value}};}
  if(kind===continuumDimensionlessKinds[0]){const r=pecletLab(values),q=r.query;return{kind,peclet:r,curves:[],stats:{'signed Peclet number':q.peclet.value,'concentration · kg/m³':q.concentration.value,'total flux · kg/(m² s)':q.totalFlux.value,'similar total transport · kg/s':r.comparison?.totalTransport.value??null}};}
  if(kind===continuumDimensionlessKinds[1]){const r=damkohlerLab(values),q=r.query;return{kind,damkohler:r,curves:[],stats:{'Damkohler number':q.damkohler.value,'sealed concentration · kg/m³':r.sealed.concentration.value,'total consumption · kg/s':q.totalConsumption.value,'analytic effectiveness':q.analyticEffectiveness.value}};}
  if(kind===continuumConstitutiveKinds[0]){const r=viscoelasticLab(values),q=r.query;return{kind,viscoelastic:r,curves:[],stats:{'Maxwell stress · Pa':q.maxwell.stress,'initial memory · Pa':q.homogeneousMemory,'Maxwell steady-cycle work · J/m³':r.cycle.maxwellSteadyCycleWork,'stable stress change · Pa':q.stableMaxwellStressChange}};}
  if(kind===continuumConstitutiveKinds[1]){const r=neoHookeanLab(values),q=r.query;return{kind,neoHookean:r,curves:[],stats:{'energy density · J/m³':q.energyDensity,'axial force · N':q.force,'normal stress · Pa':q.normalStress,'stable stretch change':q.stableStretchChange}};}
  if(kind===continuumReactionKinds[0]){const r=reversibleTransportLab(values),q=r.query;return{kind,reversibleTransport:r,curves:[],stats:{'A fraction':q.state.fractionA,'forward turnover · kg/(m³ s)':q.forwardTurnover,'window A balance · kg/s':r.window?.a.balanceResidual??null,'stable A fraction change':q.state.stableFractionChangeA}};}
  if(kind===continuumReactionKinds[1]){const r=fisherTransportLab(values),q=r.query;return{kind,fisherTransport:r,curves:[],stats:{'growth source · kg/(m³ s)':q.source,'laboratory speed · m/s':q.laboratorySpeed,'omitted source residual · kg/(m³ s)':q.omittedSourceResidual,'stable deficit · kg/m³':q.stableDeficit}};}
  if(kind===continuumDiffusionKinds[0]){const r=heatDiffusionLab(values);return{kind,heatDiffusion:r,curves:[],stats:{'heat flux · W/m²':r.query.heatFlux,'window energy rate · W':r.window.energyRate,'period sensible energy · J':r.cell.sensibleEnergy,'quadratic temperature norm · K² m³':r.cell.quadraticNorm,'stable relative amplitude change':r.query.stableRelativeDecayChange}};}
  if(kind===continuumDiffusionKinds[1]){const r=gaussianDiffusionLab(values);return{kind,gaussianDiffusion:r,curves:[],stats:{'peak concentration · kg/m³':r.peak,'declared variance · m²':r.state.variance,'window mass · kg':r.window?.massInside??null,'stable window fraction':r.window?.fraction??null,'mass-weighted mean · m':r.state.massWeightedMean}};}
  if(kind===continuumFlowKinds[0]){const r=acousticFlowLab(values),q=r.query;return{kind,acousticFlow:r,curves:[],stats:{'right signal speed · m/s':q.characteristicSpeeds[0],'left signal speed · m/s':q.characteristicSpeeds[1],'contact variable · Pa':q.contactVariable,'nonlinear mass residual · kg/(m³ s)':q.nonlinearResidual.mass,'acoustic energy · J/m³':q.acousticEnergy}};}
  if(kind===continuumFlowKinds[1]){const r=taylorGreenFlowLab(values),q=r.query;return{kind,taylorGreenFlow:r,curves:[],stats:{'convection x · m/s²':q.convection[0],'viscous acceleration x · m/s²':q.viscousAcceleration[0],'cell kinetic energy · J':r.cell.kineticEnergy,'stable amplitude change · m/s':q.decay.amplitudeChange,'tracer endpoint difference · m':r.endpointDifference}};}
  if(kind===continuumStressStrainKinds[0]){const r=tractionStressLab(values),q=r.query;return{kind,stressTraction:r,curves:[],stats:{'normal stress · Pa':q.normalStress,'signed shear · Pa':q.signedShear,'static net force x · N':r.balance.totalForce?.[0]??null,'central surface moment · N m':r.balance.surfaceMoment,'in-plane principal direction · rad':q.principalDirection}};}
  if(kind===continuumStressStrainKinds[1]){const r=finiteStrainLab(values),q=r.query;return{kind,finiteDeformation:r,curves:[],stats:{'directional engineering strain':q.stableEngineeringStrain,'Green E22':q.green[1][1],'infinitesimal epsilon11':q.infinitesimal[0][0],'skew-control Green diagonal':r.control.greenScalar,'current area · m²':q.area}};}
  if(kind===continuumAssumptionKinds[0]){const r=particleAssumptionLab(values),q=r.query;return{kind,particleAveraging:r,curves:[],stats:{'line density · kg/m':q.density,'weighted velocity · m/s':q.velocity,'dispersion transport · N':q.dispersion,'smoothing contribution · kg/(m s)':q.smoothingRate}};}
  if(kind===continuumAssumptionKinds[1]){const r=scaleAssumptionLab(values),q=r.query;return{kind,scaleSeparation:r,curves:[],stats:{'expected count':q.expectedCount,'Knudsen number':q.knudsen,'relative Poisson noise':q.relativeCountStd,'declared feasible interval · 0/1':q.windowExists?1:0,'binomial relative noise':r.count.relativeStd}};}
  const first=kind===continuumKinds[0];validate(values,first);
  if (first) {
    const r=massLab(values),q=r.query;
    return {kind,massTransport:r,curves:[],stats:{'window mass · kg':q.mass,'mass rate · kg/s':q.massRate,'relative speed · m/s':q.relativeSpeed,'stable mass change · kg':r.balance.massChange}};
  }
  const r=gasLab(values),q=r.query;
  return {kind,gasConservation:r,curves:[],stats:{'mass · kg':q.storage[0],'momentum · kg m/s':q.storage[1],'total energy · J':q.storage[2],'pressure work net · W':q.pressureWorkNet,'stable energy change · J':r.balance.changes[2]}};
}
