import type {FoundationResult} from './learning-foundations';
import type {Vec2} from './learning-simulations';
import {pixelApertureSampling,calibratedPhotonExposure,photonCountingSensor,pixelSinc} from './learning-sampling-sensors';
export const samplingSensorKinds=['pixel-aperture-sampling-aliasing','photon-counting-exposure-sensor-noise'] as const;
export type SamplingSensorKind=typeof samplingSensorKinds[number];
function validate(v:number[],bounds:number[][],ints:number[]=[]){if(v.length!==bounds.length)throw new RangeError('Wrong sampling/sensor control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||ints.includes(i)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid sampling/sensor control ${i}`);}
function pixelLab(v:number[]){
 validate(v,[[2,20],[0,1],[2,20],[0,1],[0,1],[0,4],[-Math.PI,Math.PI],[-1,1],[0,40],[-1,1],[2,16],[-16,16]],[10]);
 const pitch=v[0]/1e6,exposure=v[8]/1000;
 const system={pitch,apertureWidth:pitch*v[1],pixelHeight:v[2]/1e6,wavelength:550/1e9,irradiance:v[3]/1000,modulation:v[4],frequency:v[5]/pitch,phase:v[6],velocity:v[7]/1000,exposure,centerTime:exposure/2,origin:v[9]*pitch,radius:v[10],queryCoordinate:v[11]*pitch};
 const query=pixelApertureSampling(system),queryExposure=calibratedPhotonExposure(query.query.limitAverage,query.area,system.exposure,system.wavelength);
 const transferIntervals=Math.max(512,Math.ceil(48*Math.max(system.apertureWidth/pitch,Math.abs(system.velocity*system.exposure/pitch))));
 const transferProfile=Array.from({length:transferIntervals+1},(_,i)=>{const cyclesPerPitch=4*i/transferIntervals,frequency=cyclesPerPitch/pitch;return{cyclesPerPitch,spatial:pixelSinc(frequency*system.apertureWidth),temporal:pixelSinc(frequency*system.velocity*system.exposure),total:pixelSinc(frequency*system.apertureWidth)*pixelSinc(frequency*system.velocity*system.exposure)};});
 return{system,query,queryExposure,transferProfile};
}
function sensorLab(v:number[]){
 validate(v,[[0,1],[0,40],[0,1],[0,128],[0,16],[.05,8],[-16,256],[2,8],[1,8192],[0,255]],[7,8,9]);
 const calibration=calibratedPhotonExposure(v[0]/1000,25e-12,v[1]/1000,550/1e9);
 const system={incidentPhotonMean:calibration.incidentPhotonMean,quantumEfficiency:v[2],exposure:v[1]/1000,darkRate:v[3],readSigma:v[4],gain:v[5],offset:v[6],bits:v[7],fullWell:v[8]},query=photonCountingSensor(system);
 const stored=new Map<number,number>();for(const q of query.poisson.values){const n=Math.min(q.count,system.fullWell);stored.set(n,(stored.get(n)||0)+q.probability);}
 return{system,query,calibration,queryCode:v[9],queryCodeProbability:query.bins[v[9]]?.probability??0,storedDistribution:[...stored].sort((a,b)=>a[0]-b[0]).map(([count,probability])=>({count,probability}))};
}
export interface SamplingSensorResult extends FoundationResult {kind:SamplingSensorKind;samplingSensors:true;pixel?:ReturnType<typeof pixelLab>;sensor?:ReturnType<typeof sensorLab>;}
export function evaluateSamplingSensors(kind:SamplingSensorKind,v:number[]):SamplingSensorResult{
 if(kind===samplingSensorKinds[0]){const r=pixelLab(v),q=r.query;return{kind,samplingSensors:true,pixel:r,curves:[q.profile.map(p=>[p.coordinate*1e6,p.limitAverage*1000]as Vec2)],stats:{'box mean query · mW/m²':q.apertureAverageDefined?q.query.limitAverage*1000:null,'limit query · mW/m²':q.query.limitAverage*1000,'energy query · aJ':r.queryExposure.energy*1e18,'photons query · 1':r.queryExposure.incidentPhotonMean,'H spatial · 1':q.spatialTransfer,'H temporal · 1':q.temporalTransfer,'H total · 1':q.transfer,'alias · cycles/mm':q.aliasFrequency/1000,'Nyquist · cycles/mm':q.nyquistFrequency/1000,'sample count · 1':q.samples.length,'reconstruction query · mW/m²':q.query.finiteReconstruction*1000,'sampling margin · cycles/mm':(q.nyquistFrequency-r.system.frequency)/1000}};}
 if(kind!==samplingSensorKinds[1])throw new RangeError('Unknown sampling/sensor lesson');const r=sensorLab(v),q=r.query;
 return{kind,samplingSensors:true,sensor:r,curves:[q.bins.map(p=>[p.code,p.probability]as Vec2)],stats:{'photons · 1':r.calibration.incidentPhotonMean,'photo mean · e':q.photoMean,'dark mean · e':q.darkMean,'preADC SNR · 1':q.unsaturatedSignalToNoise,'analog mean · DN':q.unsaturatedAnalogMean,'analog variance · DN²':q.unsaturatedAnalogVariance,'stored mean · e':q.storedMean,'ADC mean · DN':q.adcMean,'ADC variance · DN²':q.adcVariance,'overflow · 1':q.fullWellOverflowProbability,'ADC clip high · 1':q.adcUpperClippingProbability,'P query code · 1':r.queryCodeProbability}};
}
