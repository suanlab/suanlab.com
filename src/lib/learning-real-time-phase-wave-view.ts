/** Bounded drawing projections from owned validated observations only. */
import type {RealTimePhaseValidatedSuccess} from './learning-real-time-phase-protocol';
import {scalarFieldProjection as project} from './learning-scalar-field-view';
export type RealTimeRange=Readonly<{lower:string;upper:string}>;
export const realTimeRangeMidpoint=(r:RealTimeRange):number|null=>{
  const a=project(r.lower),b=project(r.upper);return a===null||b===null?null:a/2+b/2;
};
export const realTimeUnitPhase=(p:Readonly<{real:RealTimeRange;imaginary:RealTimeRange}>)=>({real:realTimeRangeMidpoint(p.real),imaginary:realTimeRangeMidpoint(p.imaginary)});
export function realTimeChainDrawing(response:RealTimePhaseValidatedSuccess,selectedMode:number){
  const r=response.result;if(r.kind!=='chain')return null;
  const n=r.input.intervals,d=n-1,k=Number.isInteger(selectedMode)?Math.max(1,Math.min(d,selectedMode)):1;
  const status=k<=r.exact.inertia.negative?'negative':k<=r.exact.inertia.negative+r.exact.inertia.zero?'zero':'positive';
  const diagonal=project(r.exact.hessian.diagonal),off=project(r.exact.hessian.offDiagonal);
  const approximate=diagonal===null||off===null?null:diagonal+2*off*Math.cos(k*Math.PI/n);
  const agrees=approximate!==null&&(status==='zero'?false:status==='positive'?approximate>0:approximate<0);
  const mode=Array.from({length:n+1},(_,i)=>({x:i/n,y:i===0||i===n?0:Math.sin(i*k*Math.PI/n)}));
  const path=r.exact.stationary?[r.input.start,...r.exact.stationary,r.input.end].map((value,i)=>({x:i/n,y:project(value)})):null;
  return {mode,k,status,diagnosticEigenvalue:agrees?approximate:null,path,pathAvailable:!!path&&path.every(p=>p.y!==null),
    stationaryStatus:r.exact.stationaryStatus,unregulatedPhase:realTimeUnitPhase(r.wave.unregulated.unitPhase),regulatedPhase:r.wave.regulated?realTimeUnitPhase(r.wave.regulated.unitPhase):null};
}
export function realTimePacketDrawing(response:RealTimePhaseValidatedSuccess){
  const r=response.result;if(r.kind!=='packet')return null;
  const sigma=realTimeRangeMidpoint(r.wave.standardDeviation),normal=realTimeRangeMidpoint(r.wave.normalization);
  const points=r.wave.samples.map((sample,index)=>{
    const density=realTimeRangeMidpoint(sample.density),re=realTimeRangeMidpoint(sample.value.real),im=realTimeRangeMidpoint(sample.value.imaginary);
    const densityFactor=2**(sample.density.scalePowerOfTwo+r.wave.standardDeviation.scalePowerOfTwo);
    const waveFactor=2**(sample.value.scalePowerOfTwo-r.wave.normalization.scalePowerOfTwo);
    const value=(n:number|null)=>n===null||normal===null||normal<=0?null:n/normal*waveFactor;
    const scaledDensity=sigma===null||density===null?null:sigma*density*densityFactor;
    return {index,x:sample.standardDeviations,density:scaledDensity!==null&&Number.isFinite(scaledDensity)?scaledDensity:null,real:value(re),imaginary:value(im),phase:realTimeUnitPhase(sample.unitPhase)};
  });
  return {points,center:r.exact.center,variance:r.exact.finalCovariance[0][0],normalization:r.wave.normalization,standardDeviation:r.wave.standardDeviation};
}
/** Explicit binary64 continuous-oscillator diagnostic; never classifies a
 * finite caustic or supplies a certificate for the splitting error. */
export function realTimeContinuumDiagnostic(response:RealTimePhaseValidatedSuccess){
  const r=response.result;if(r.kind==='canonical-fold')return null;
  const time=project(r.input.time),mass=project(r.input.mass),omega=project(r.input.omegaSquared);
  if(time===null||mass===null||omega===null)return null;
  if(omega===0)return [[1,time/mass],[0,1]];
  const w=Math.sqrt(Math.abs(omega)),z=w*time,diagonal=omega>0?Math.cos(z):Math.cosh(z),sine=omega>0?Math.sin(z):Math.sinh(z);
  return [[diagonal,sine/(mass*w)],[(omega>0?-1:1)*mass*w*sine,diagonal]];
}
/** A fixed physical-control sweep, not an alternative input domain. Current u
 * outside this window is retained separately by its own validated response.
 */
export const realTimeFoldControls=Object.freeze(Array.from({length:33},(_,i)=>{const n=i-16;return n%8===0?String(n/8):`${n}/8`;}));
