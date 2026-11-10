/** Smooth local classical potentials. Analytic family identities are not independent certificates.
 * Coordinates xi=x/L and tau=ct/L are dimensionless; SI observations carry separate scales.
 */
import {maxwellMedium} from './learning-maxwell';
import type {StaticVector} from './learning-electro-magnetostatics';
type Value=number|null;
const MIN_NORMAL=2**-1022;
function bound(x:number,lo:number,hi:number,label:string){if(!Number.isFinite(x)||x<lo||x>hi)throw new RangeError(`${label} outside [${lo},${hi}]`);}
function mul(...xs:Value[]):Value{if(xs.includes(0))return 0;if(xs.includes(null))return null;let v=1;for(const x of xs){v*=x!;if(!Number.isFinite(v)||v===0)return null;}return v||0;}
function add(...xs:Value[]):Value{if(xs.includes(null))return null;let s=0,e=0;for(const x of xs){const t=s+x!;e+=Math.abs(s)>=Math.abs(x!)?(s-t)+x!:(x!-t)+s;s=t;}const r=s+e;return Number.isFinite(r)?r||0:null;}
const neg=(x:Value)=>mul(-1,x);
const vec=(v:Value[]):StaticVector|null=>v.includes(null)?null:v as StaticVector;
const mapped=(v:StaticVector|null,s:Value)=>s===0?[0,0,0] as StaticVector:v===null?null:vec(v.map(x=>mul(x,s)));
function point(q:StaticVector,tau:number){if(!Array.isArray(q)||q.length!==3)throw new RangeError('Three normalized coordinates required');for(let i=0;i<3;i++)bound(q[i],-4,4,'Normalized coordinate');bound(tau,-4,4,'Normalized time');}
/** Compensated product contrast. Conservatively unavailable if products cannot support double-double error terms.
 * Identical products cancel structurally, including products too small to observe separately.
 */
export function gaugeProductDifference(a:number,b:number,c:number,d:number):Value{
 for(const x of[a,b,c,d])bound(x,-16,16,'Product factor');
 if(a===c&&b===d||a===d&&b===c)return 0;
 const p=mul(a,b),q=mul(c,d);if(p===null||q===null)return null;
 if(p!==0&&Math.abs(p)<MIN_NORMAL*2**106||q!==0&&Math.abs(q)<MIN_NORMAL*2**106)return null;
 const split=(x:number)=>{const z=134217729*x,h=z-(z-x);return[h,x-h];};
 const error=(x:number,y:number,p:number)=>{const[h,l]=split(x),[j,m]=split(y);return((h*j-p)+h*m+l*j)+l*m;};
 const e=error(a,b,p)-error(c,d,q),s=p-q,v=s-p,tail=(p-(s-v))+(-q-v),r=s+(tail+e);
 return Number.isFinite(r)?r||0:null;
}
export interface GaugeScales{length?:number;potential?:number;permittivity?:number;permeability?:number;}
export function gaugeScales(s:GaugeScales={}){
 const length=s.length??1,potential=s.potential??1;bound(length,.25,4,'Length scale m');bound(potential,0,4,'Potential scale V');const m=maxwellMedium(s),c=m.speed;
 return{...m,length,potential,normalizedTimeSeconds:length/c,vectorPotential:mul(potential,1/c),gaugeFunction:mul(potential,length,1/c),electric:mul(potential,1/length),magnetic:mul(potential,1/c,1/length),lorenz:mul(potential,1/c,1/length),charge:mul(m.permittivity,potential,1/length,1/length),current:mul(m.permittivity,c,potential,1/length,1/length),gauss:mul(potential,1/length,1/length),ampere:mul(potential,1/c,1/length,1/length)};
}
export type GaugeTransformMode='coordinated'|'vector-only'|'scalar-only'|'original';
export interface TimeGauge extends GaugeScales{magneticConstant:number;magneticRate:number;magneticAcceleration:number;gaugeAmplitude:number;gaugeConstant:number;gaugeRate:number;gaugeAcceleration:number;mode?:GaugeTransformMode;}
function gaugeParameters(s:TimeGauge){for(const k of['magneticConstant','magneticRate','magneticAcceleration','gaugeAmplitude','gaugeConstant','gaugeRate','gaugeAcceleration']as const)bound(s[k],-4,4,k);const mode=s.mode??'coordinated';if(!['coordinated','vector-only','scalar-only','original'].includes(mode))throw new RangeError('Unknown gauge transformation');return{scales:gaugeScales(s),mode,A:mode==='coordinated'||mode==='vector-only',V:mode==='coordinated'||mode==='scalar-only'};}
export function timeGaugeState(s:TimeGauge,q:StaticVector,tau:number){
 const p=gaugeParameters(s);point(q,tau);const[x,y,z]=q,b=add(s.magneticConstant,mul(s.magneticRate,tau),mul(.5,s.magneticAcceleration,tau,tau)),bt=add(s.magneticRate,mul(s.magneticAcceleration,tau));
 const g=add(s.gaugeConstant,mul(s.gaugeRate,tau),mul(.5,s.gaugeAcceleration,tau,tau)),gt=add(s.gaugeRate,mul(s.gaugeAcceleration,tau)),alpha=s.gaugeAmplitude;
 const grad=[mul(alpha,g,y,z),mul(alpha,g,x,z),mul(alpha,g,x,y)],gradt=[mul(alpha,gt,y,z),mul(alpha,gt,x,z),mul(alpha,gt,x,y)],gradtt=[mul(alpha,s.gaugeAcceleration,y,z),mul(alpha,s.gaugeAcceleration,x,z),mul(alpha,s.gaugeAcceleration,x,y)];
 const baseA=[mul(-.5,b,y),mul(.5,b,x),0],baseE=[mul(.5,bt,y),mul(-.5,bt,x),0],baseJ=[mul(-.5,s.magneticAcceleration,y),mul(.5,s.magneticAcceleration,x),0];
 const a=vec(baseA.map((v,i)=>add(v,p.A?grad[i]:0))),v=p.V?mul(-alpha,x,y,z,gt):0;
 const scalarE=vec(gradt.map(v=>p.V?v:0)),vectorE=vec(baseE.map((v,i)=>add(v,p.A?neg(gradt[i]):0)));
 const factor=Number(p.V)-Number(p.A),electric=vec(baseE.map((v,i)=>add(v,mul(factor,gradt[i])))),magnetic=vec([0,0,b]);
 const naiveElectric=scalarE===null||vectorE===null?null:vec(scalarE.map((v,i)=>add(v,vectorE[i])));
 const lorenz=p.V?mul(-alpha,x,y,z,s.gaugeAcceleration):0,current=vec(baseJ.map((v,i)=>add(v,mul(-factor,gradtt[i]))));
 const normalized={vectorPotential:a,scalarPotential:v,chi:mul(alpha,x,y,z,g),g,gt,magneticCoefficient:b,magneticRate:bt,gaugeGradient:vec(grad),electricScalarPart:scalarE,electricVectorPart:vectorE,electric,naiveElectric,magnetic,lorenz,charge:0,current,currentChange:vec(gradtt.map(v=>mul(-factor,v))),baseElectric:vec(baseE),baseCurrent:vec(baseJ),faradayResidual:[0,0,0] as StaticVector,magneticDivergence:0};
 return{mode:p.mode,position:[...q]as StaticVector,tau,scales:p.scales,normalized,
  si:{position:q.map(x=>x*p.scales.length)as StaticVector,time:mul(tau,p.scales.normalizedTimeSeconds),vectorPotential:mapped(a,p.scales.vectorPotential),scalarPotential:mul(v,p.scales.potential),chi:mul(normalized.chi,p.scales.gaugeFunction),electricScalarPart:mapped(scalarE,p.scales.electric),electricVectorPart:mapped(vectorE,p.scales.electric),electric:mapped(electric,p.scales.electric),magnetic:mapped(magnetic,p.scales.magnetic),lorenz:mul(lorenz,p.scales.lorenz),charge:0,current:mapped(current,p.scales.current)},
  physicalFieldFamilyUnchanged:p.scales.potential===0||factor===0||alpha===0||s.gaugeRate===0&&s.gaugeAcceleration===0,
  sourceFamilyUnchanged:p.scales.potential===0||factor===0||alpha===0||s.gaugeAcceleration===0,
  lorenzFamilySatisfied:p.scales.potential===0||!p.V||alpha===0||s.gaugeAcceleration===0,
  diagnosticMeaning:'analytic-local-family-not-independent-certification-or-global-boundary-solution' as const};
}
export function timeGaugeLoop(s:TimeGauge,tau:number,z:number,width:number,height:number){
 for(const v of[width,height])bound(v,0,4,'Normalized loop length');const r=timeGaugeState(s,[0,0,z],tau),x=width/2,y=height/2,area=mul(width,height);
 if(width!==0&&x===0||height!==0&&y===0)return{area,vertices:null,edgeIntegrals:null,stableCirculation:null,naiveCirculation:null,magneticFlux:null,electricCirculation:null,status:'geometry-arithmetic-range' as const};
 const positions:StaticVector[]=[[0,-y,z],[x,0,z],[0,y,z],[-x,0,z]],lengths=[width,height,-width,-height],indices=[0,1,0,1];
 const edges=positions.map((q,i)=>mul(timeGaugeState(s,q,tau).normalized.vectorPotential?.[indices[i]]??null,lengths[i]));
 const stable=mul(r.normalized.magneticCoefficient,area),ef=mul(-1,r.normalized.magneticRate,area),fluxScale=mul(r.scales.potential,r.scales.length,1/r.scales.speed);
 return{area,vertices:[[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]],edgeIntegrals:edges,stableCirculation:stable,naiveCirculation:edges.includes(null)?null:edges.reduce<number>((sum,v)=>sum+v!,0),magneticFlux:mul(stable,fluxScale),electricCirculation:mul(ef,r.scales.potential),status:area===null?'arithmetic-range' as const:area===0?'zero-area' as const:'finite' as const};
}
export interface LorenzMode extends GaugeScales{scalarAmplitude:number;vectorAmplitude:number;wavenumber:number;frequency:number;}
function harmonicParameters(s:LorenzMode){bound(s.scalarAmplitude,-4,4,'Scalar amplitude');bound(s.vectorAmplitude,-4,4,'Vector amplitude');bound(s.wavenumber,0,16,'Normalized wavenumber');bound(s.frequency,0,16,'Normalized frequency');return gaugeScales(s);}
function sourceObservation(scales:ReturnType<typeof gaugeScales>,rho:Value,j:Value,gauss:Value,ampere:Value){return{charge:mul(rho,scales.charge),current:mapped(vec([0,0,j]),scales.current),gaussResidual:mul(gauss,scales.gauss),ampereResidual:mapped(vec([0,0,ampere]),scales.ampere)};}
export function lorenzHarmonicState(s:LorenzMode,z:number,tau:number){
 const scales=harmonicParameters(s);point([0,0,z],tau);const k=s.wavenumber,w=s.frequency,v=s.scalarAmplitude,a=s.vectorAmplitude;
 const phase=add(mul(k,z),neg(mul(w,tau))),sn=phase===null?null:Math.sin(phase),cs=phase===null?null:Math.cos(phase);
 const dispersion=mul(k-w,k+w),naiveDispersion=k*k-w*w,Q=gaugeProductDifference(k,v,w,a),C=gaugeProductDifference(w,v,k,a);
 const electric=mul(Q,sn),G=mul(C,sn),Gt=mul(-w,C,cs),Gz=mul(k,C,cs),rho=mul(dispersion,v,cs),j=mul(dispersion,a,cs),fieldRho=mul(k,Q,cs),fieldJ=mul(w,Q,cs),continuity=mul(dispersion,C,sn);
 const normalized={phase,scalarPotential:mul(v,cs),vectorPotential:vec([0,0,mul(a,cs)]),electric:vec([0,0,electric]),magnetic:[0,0,0]as StaticVector,electricCoefficient:Q,lorenzCoefficient:C,dispersion,naiveDispersion,naiveElectricCoefficient:k*v-w*a,naiveLorenzCoefficient:w*v-k*a,lorenz:G,lorenzTimeDerivative:Gt,lorenzGradient:vec([0,0,Gz]),assignedCharge:rho,assignedCurrent:j,fieldCharge:fieldRho,fieldCurrent:fieldJ,sourceContinuity:continuity,lorenzWaveOperator:neg(continuity),gaussResidual:neg(Gt),ampereResidual:Gz,naiveGaussResidual:fieldRho===null||rho===null?null:fieldRho-rho,naiveAmpereResidual:fieldJ===null||j===null?null:fieldJ-j,initialLorenz:mul(C,Math.sin(k*z)),initialLorenzTimeDerivative:mul(-w,C,Math.cos(k*z)),scalarWaveEquationResidual:0,vectorWaveEquationResidual:0};
 const coefficientZero=C===null?null:C===0;
 return{family:'traveling-longitudinal-harmonic' as const,z,tau,scales,normalized,
  si:{time:mul(tau,scales.normalizedTimeSeconds),z:z*scales.length,angularFrequency:mul(w,scales.speed,1/scales.length),wavenumber:k/scales.length,scalarPotential:mul(normalized.scalarPotential,scales.potential),vectorPotential:mapped(normalized.vectorPotential,scales.vectorPotential),electric:mapped(normalized.electric,scales.electric),magnetic:[0,0,0]as StaticVector,lorenz:mul(G,scales.lorenz),assigned:sourceObservation(scales,rho,j,neg(Gt),Gz),fieldImplied:sourceObservation(scales,fieldRho,fieldJ,0,0)},
  lorenzFamilySatisfied:scales.potential===0?true:coefficientZero,assignedSourceFamilyConserved:scales.potential===0||k===w?true:coefficientZero,assignedSourceMaxwellFamilyCompatible:scales.potential===0?true:coefficientZero,
  diagnosticMeaning:'separated-wave-equation-identities-do-not-certify-Maxwell-or-initial-boundary-constraints' as const};
}
export interface InitialLorenz extends GaugeScales{amplitude:number;wavenumber:number;}
export function lorenzInitialConstraintState(s:InitialLorenz,z:number,tau:number){
 bound(s.amplitude,-4,4,'Initial amplitude');bound(s.wavenumber,0,16,'Normalized wavenumber');const scales=gaugeScales(s);point([0,0,z],tau);const a=s.amplitude,k=s.wavenumber,sz=Math.sin(k*z),cz=Math.cos(k*z),st=Math.sin(k*tau),ct=Math.cos(k*tau);
 const az=mul(a,sz,st),ez=mul(-a,k,sz,ct),G=mul(a,k,cz,st),Gt=mul(a,k,k,cz,ct),Gz=mul(-a,k,k,sz,st),rho=neg(Gt),j=Gz;
 const normalized={scalarPotential:0,vectorPotential:vec([0,0,az]),electric:vec([0,0,ez]),magnetic:[0,0,0]as StaticVector,lorenz:G,lorenzTimeDerivative:Gt,lorenzGradient:vec([0,0,Gz]),assignedCharge:0,assignedCurrent:0,fieldCharge:rho,fieldCurrent:j,gaussResidual:rho,ampereResidual:j,initialLorenz:0,initialLorenzTimeDerivative:mul(a,k,k,cz),initialElectric:mul(-a,k,sz),scalarWaveEquationResidual:0,vectorWaveEquationResidual:0,sourceContinuity:0};
 return{family:'zero-initial-g-nonzero-initial-dt-g' as const,z,tau,scales,normalized,si:{time:mul(tau,scales.normalizedTimeSeconds),z:z*scales.length,vectorPotential:mapped(normalized.vectorPotential,scales.vectorPotential),electric:mapped(normalized.electric,scales.electric),magnetic:[0,0,0]as StaticVector,lorenz:mul(G,scales.lorenz),assigned:sourceObservation(scales,0,0,rho,j),fieldImplied:sourceObservation(scales,rho,j,0,0)},lorenzFamilySatisfied:scales.potential===0||a===0||k===0,assignedSourceFamilyConserved:true,assignedSourceMaxwellFamilyCompatible:scales.potential===0||a===0||k===0,diagnosticMeaning:'initial-g-alone-insufficient-free-potential-wave-counterexample' as const};
}
/** Constant G need not violate Maxwell: v=offset*tau,a=0 gives E=B=0. */
export function constantLorenzOffsetState(s:GaugeScales,offset:number,tau:number){bound(offset,-4,4,'Lorenz offset');point([0,0,0],tau);const scales=gaugeScales(s);return{scales,normalized:{scalarPotential:mul(offset,tau),lorenz:offset,electric:[0,0,0]as StaticVector,magnetic:[0,0,0]as StaticVector,charge:0,current:0,gaussResidual:0,ampereResidual:0},lorenzFamilySatisfied:scales.potential===0||offset===0,assignedSourceMaxwellFamilyCompatible:true};}
