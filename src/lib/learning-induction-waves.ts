/** Prescribed bounded induction fields and smooth transverse plane pulses.
 * Shared analytic identities are diagnostics, not independent verification.
 */
import {maxwellMedium,type MaxwellMedium} from './learning-maxwell';
import type {StaticVector} from './learning-electro-magnetostatics';

type Value=number|null;
function bound(x:number,lo:number,hi:number,name:string){
 if(!Number.isFinite(x)||x<lo||x>hi)throw new RangeError(`${name} outside [${lo},${hi}]`);
}
function point(v:StaticVector){
 if(!Array.isArray(v)||v.length!==3)throw new RangeError('Three Cartesian coordinates required');
 for(let i=0;i<3;i++)bound(v[i],-4,4,'Coordinate m');
}
// Conservative arithmetic: a lost nonzero intermediate is unavailable even
// when a different evaluation order might recover a final expression.
function mul(...xs:Value[]):Value{
 if(xs.includes(0))return 0;if(xs.some(x=>x===null))return null;
 let value=1;for(const x of xs){value*=x!;if(!Number.isFinite(value)||value===0)return null;}return value;
}
function sum(xs:Value[]):Value{
 if(xs.some(x=>x===null))return null;let total=0,correction=0;
 for(const x of xs){const y=x!-correction,next=total+y;correction=(next-total)-y;total=next;}
 return Number.isFinite(total)?total||0:null;
}
function divide(x:Value,y:Value):Value{
 if(x===null||y===null||y===0)return null;const value=x/y;
 return !Number.isFinite(value)||x!==0&&value===0?null:value||0;
}
const availability=(x:Value)=>x===null?'arithmetic-range' as const:'finite' as const;
const vector=(xs:Value[]):StaticVector|null=>xs.some(x=>x===null)?null:xs as StaticVector;

export interface UniformInduction extends MaxwellMedium{
 magneticOffset:number;magneticRate:number;magneticAcceleration:number;
}
export interface InductionLoop extends UniformInduction{
 initialWidth:number;widthRate:number;height:number;orientation?:1|-1;
}
function inductionParameters(s:UniformInduction){
 const medium=maxwellMedium(s);
 bound(s.magneticOffset,-4,4,'Magnetic offset T');bound(s.magneticRate,-4,4,'Magnetic rate T/s');
 bound(s.magneticAcceleration,-4,4,'Magnetic acceleration T/s²');return medium;
}
export function uniformInductionState(s:UniformInduction,position:StaticVector,time:number){
 const medium=inductionParameters(s);point(position);bound(time,-2,2,'Time s');
 const magnetic=sum([s.magneticOffset,mul(s.magneticRate,time),mul(.5,s.magneticAcceleration,time,time)]);
 const rate=sum([s.magneticRate,mul(s.magneticAcceleration,time)]);
 const electric=vector([mul(.5,rate,position[1]),mul(-.5,rate,position[0]),0]);
 const electricTime=vector([mul(.5,s.magneticAcceleration,position[1]),mul(-.5,s.magneticAcceleration,position[0]),0]);
 const current=vector([mul(-.5,medium.permittivity,s.magneticAcceleration,position[1]),mul(.5,medium.permittivity,s.magneticAcceleration,position[0]),0]);
 const displacement=vector([mul(.5,medium.permittivity,s.magneticAcceleration,position[1]),mul(-.5,medium.permittivity,s.magneticAcceleration,position[0]),0]);
 const totalCurrent=current===null||displacement===null?null:vector(current.map((x,i)=>sum([x,displacement[i]])));
 return{...medium,position:[...position] as StaticVector,time,magneticValue:magnetic,magneticRate:rate,
  electricField:electric,magneticField:vector([0,0,magnetic]),electricFieldStatus:electric===null?'arithmetic-range' as const:'finite' as const,magneticFieldStatus:availability(magnetic),
  electricTimeDerivative:electricTime,magneticTimeDerivative:vector([0,0,rate]),electricCurl:vector([0,0,mul(-1,rate)]),magneticCurl:[0,0,0] as StaticVector,
  electricDivergence:0,magneticDivergence:0,chargeDensity:0,currentDivergence:0,chargeTimeDerivative:0,
  conductionCurrent:current,displacementCurrent:displacement,totalCurrent,faradayResidualZ:sum([mul(-1,rate),rate]),
  ampereResidual:totalCurrent===null?null:vector(totalCurrent.map(x=>mul(-medium.permeability,x))),
  modelClass:'bounded-prescribed-uniform-magnetic-family' as const,diagnosticMeaning:'analytic-balance-requires-prescribed-boundary-sources' as const};
}
function loopParameters(s:InductionLoop){
 inductionParameters(s);bound(s.initialWidth,1,2,'Initial width m');bound(s.widthRate,-.25,.25,'Width rate m/s');bound(s.height,0,4,'Height m');
 const orientation=s.orientation===undefined?1:s.orientation;
 if(orientation!==1&&orientation!==-1)throw new RangeError('Loop orientation must be ±1');return orientation;
}
export function inductionMovingLoop(s:InductionLoop,time:number){
 const orientation=loopParameters(s),q=uniformInductionState(s,[0,0,0],time),width=sum([s.initialWidth,mul(s.widthRate,time)]);
 // These parameter bounds guarantee .5<=width<=2.5 throughout [-2,2]s.
 const area=mul(orientation,width,s.height),flux=mul(q.magneticValue,area);
 const electric=mul(-orientation,q.magneticRate,width,s.height),motional=mul(-orientation,q.magneticValue,s.widthRate,s.height),emf=sum([electric,motional]);
 const fluxRate=sum([mul(orientation,q.magneticRate,width,s.height),mul(orientation,q.magneticValue,s.widthRate,s.height)]);
 const forwardVertices:StaticVector[]|null=width===null?null:[[0,0,0],[width,0,0],[width,s.height,0],[0,s.height,0]],vertices=forwardVertices===null?null:orientation===1?forwardVertices:[forwardVertices[0],forwardVertices[3],forwardVertices[2],forwardVertices[1]];
 const half=mul(-.5,q.magneticRate,width,s.height),electricForward:Value[]=[0,half,half,0],motionForward:Value[]=[0,mul(-1,q.magneticValue,s.widthRate,s.height),0,0];
 const orient=(edges:Value[])=>orientation===1?edges:[...edges].reverse().map(x=>mul(-1,x));
 const mean=divide(flux,area);
 return{time,width,widthStatus:availability(width),height:s.height,widthRate:s.widthRate,orientation,vertices,orientedAreaZ:area,magneticFlux:flux,magneticFluxStatus:availability(flux),
  electricEdgeIntegrals:orient(electricForward),motionalEdgeIntegrals:orient(motionForward),electricCirculation:electric,motionalCirculation:motional,totalEmf:emf,
  fixedSurfaceNegativeFluxRate:mul(-orientation,q.magneticRate,width,s.height),movingSurfaceFluxRate:fluxRate,negativeMovingFluxRate:mul(-1,fluxRate),
  balanceResidual:sum([emf,fluxRate]),meanBzFromFlux:mean,normalizationStatus:s.height===0?'zero-area' as const:mean===null?'arithmetic-range' as const:'finite' as const,
  velocityModel:'continuous-stretch-vx=widthRate*x/width' as const,diagnosticMeaning:'prescribed-test-loop-not-solved-conductor-dynamics' as const};
}
export function inductionLoopInterval(s:InductionLoop,startTime:number,endTime:number){
 const initial=inductionMovingLoop(s,startTime),final=inductionMovingLoop(s,endTime),dt=endTime-startTime,tSum=startTime+endTime,tSquares=startTime*startTime+startTime*endTime+endTime*endTime;
 const {magneticOffset:b0,magneticRate:b1,magneticAcceleration:b2,initialWidth:L,widthRate:u,height:H}=s,o=initial.orientation;
 const p1=sum([mul(b0,u),mul(b1,L)]),p2=sum([mul(b1,u),mul(.5,b2,L)]),p3=mul(.5,b2,u);
 const change=mul(o,H,dt,sum([p1,mul(p2,tSum),mul(p3,tSquares)]));
 const electric=mul(-o,H,dt,sum([mul(b1,L),mul(.5,sum([mul(b1,u),mul(b2,L)]),tSum),mul(1/3,b2,u,tSquares)]));
 const motional=mul(-o,H,u,dt,sum([b0,mul(.5,b1,tSum),mul(1/6,b2,tSquares)])),integrated=sum([electric,motional]);
 return{initial,final,startTime,endTime,stableFluxChange:change,naiveEndpointFluxChange:sum([final.magneticFlux,mul(-1,initial.magneticFlux)]),
  integratedElectricCirculation:electric,integratedMotionalCirculation:motional,integratedEmf:integrated,balanceResidual:sum([integrated,change]),
  diagnosticMeaning:'analytic-polynomial-interval-balance-not-independent-integration' as const};
}

export interface TransverseGaussianPulse extends MaxwellMedium{
 amplitude:number;width:number;rightFraction:number;
 magneticMode?:'compatible'|'zero-magnetic-negative-control';
}
function gaussian(amplitude:number,u:Value,width:number):Value{
 if(amplitude===0)return 0;if(u===null)return null;
 const value=Math.sign(amplitude)*Math.exp(Math.log(Math.abs(amplitude))-.5*(u/width)**2);
 return value===0||!Number.isFinite(value)?null:value;
}
export function transverseGaussianState(s:TransverseGaussianPulse,z:number,time:number){
 const medium=maxwellMedium(s),mode=s.magneticMode===undefined?'compatible':s.magneticMode;
 bound(s.amplitude,-4,4,'Pulse amplitude V/m');bound(s.width,.1,4,'Pulse width m');bound(s.rightFraction,0,1,'Right-moving fraction');bound(z,-4,4,'Axial position m');bound(time,-4,4,'Time s');
 if(mode!=='compatible'&&mode!=='zero-magnetic-negative-control')throw new RangeError('Unknown pulse magnetic mode');
 const shift=mul(medium.speed,time);if(shift!==null&&Math.abs(shift)>8)throw new RangeError('Pulse displacement exceeds supported ±8m');
 const plus=gaussian(s.amplitude,sum([z,mul(-1,shift)]),s.width),minus=gaussian(s.amplitude,sum([z,shift]),s.width),width2=s.width*s.width;
 const right=mul(s.rightFraction,plus),left=mul(1-s.rightFraction,minus),electric=sum([right,left]),naiveCB=sum([right,mul(-1,left)]);
 const d=mul(2,z,shift,1/width2);
 // Cancel nearly equal Gaussian profiles before evaluating their difference.
 // Never evaluate exp()*sinh() with large opposite exponents.
 const difference=s.amplitude===0||z===0||time===0?0:d===null?null:d>=0?mul(plus,-Math.expm1(-d)):mul(minus,Math.expm1(d));
 const compatibleCB=s.rightFraction===1?plus:s.rightFraction===0?mul(-1,minus):mul(.5,sum([mul(2*s.rightFraction-1,sum([plus,minus])),difference]));
 const compatibleB=divide(compatibleCB,medium.speed),cz=divide(mul(-1,sum([mul(z,compatibleCB),mul(-1,shift,electric)])),width2);
 const ez=divide(mul(-1,sum([mul(z,electric),mul(-1,shift,compatibleCB)])),width2),et=mul(-medium.speed,cz),bt=divide(mul(-1,ez),1);
 const ezz=sum([mul(sum([mul(sum([z*z,mul(shift,shift)]),1/(width2*width2)),-1/width2]),electric),mul(-2,z,shift,1/(width2*width2),compatibleCB)]);
 const ett=mul(medium.speed,medium.speed,ezz),magnetic=mode==='compatible'?compatibleB:0,actualBz=mode==='compatible'?divide(cz,medium.speed):0,actualBt=mode==='compatible'?bt:0;
 const expectedEt=divide(et,medium.speed*medium.speed),admissible=mode==='compatible'||s.amplitude===0;
 return{...medium,z,time,width:s.width,amplitude:s.amplitude,rightFraction:s.rightFraction,magneticMode:mode,displacement:shift,
  rightElectric:right,leftElectric:left,electricValue:electric,magneticValue:magnetic,compatibleMagneticValue:compatibleB,stableCB:compatibleCB,naiveCB,
  electricField:vector([electric,0,0]),magneticField:vector([0,magnetic,0]),electricFieldStatus:availability(electric),magneticFieldStatus:availability(magnetic),
  electricSpatialDerivative:ez,magneticSpatialDerivative:actualBz,electricTimeDerivative:et,magneticTimeDerivative:actualBt,
  electricSecondSpatialDerivative:ezz,electricSecondTimeDerivative:ett,electricWaveResidual:sum([ett,mul(-medium.speed,medium.speed,ezz)]),
  electricDivergence:0,magneticDivergence:0,chargeDensity:0,conductionCurrent:[0,0,0] as StaticVector,
  faradayResidualY:sum([ez,actualBt]),ampereResidualX:sum([mul(-1,actualBz),mul(-1,expectedEt)]),admissibleSourceFreeSolution:admissible,
  solutionClass:admissible?'compatible-transverse-plane-pulse' as const:'negative-control-scalar-wave-with-zero-B' as const,
  diagnosticMeaning:'analytic-shared-family-derivatives-not-independent-verification' as const};
}
