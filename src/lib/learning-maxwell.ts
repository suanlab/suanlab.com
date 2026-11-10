/** Constant-coefficient SI teaching families,not an antenna/material solver.
 * Analytic identities are diagnostics; independent numerical checks are separate.
 */
import {teachingVacuumPermeability,type StaticVector}from './learning-electro-magnetostatics';
export const teachingMaxwellPermeability=teachingVacuumPermeability;
/** Derived from the rounded teaching mu and SI c; not an independent exact eps. */
export const teachingMaxwellPermittivity=1/(teachingMaxwellPermeability*299792458**2);
export interface MaxwellMedium{permittivity?:number;permeability?:number;}
export interface PlaneMaxwellWave extends MaxwellMedium{amplitude:number;wavenumber:number;direction?:1|-1;phaseOffset?:number;frequencyRatio?:number;}
export interface RadialMaxwellSource extends MaxwellMedium{initialGradient:number;gradientRate:number;chargeOffset?:number;}
type Value=number|null;
function bound(x:number,lo:number,hi:number,name:string){if(!Number.isFinite(x)||x<lo||x>hi)throw new RangeError(`${name} outside [${lo},${hi}]`);}
function position(v:StaticVector){if(!Array.isArray(v)||v.length!==3)throw new RangeError('Three Cartesian coordinates required');for(let i=0;i<3;i++)bound(v[i],-4,4,'Coordinate m');}
function product(...xs:Value[]):Value{if(xs.includes(0))return 0;if(xs.some(x=>x===null))return null;let p=1;for(const x of xs){p*=x!;if(!Number.isFinite(p)||p===0)return null;}return p;}
function sum(xs:Value[]):Value{if(xs.some(x=>x===null))return null;let s=0,c=0;for(const x of xs){const y=x!-c,t=s+y;c=(t-s)-y;s=t;}return Number.isFinite(s)?s||0:null;}
const state=(x:Value)=>x===null?'arithmetic-range' as const:'finite' as const;
const vector=(v:Value[]):StaticVector|null=>v.some(x=>x===null)?null:v as StaticVector;
function vectorSum(a:StaticVector|null,b:StaticVector|null){return a===null||b===null?null:a.map((x,i)=>sum([x,b[i]]))as StaticVector;}
function ratio(x:Value,denominator:Value):Value{if(x===null||denominator===null||denominator===0)return null;const q=x/denominator;return !Number.isFinite(q)||x!==0&&q===0?null:q||0;}
export function maxwellMedium(s:MaxwellMedium){
 const eps=s.permittivity===undefined?teachingMaxwellPermittivity:s.permittivity,mu=s.permeability===undefined?teachingMaxwellPermeability:s.permeability;
 bound(eps,1e-12,4,'Permittivity F/m');bound(mu,1e-12,4,'Permeability H/m');return{permittivity:eps,permeability:mu,speed:1/Math.sqrt(eps*mu)};
}
function waveParameters(s:PlaneMaxwellWave){
 const medium=maxwellMedium(s),direction=s.direction===undefined?1:s.direction,phaseOffset=s.phaseOffset===undefined?0:s.phaseOffset,frequencyRatio=s.frequencyRatio===undefined?1:s.frequencyRatio;
 bound(s.amplitude,-4,4,'Electric amplitude V/m');bound(s.wavenumber,0,16,'Wavenumber rad/m');bound(phaseOffset,-Math.PI,Math.PI,'Phase rad');bound(frequencyRatio,.5,1.5,'Frequency ratio');if(direction!==1&&direction!==-1)throw new RangeError('Propagation direction must be ±1');
 return{...medium,direction,phaseOffset,frequencyRatio,omega:product(frequencyRatio,medium.speed,s.wavenumber)};
}
export function maxwellPlaneWaveState(s:PlaneMaxwellWave,point:StaticVector,time:number){
 const p=waveParameters(s);position(point);bound(time,-4,4,'Time s');const temporalPhase=product(p.omega,time);
 // A declared finite teaching phase budget,not a hidden modulo or input clamp.
 if(temporalPhase!==null&&Math.abs(temporalPhase)>1e6)throw new RangeError('Temporal phase exceeds the supported 1e6 rad budget');
 const phase=sum([product(p.direction,s.wavenumber,point[2]),product(-1,temporalPhase),p.phaseOffset]);
 const cos=phase===null?null:Math.cos(phase),sin=phase===null?null:Math.sin(phase);
 const ex=product(s.amplitude,cos),by=product(p.direction,s.amplitude,1/p.speed,cos);
 const et=product(s.amplitude,p.omega,sin),bt=product(p.direction,s.amplitude,p.frequencyRatio,s.wavenumber,sin);
 const curlEy=product(-p.direction,s.amplitude,s.wavenumber,sin),curlBx=product(s.amplitude,s.wavenumber,1/p.speed,sin);
 const displacement=product(p.permittivity,et),ampereExpected=product(p.permeability,displacement);
 const electric=vector([ex,0,0]),magnetic=vector([0,by,0]);
 const admissible=p.frequencyRatio===1||s.wavenumber===0||s.amplitude===0;
 return{...p,point:[...point]as StaticVector,time,amplitude:s.amplitude,wavenumber:s.wavenumber,phaseParameter:phase,phaseStatus:state(phase),
  electricField:electric,magneticField:magnetic,electricFieldStatus:state(ex),magneticFieldStatus:state(by),
  electricTimeDerivative:vector([et,0,0]),magneticTimeDerivative:vector([0,bt,0]),electricCurl:vector([0,curlEy,0]),magneticCurl:vector([curlBx,0,0]),
  electricDivergence:0,magneticDivergence:0,chargeDensity:0,conductionCurrent:[0,0,0]as StaticVector,displacementCurrent:vector([displacement,0,0]),
  faradayResidualY:sum([curlEy,bt]),ampereResidualX:sum([curlBx,product(-1,ampereExpected)]),
  admissibleSourceFreeSolution:admissible,solutionClass:admissible?'source-free-plane-family' as const:'negative-control-detuned-family' as const,
  residualMeaning:'analytic-shared-family-diagnostic-not-independent-verification' as const};
}
// Analytic extension removes a coordinate quotient singularity. Small arguments
// use a Taylor evaluation; this does not replace a small physical width by zero.
function sinc(x:number){return x===0?1:Math.abs(x)<1e-4?1-x*x/6+x**4/120:Math.sin(x)/x;}
export function maxwellPlaneWaveRectangle(s:PlaneMaxwellWave,plane:'faraday-xz'|'ampere-yz',span:number,bottom:number,top:number,time:number){
 if(plane!=='faraday-xz'&&plane!=='ampere-yz')throw new RangeError('Unknown measuring plane');
 for(const x of[span,bottom,top])bound(x,-4,4,'Rectangle coordinate m');
 const middle=maxwellPlaneWaveState(s,[0,0,bottom+(top-bottom)/2],time),low=maxwellPlaneWaveState(s,[0,0,bottom],time),high=maxwellPlaneWaveState(s,[0,0,top],time),height=top-bottom;
 const sin=middle.phaseParameter===null?null:Math.sin(middle.phaseParameter),cos=middle.phaseParameter===null?null:Math.cos(middle.phaseParameter),factor=sinc(s.wavenumber*height/2),electric=plane==='faraday-xz';
 const electricIntegral=product(span,s.amplitude,height,cos,factor);
 const magneticIntegral=product(-span,middle.direction,s.amplitude,1/middle.speed,height,cos,factor);
 const line=electric?product(span,middle.direction,s.amplitude,s.wavenumber,height,sin,factor):product(span,s.amplitude,s.wavenumber,height,1/middle.speed,sin,factor);
 const electricFluxRate=product(span,s.amplitude,height,middle.omega,sin,factor);
 const negativeMagneticFluxRate=product(span,middle.direction,s.amplitude,middle.frequencyRatio,s.wavenumber,height,sin,factor);
 const displacement=product(middle.permittivity,electricFluxRate),rhs=electric?negativeMagneticFluxRate:product(middle.permeability,displacement);
 const area=product(electric?-1:1,span,height),field=electric?magneticIntegral:electricIntegral;
 const naive=electric?product(span,sum([low.electricField?.[0]??null,product(-1,high.electricField?.[0]??null)])):product(span,sum([low.magneticField?.[1]??null,product(-1,high.magneticField?.[1]??null)]));
 return{plane,span,bottom,top,time,orientedAreaComponent:area,normalAxis:electric?'y' as const:'x' as const,
  flux:field,fluxStatus:state(field),lineIntegral:line,lineIntegralStatus:state(line),naiveEndpointLineIntegral:naive,
  fluxTimeDerivative:electric?product(-1,negativeMagneticFluxRate):electricFluxRate,
  displacementCurrent:electric?null:displacement,conductionCurrent:0,rightHandSide:rhs,residual:sum([line,product(-1,rhs)]),
  meanCartesianField:ratio(field,area),normalizationStatus:span===0||height===0?'zero-area' as const:ratio(field,area)===null?'arithmetic-range' as const:'finite' as const,
  admissibleSourceFreeSolution:middle.admissibleSourceFreeSolution,residualMeaning:'analytic-identity-not-independent-flux-check' as const};
}
export function maxwellPlaneWaveBox(s:PlaneMaxwellWave,xLength:number,yLength:number,bottom:number,top:number,time:number){
 bound(xLength,0,4,'Box x length m');bound(yLength,0,4,'Box y length m');if(top<bottom)throw new RangeError('Box z bounds must be ordered');
 const e=maxwellPlaneWaveRectangle(s,'ampere-yz',yLength,bottom,top,time),b=maxwellPlaneWaveRectangle(s,'faraday-xz',xLength,bottom,top,time);
 const electricFaces=[product(-1,e.flux),e.flux,0,0,0,0],magneticFaces=[0,0,b.flux,product(-1,b.flux),0,0];
 return{faceOrder:['x-','x+','y-','y+','z-','z+'],electricFaceFluxes:electricFaces,magneticFaceFluxes:magneticFaces,
  electricClosedFlux:sum(electricFaces),magneticClosedFlux:sum(magneticFaces),enclosedCharge:0,
  geometryStatus:xLength===0||yLength===0||top===bottom?'zero-volume' as const:'finite' as const,
  diagnosticMeaning:'opposing-face-analytic-cancellation-not-independent-cubature' as const};
}
function radialParameters(s:RadialMaxwellSource){
 const medium=maxwellMedium(s),offset=s.chargeOffset===undefined?0:s.chargeOffset;
 bound(s.initialGradient,-4,4,'Initial gradient V/m²');bound(s.gradientRate,-4,4,'Gradient rate V/(m² s)');bound(offset,-4,4,'Charge offset C/m³');return{...medium,offset};
}
export function maxwellRadialState(s:RadialMaxwellSource,point:StaticVector,time:number){
 const p=radialParameters(s);position(point);bound(time,-4,4,'Time s');const gradient=sum([s.initialGradient,product(s.gradientRate,time)]);
 const electric=vector(point.map(x=>product(gradient,x))),et=vector(point.map(x=>product(s.gradientRate,x)));
 const current=vector(point.map(x=>product(-p.permittivity,s.gradientRate,x))),displacement=vector(point.map(x=>product(p.permittivity,s.gradientRate,x)));
 const rho=sum([product(3,p.permittivity,gradient),p.offset]),divE=product(3,gradient),rhoT=product(3,p.permittivity,s.gradientRate),divJ=product(-3,p.permittivity,s.gradientRate);
 const currentSum=vectorSum(current,displacement);
 return{...p,point:[...point]as StaticVector,time,initialGradient:s.initialGradient,gradientRate:s.gradientRate,gradient,
  electricField:electric,electricFieldStatus:electric===null?'arithmetic-range' as const:'finite' as const,magneticField:[0,0,0]as StaticVector,
  electricTimeDerivative:et,magneticTimeDerivative:[0,0,0]as StaticVector,electricCurl:[0,0,0]as StaticVector,magneticCurl:[0,0,0]as StaticVector,
  conductionCurrent:current,displacementCurrent:displacement,currentDensitySum:currentSum,
  ampereResidual:currentSum===null?null:vector(currentSum.map(x=>product(-p.permeability,x))),
  chargeDensity:rho,chargeDensityStatus:state(rho),chargeTimeDerivative:rhoT,currentDivergence:divJ,electricDivergence:divE,magneticDivergence:0,
  continuityResidual:sum([rhoT,divJ]),gaussConstraintResidual:ratio(-p.offset,p.permittivity),
  gaussResidualFromRoundedValues:sum([divE,product(-1,ratio(rho,p.permittivity))]),
  admissibleMaxwellSolution:p.offset===0,solutionClass:p.offset===0?'bounded-compatible-radial-source' as const:'negative-control-initial-gauss-offset' as const,
  residualMeaning:'analytic-balance-not-independent-verification' as const};
}
export function maxwellRadialSphere(s:RadialMaxwellSource,radius:number,time:number){
 bound(radius,0,4,'Control sphere radius m');const q=maxwellRadialState(s,[radius,0,0],time),volume=product(4*Math.PI/3,radius,radius,radius),area=product(4*Math.PI,radius,radius);
 const flux=product(4*Math.PI,q.gradient,radius,radius,radius),charge=sum([product(4*Math.PI,q.permittivity,q.gradient,radius,radius,radius),product(q.offset,volume)]);
 const outward=product(-4*Math.PI,q.permittivity,s.gradientRate,radius,radius,radius),rate=product(4*Math.PI,q.permittivity,s.gradientRate,radius,radius,radius),defect=product(-q.offset,volume);
 return{radius,time,volume,area,electricFlux:flux,electricFluxStatus:state(flux),magneticFlux:0,enclosedCharge:charge,enclosedChargeStatus:state(charge),
  outwardConductionCurrent:outward,outwardDisplacementCurrent:product(-1,outward),chargeRate:rate,continuityResidual:sum([rate,outward]),
  gaussChargeDefect:defect,gaussFluxDefect:ratio(defect,q.permittivity),gaussResidualFromRoundedValues:sum([flux,product(-1,ratio(charge,q.permittivity))]),
  meanRadialFromFlux:ratio(flux,area),normalizationStatus:radius===0?'zero-area' as const:ratio(flux,area)===null?'arithmetic-range' as const:'finite' as const,
  admissibleMaxwellSolution:q.admissibleMaxwellSolution,residualMeaning:'analytic-sphere-balance-not-independent-cubature' as const};
}
export function maxwellRadialBalance(s:RadialMaxwellSource,radius:number,startTime:number,endTime:number){
 const initial=maxwellRadialSphere(s,radius,startTime),final=maxwellRadialSphere(s,radius,endTime),medium=maxwellMedium(s),dt=endTime-startTime;
 const change=product(4*Math.PI,medium.permittivity,s.gradientRate,radius,radius,radius,dt),inflow=product(-1,initial.outwardConductionCurrent,dt);
 return{initial,final,startTime,endTime,chargeChange:change,naiveEndpointChange:sum([final.enclosedCharge,product(-1,initial.enclosedCharge)]),integratedNetInflow:inflow,
  residual:sum([change,product(-1,inflow)]),residualMeaning:'constant-rate-time-integral-not-independent-validation' as const};
}
