/** Constant positive coefficients, fixed measuring regions and prescribed fields.
 * Analytic balances diagnose these families; they are not independent certificates.
 */
import {uniformInductionState,transverseGaussianState,type UniformInduction,type TransverseGaussianPulse} from './learning-induction-waves';
import type {StaticVector} from './learning-electro-magnetostatics';

type Value=number|null;
function product(...xs:Value[]):Value{
 if(xs.includes(0))return 0;if(xs.includes(null))return null;
 let v=1;for(const x of xs){v*=x!;if(!Number.isFinite(v)||v===0)return null;}return v;
}
function sum(...xs:Value[]):Value{
 if(xs.includes(null))return null;let v=0,c=0;
 for(const x of xs){const y=x!-c,next=v+y;c=(next-v)-y;v=next;}return Number.isFinite(v)?v||0:null;
}
function quotient(a:Value,b:Value):Value{
 if(a===null||b===null||b===0)return null;const q=a/b;
 return !Number.isFinite(q)||a!==0&&q===0?null:q||0;
}
function bound(x:number,min:number,max:number,label:string){if(!Number.isFinite(x)||x<min||x>max)throw new RangeError(`${label} outside [${min},${max}]`);}
const status=(x:Value)=>x===null?'arithmetic-range' as const:'finite' as const;
const vector=(v:Value[]):StaticVector|null=>v.includes(null)?null:v as StaticVector;

export interface InductionEnergySource extends UniformInduction{
 currentMode?:'required-current'|'omitted-current-negative-control';
}
export function inductionEnergyState(s:InductionEnergySource,position:StaticVector,time:number){
 const q=uniformInductionState(s,position,time),mode=s.currentMode??'required-current';
 if(mode!=='required-current'&&mode!=='omitted-current-negative-control')throw new RangeError('Unknown energy current mode');
 const eps=q.permittivity,mu=q.permeability,r2=sum(product(position[0],position[0]),product(position[1],position[1]));
 const electricEnergy=product(eps,q.magneticRate,q.magneticRate,r2,1/8);
 const magneticEnergy=product(q.magneticValue,q.magneticValue,1/(2*mu));
 const electricStorageRate=product(eps,q.magneticRate,s.magneticAcceleration,r2,1/4);
 const magneticStorageRate=product(q.magneticValue,q.magneticRate,1/mu);
 const fieldToMatterPowerDensity=mode==='required-current'?product(-1,electricStorageRate):0;
 const poyntingDivergence=product(-1,magneticStorageRate);
 const poynting=vector([product(-.5,q.magneticValue,q.magneticRate,position[0],1/mu),product(-.5,q.magneticValue,q.magneticRate,position[1],1/mu),0]);
 const energyTimeDerivative=sum(magneticStorageRate,electricStorageRate);
 const groupedResidual=sum(sum(magneticStorageRate,poyntingDivergence),sum(electricStorageRate,fieldToMatterPowerDensity));
 // Deliberately retain ordinary summation as an explicitly labelled comparison.
 const naiveTotalResidual=energyTimeDerivative===null||poyntingDivergence===null||fieldToMatterPowerDensity===null?null:energyTimeDerivative+poyntingDivergence+fieldToMatterPowerDensity;
 return{...q,currentMode:mode,conductionCurrent:mode==='required-current'?q.conductionCurrent:[0,0,0] as StaticVector,
  totalCurrent:mode==='required-current'?q.totalCurrent:q.displacementCurrent,
  electricEnergyDensity:electricEnergy,magneticEnergyDensity:magneticEnergy,energyDensity:sum(electricEnergy,magneticEnergy),
  electricStorageRate,magneticStorageRate,energyTimeDerivative,poynting,poyntingDivergence,fieldToMatterPowerDensity,
  groupedResidual,naiveTotalResidual,energyStatus:status(sum(electricEnergy,magneticEnergy)),
  ampereResidual:mode==='required-current'?q.ampereResidual:q.displacementCurrent===null?null:vector(q.displacementCurrent.map(x=>product(-mu,x))),
  maxwellFamilyCompatible:mode==='required-current'||s.magneticAcceleration===0,
  diagnosticMeaning:'component-grouped-analytic-energy-balance-not-independent-verification' as const};
}

export function inductionCylinderEnergy(s:InductionEnergySource,radius:number,height:number,time:number){
 bound(radius,0,4,'Cylinder radius m');bound(height,0,4,'Cylinder height m');
 const q=inductionEnergyState(s,[0,0,0],time),eps=q.permittivity,mu=q.permeability;
 const volume=product(Math.PI,radius,radius,height);
 const electricEnergy=product(eps,Math.PI,height,radius,radius,radius,radius,q.magneticRate,q.magneticRate,1/16);
 const magneticEnergy=product(volume,q.magneticValue,q.magneticValue,1/(2*mu));
 const electricStorageRate=product(eps,Math.PI,height,radius,radius,radius,radius,q.magneticRate,s.magneticAcceleration,1/8);
 const magneticStorageRate=product(volume,q.magneticValue,q.magneticRate,1/mu);
 const outwardPower=product(-1,magneticStorageRate);
 const fieldToMatterPower=q.currentMode==='required-current'?product(-1,electricStorageRate):0;
 const energyRate=sum(magneticStorageRate,electricStorageRate);
 return{radius,height,time,volume,electricEnergy,magneticEnergy,totalEnergy:sum(electricEnergy,magneticEnergy),electricStorageRate,magneticStorageRate,
  energyRate,outwardPower,fieldToMatterPower,groupedResidual:sum(sum(magneticStorageRate,outwardPower),sum(electricStorageRate,fieldToMatterPower)),
  naiveTotalResidual:energyRate===null||outwardPower===null||fieldToMatterPower===null?null:energyRate+outwardPower+fieldToMatterPower,
  regionType:'fixed-cylinder-outward-normal' as const,diagnosticMeaning:'analytic-volume-and-surface-formulas-not-independent-quadrature' as const};
}

export function transversePulseEnergy(s:TransverseGaussianPulse,z:number,time:number){
 const q=transverseGaussianState(s,z,time),eps=q.permittivity,mu=q.permeability;
 const electricEnergy=product(.5,eps,q.electricValue,q.electricValue),magneticEnergy=product(.5,q.magneticValue,q.magneticValue,1/mu);
 const energyDensity=sum(electricEnergy,magneticEnergy);
 // Use the stable cB, not subtraction of already rounded squared profiles.
 const poyntingZ=q.magneticMode==='compatible'?product(eps,q.speed,q.electricValue,q.stableCB):0;
 const naivePoyntingZ=product(q.electricValue,q.magneticValue,1/mu);
 const electricStorageRate=product(eps,q.electricValue,q.electricTimeDerivative);
 const magneticStorageRate=product(q.magneticValue,q.magneticTimeDerivative,1/mu);
 const poyntingDivergence=q.magneticMode==='compatible'?product(sum(product(q.electricSpatialDerivative,q.magneticValue),product(q.electricValue,q.magneticSpatialDerivative)),1/mu):0;
 const energyTimeDerivative=sum(electricStorageRate,magneticStorageRate);
 const componentEnergyReference=product(eps,sum(product(q.rightElectric,q.rightElectric),product(q.leftElectric,q.leftElectric)));
 const naiveComponentFluxReference=product(eps,q.speed,sum(product(q.rightElectric,q.rightElectric),product(-1,q.leftElectric,q.leftElectric)));
 // A Gaussian with nonzero amplitude is mathematically nonzero at finite z,t.
 // Its normalized analytic ratio can exist when absolute double energies cannot.
 let analyticCompatibleVelocity:Value=null;
 if(s.amplitude!==0){
  if(s.rightFraction===0)analyticCompatibleVelocity=-q.speed;
  else if(s.rightFraction===1)analyticCompatibleVelocity=q.speed;
  else{
   const spatial=product(2,z,q.displacement,1/(s.width*s.width));
   const halfLogRatio=sum(Math.log(s.rightFraction)-Math.log1p(-s.rightFraction),spatial);
   analyticCompatibleVelocity=halfLogRatio===null?null:product(q.speed,Math.tanh(halfLogRatio));
  }
 }
 const observedVelocity=quotient(poyntingZ,energyDensity);
 const compatible=q.magneticMode==='compatible';
 const wholeLineReference=product(eps,s.amplitude,s.amplitude,s.width,Math.sqrt(Math.PI),s.rightFraction**2+(1-s.rightFraction)**2);
 const overlap=q.displacement===null?null:Math.exp(-((q.displacement/s.width)**2));
 const overlapTerm=overlap===0&&s.rightFraction!==0&&s.rightFraction!==1?null:product(2,s.rightFraction,1-s.rightFraction,overlap);
 const zeroBWholeLine=product(.5,eps,s.amplitude,s.amplitude,s.width,Math.sqrt(Math.PI),sum(s.rightFraction**2,(1-s.rightFraction)**2,overlapTerm));
 return{...q,electricEnergyDensity:electricEnergy,magneticEnergyDensity:magneticEnergy,energyDensity,energyStatus:status(energyDensity),
  poyntingZ,poyntingStatus:status(poyntingZ),naivePoyntingZ,componentEnergyReference,naiveComponentFluxReference,
  electricStorageRate,magneticStorageRate,energyTimeDerivative,poyntingDivergence,fieldToMatterPowerDensity:0,
  energyResidual:sum(energyTimeDerivative,poyntingDivergence),observedEnergyVelocity:observedVelocity,
  observedVelocityStatus:energyDensity===0?'zero-energy-undefined' as const:observedVelocity===null?'arithmetic-range' as const:'finite' as const,
  analyticCompatibleVelocity,analyticCompatibleVelocityMeaning:'compatible-family-ratio-not-an-observed-quotient-or-information-speed' as const,
  actualAnalyticVelocity:s.amplitude===0?null:compatible?analyticCompatibleVelocity:0,
  wholeLineCompatibleEnergyPerArea:wholeLineReference,wholeLineActualEnergyPerArea:compatible?wholeLineReference:zeroBWholeLine,
  wholeLineMeaning:compatible?'time-independent-energy-per-transverse-area' as const:'actual-zero-B-energy-differs-from-compatible-reference' as const,
  diagnosticMeaning:'analytic-energy-diagnostics-do-not-certify-full-maxwell-system' as const};
}

// Eight-node Gauss quadrature on subintervals <=sigma/2. We preserve the actual
// thin physical width outside the unit-coordinate integral. No error bound claimed.
const nodes=[-.9602898564975363,-.7966664774136267,-.525532409916329,-.1834346424956498,.1834346424956498,.525532409916329,.7966664774136267,.9602898564975363];
const weights=[.1012285362903763,.2223810344533745,.3137066458778873,.362683783378362,.362683783378362,.3137066458778873,.2223810344533745,.1012285362903763];
function integrate(s:TransverseGaussianPulse,a:number,b:number,time:number,key:'energyDensity'|'energyTimeDerivative'){
 if(a===b)return 0;const width=b-a,parts=Math.max(1,Math.ceil(width/(s.width*.5))),values:Value[]=[];
 for(let i=0;i<parts;i++)for(let j=0;j<nodes.length;j++){
  const z=a+width*(i+.5+.5*nodes[j])/parts;
  values.push(product(weights[j],transversePulseEnergy(s,z,time)[key]));
 }
 return product(width,.5/parts,sum(...values));
}
function squaredGaussianDifference(s:TransverseGaussianPulse,a:number,b:number,center:Value,weight:number):Value{
 if(weight===0||s.amplitude===0||a===b)return 0;if(center===null)return null;
 const exponentDifference=product(-(b-a),a+b-2*center,1/(s.width*s.width));
 if(exponentDifference===0)return 0;if(exponentDifference===null)return null;
 const endpoint=exponentDifference>=0?b:a;
 const logValue=2*(Math.log(Math.abs(s.amplitude))+Math.log(weight))-((endpoint-center)/s.width)**2;
 const large=Math.exp(logValue);if(large===0||!Number.isFinite(large))return null;
 return exponentDifference>=0?product(large,-Math.expm1(-exponentDifference)):product(large,Math.expm1(exponentDifference));
}
export function transversePulseSlab(s:TransverseGaussianPulse,start:number,end:number,time:number){
 bound(start,-4,4,'Slab start m');bound(end,-4,4,'Slab end m');if(end<start)throw new RangeError('Slab end must not precede start');
 const left=transversePulseEnergy(s,start,time),right=transversePulseEnergy(s,end,time);
 const gp=squaredGaussianDifference(s,start,end,left.displacement,s.rightFraction);
 const gm=squaredGaussianDifference(s,start,end,product(-1,left.displacement),1-s.rightFraction);
 const stableOutwardPower=left.magneticMode==='compatible'?product(left.permittivity,left.speed,sum(gp,product(-1,gm))):0;
 const energyPerArea=integrate(s,start,end,time,'energyDensity'),energyRatePerArea=integrate(s,start,end,time,'energyTimeDerivative');
 return{start,end,time,width:end-start,leftPoynting:left.poyntingZ,rightPoynting:right.poyntingZ,
  naiveOutwardPowerPerArea:sum(right.poyntingZ,product(-1,left.poyntingZ)),stableOutwardPowerPerArea:stableOutwardPower,
  energyPerArea,energyRatePerArea,balanceResidual:sum(energyRatePerArea,stableOutwardPower),
  regionType:'fixed-slab-per-transverse-area' as const,integrationMethod:'composite-eight-node-gauss-unit-coordinate' as const,
  diagnosticMeaning:'analytic-boundary-difference-plus-numerical-storage-integral-not-a-certified-error-bound' as const};
}

/** Static E=(0,x,0), B=J=0: energy balance passes, Faraday fails. */
export function energyOnlyNegativeControl(x:number){
 bound(x,-4,4,'Coordinate m');
 return{electricField:[0,x,0] as StaticVector,magneticField:[0,0,0] as StaticVector,energyTimeDerivative:0,poynting:[0,0,0] as StaticVector,
  poyntingDivergence:0,fieldToMatterPowerDensity:0,energyResidual:0,faradayResidualZ:1,maxwellFamilyCompatible:false,
  diagnosticMeaning:'energy-balance-alone-is-insufficient' as const};
}
