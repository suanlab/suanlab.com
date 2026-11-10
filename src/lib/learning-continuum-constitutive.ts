/** Exact teaching families. Signed logarithms retain nonzero observations outside linear range. */
import {binaryProduct as raw,binarySum as rawSum,binaryNegate as rawNegate,binaryRatioObservation as rawRatio}from './learning-binary-products';
export type ConstitutiveValue=number|null;
interface Q{sign:number;log:number;}
const zero:Q={sign:0,log:0};
function bound(v:number,lo:number,hi:number,name:string){if(!Number.isFinite(v)||v<lo||v>hi)throw new RangeError(`Invalid ${name}`);}
function q(...values:number[]):Q{if(values.includes(0))return zero;return{sign:values.reduce((s,v)=>s*Math.sign(v),1),log:values.reduce((s,v)=>s+Math.log(Math.abs(v)),0)};}
const mul=(a:Q,b:Q):Q=>a.sign===0||b.sign===0?zero:{sign:a.sign*b.sign,log:a.log+b.log};
const shift=(a:Q,log:number):Q=>a.sign===0?zero:{sign:a.sign,log:a.log+log};
const neg=(a:Q):Q=>({sign:-a.sign,log:a.log});
function add(...terms:Q[]):Q{let top=-Infinity;for(const v of terms)if(v.sign!==0)top=Math.max(top,v.log);if(top===-Infinity)return zero;const n=terms.reduce((a,v)=>a+(v.sign===0?0:v.sign*Math.exp(v.log-top)),0);return n===0?zero:{sign:Math.sign(n),log:top+Math.log(Math.abs(n))};}
function value(a:Q):ConstitutiveValue{if(a.sign===0)return 0;const n=a.sign*Math.exp(a.log);return n===0||!Number.isFinite(n)?null:n;}
const log=(a:Q)=>a.sign===0?null:a.log;
function expm1(a:Q):Q{if(a.sign!==0&&a.log<Math.log(1e-16))return a;const n=value(a);return n===null?a:q(Math.expm1(n));}
function sin(a:Q):Q{if(a.sign!==0&&a.log<Math.log(1e-8))return a;const n=value(a);return n===null?a:q(Math.sin(n));}
function cos(a:Q):Q{return q(Math.cos(value(a)??0));}
function sinh(a:Q):Q{if(a.sign!==0&&a.log<Math.log(1e-8))return a;const n=value(a);return n===null?a:q(Math.sinh(n));}
const observedSum=(...terms:ConstitutiveValue[]):ConstitutiveValue=>terms.includes(null)?null:(terms as number[]).reduce((a,b)=>a+b,0);
const observedNeg=(v:ConstitutiveValue)=>v===null?null:-v;
const difference=(a:ConstitutiveValue,b:ConstitutiveValue)=>observedSum(a,observedNeg(b));
export interface ViscoelasticParameters{elasticModulus:number;viscosity:number;strainAmplitude:number;angularFrequency:number;initialElasticStrain:number;}
function validateViscoelastic(p:ViscoelasticParameters){bound(p.elasticModulus,.25,4,'elastic modulus');bound(p.viscosity,.0625,4,'viscosity');bound(p.strainAmplitude,-.005,.005,'small strain amplitude');bound(p.angularFrequency,0,4,'angular frequency');bound(p.initialElasticStrain,-.005,.005,'initial elastic strain');}
function viscoelasticLogs(p:ViscoelasticParameters,time:number|Q){
 const E=p.elasticModulus,eta=p.viscosity,omega=p.angularFrequency,amplitude=p.strainAmplitude,rate=E/eta,tau=eta/E;
 const timeQ=typeof time==='number'?q(time):time,phase=mul(q(omega),timeQ),z=mul(q(rate),timeQ),roundedZ=value(z)??0,decayLog=-roundedZ,x=q(omega,tau),roundedX=omega*tau,denLog=Math.log1p(roundedX*roundedX);
 const strain=mul(q(amplitude),sin(phase)),strainRate=mul(q(amplitude,omega),cos(phase)),strainAcceleration=neg(mul(q(amplitude,omega,omega),sin(phase)));
 const eamp=q(E,amplitude),initial=q(E,p.initialElasticStrain),mix=neg(expm1(neg(z))),cosMinusOne=neg(mul(q(2),mul(sin(mul(phase,q(.5))),sin(mul(phase,q(.5))))));
 const a=shift(mul(eamp,mul(x,x)),-denLog),b=shift(mul(eamp,x),-denLog);
 const forcing=shift(mul(eamp,add(mul(mul(x,x),sin(phase)),mul(x,add(cosMinusOne,mix)))),-denLog);
 const memory=shift(initial,decayLog);let stress=add(memory,forcing);
 let stressRate=add(shift(neg(mul(initial,q(rate))),decayLog),shift(mul(eamp,add(mul(mul(x,x),mul(q(omega),cos(phase))),mul(x,add(neg(mul(q(omega),sin(phase))),shift(q(rate),decayLog))))),-denLog));
 let change=add(mul(initial,expm1(neg(z))),forcing);
 if(roundedZ+Math.abs(value(phase)??0)<.05){
  // E*(a*omega-E*e0/eta): form the complete input ratio before observing
  // the small difference. Rounded drive/release products can falsely agree.
  const initialRateObservation=rawRatio(rawSum([raw(E,eta,amplitude,omega),rawNegate(raw(E,E,p.initialElasticStrain))]),raw(eta));
  const initialRate:Q={sign:initialRateObservation.sign,log:initialRateObservation.logMagnitude??0};
  const correctionIntegral=quadrature(u=>{const half=sin(mul(phase,q(u,.5)));return[shift(neg(mul(q(2),mul(half,half))),-roundedZ*(1-u))];},0,1,1)[0];
  const correction=mul(mul(q(E,amplitude,omega),timeQ),correctionIntegral);
  change=add(shift(mul(initialRate,mix),-Math.log(rate)),correction);
  stress=add(initial,change);
  stressRate=add(shift(initialRate,decayLog),mul(q(E,amplitude,omega),cosMinusOne),neg(mul(q(rate),correction)));
 }
 const steady=add(mul(a,sin(phase)),mul(b,cos(phase))),transient=shift(add(initial,neg(b)),decayLog);
 const kvStress=add(mul(q(E),strain),mul(q(eta),strainRate)),kvStressRate=add(mul(q(E),strainRate),mul(q(eta),strainAcceleration));
 const model=(s:Q,ds:Q,maxwell:boolean)=>{
  const energy=maxwell?mul(q(.5/E),mul(s,s)):mul(q(.5*E),mul(strain,strain));
  const energyRate=maxwell?mul(q(1/E),mul(s,ds)):mul(q(E),mul(strain,strainRate));
  const dissipation=maxwell?mul(q(1/eta),mul(s,s)):mul(q(eta),mul(strainRate,strainRate));
  return{stress:s,stressRate:ds,energy,energyRate,dissipation,power:mul(s,strainRate)};
 };
 return{rate,tau,phase,z,strain,strainRate,strainAcceleration,initial,stress,stressRate,steady,transient,memory,change,a,b,x,denLog,
  kelvin:model(kvStress,kvStressRate,false),maxwell:model(stress,stressRate,true)};
}
export function viscoelasticAt(p:ViscoelasticParameters,time:number){
 validateViscoelastic(p);bound(time,0,16,'time');const r=viscoelasticLogs(p,time),E=p.elasticModulus,eta=p.viscosity;
 const observe=(m:typeof r.maxwell)=>{const power=value(m.power),rate=value(m.energyRate),diss=value(m.dissipation);return{stress:value(m.stress),stressSign:m.stress.sign,logStressMagnitude:log(m.stress),stressRate:value(m.stressRate),energy:value(m.energy),logEnergy:log(m.energy),energyRate:rate,dissipation:diss,logDissipation:log(m.dissipation),power,balanceResidual:observedSum(power,observedNeg(rate),observedNeg(diss)),omittedDissipationResidual:difference(power,rate)};};
 const sigma=value(r.stress),initial=value(r.initial),elastic=mul(r.stress,q(1/E));
 return{time,relaxationTime:r.tau,relaxationRate:r.rate,phase:value(r.phase),strain:value(r.strain),strainSign:r.strain.sign,logStrainMagnitude:log(r.strain),strainRate:value(r.strainRate),strainAcceleration:value(r.strainAcceleration),
  kelvin:observe(r.kelvin),maxwell:observe(r.maxwell),maxwellSteadyStress:value(r.steady),maxwellTransientStress:value(r.transient),homogeneousMemory:value(r.memory),memorySign:r.memory.sign,logMemoryMagnitude:log(r.memory),
  stableMaxwellStressChange:value(r.change),stressChangeSign:r.change.sign,logStressChangeMagnitude:log(r.change),roundedMaxwellStressChange:difference(sigma,initial),
  maxwellElasticStrain:value(elastic),maxwellViscousStrain:value(add(r.strain,neg(elastic))),maxwellActualViscousRate:value(add(r.strainRate,neg(mul(r.stressRate,q(1/E))))),maxwellDashpotRate:value(mul(r.stress,q(1/eta))),
  maxwellEquationResidual:observedSum(value(r.stressRate),value(mul(r.stress,q(r.rate))),observedNeg(value(mul(q(E),r.strainRate)))),
  omittedRelaxationResidual:observedSum(value(r.stressRate),observedNeg(value(mul(q(E),r.strainRate)))),
  residualMeaning:'analytic-family-diagnostic-not-independent-validation' as const};
}
const nodes=[.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,.755404408355003,.8656312023878318,.9445750230732326,.9894009349916499];
const weights=[.1894506104550685,.1826034150449236,.16915651939500254,.14959598881657673,.12462897125553387,.09515851168249278,.06225352393864789,.027152459411754095];
function quadrature(fn:(x:number,position?:Q)=>Q[],start:number,end:number,panels:number):Q[]{
 const width=end-start,columns:Q[][]=[];if(width===0)return fn(start).map(()=>zero);
 for(let j=0;j<panels;j++)for(let i=0;i<nodes.length;i++)for(const sign of[-1,1]){const x=start+(j+.5+sign*.5*nodes[i])*width/panels;const fraction=(j+.5+sign*.5*nodes[i])/panels,position=add(q(start),q(width,fraction)),values=fn(x,position),scale=Math.log(Math.abs(width))-Math.log(2*panels)+Math.log(weights[i]);values.forEach((v,k)=>(columns[k]??=[]).push(mul(shift(v,scale),q(Math.sign(width)))));}
 return columns.map(c=>add(...c));
}
function viscoelasticEnergyChanges(p:ViscoelasticParameters,start:number,end:number){
 const first=viscoelasticLogs(p,start),last=viscoelasticLogs(p,end),phaseMid=q(p.angularFrequency,start+end,.5),phaseHalf=q(p.angularFrequency,end-start,.5);
 const sinDiff=mul(q(2),mul(cos(phaseMid),sin(phaseHalf))),cosDiff=neg(mul(q(2),mul(sin(phaseMid),sin(phaseHalf))));
 const strainDiff=mul(q(p.strainAmplitude),sinDiff),strainSum=mul(q(2,p.strainAmplitude),mul(sin(phaseMid),cos(phaseHalf)));
 const stressDiff=start===end?zero:start===0?last.change:end===0?neg(first.change):add(mul(first.a,sinDiff),mul(first.b,cosDiff),mul(shift(add(first.initial,neg(first.b)),-first.rate*start),expm1(neg(q(first.rate,end-start)))));
 const stressSum=add(first.stress,last.stress);
 return{kelvin:mul(q(.5*p.elasticModulus),mul(strainDiff,strainSum)),maxwell:mul(q(.5/p.elasticModulus),mul(stressDiff,stressSum))};
}
export function viscoelasticWindow(p:ViscoelasticParameters,start:number,end:number){
 validateViscoelastic(p);bound(start,0,16,'start time');bound(end,0,16,'end time');if(end<start)throw new RangeError('Expected start <= end');
 const panels=Math.max(1,Math.ceil((end-start)*Math.max(p.elasticModulus/p.viscosity,p.angularFrequency)/2));
 const sums=quadrature((time,position)=>{const r=viscoelasticLogs(p,position??time);return[r.kelvin.power,r.kelvin.dissipation,r.maxwell.power,r.maxwell.dissipation];},start,end,panels),changes=viscoelasticEnergyChanges(p,start,end);
 const observe=(index:number,change:Q)=>{const work=value(sums[index]),dissipation=value(sums[index+1]),energyChange=value(change);return{work,workSign:sums[index].sign,logWorkMagnitude:log(sums[index]),dissipation,logDissipation:log(sums[index+1]),energyChange,energyChangeSign:change.sign,logEnergyChangeMagnitude:log(change),balanceResidual:observedSum(work,observedNeg(energyChange),observedNeg(dissipation)),omittedDissipationResidual:difference(work,energyChange)};};
 return{start,end,panels,kelvin:observe(0,changes.kelvin),maxwell:observe(2,changes.maxwell),integrationMeaning:'physical-time-quadrature-not-a-certified-error-bound' as const};
}
function modulusTerms(p:ViscoelasticParameters){
 validateViscoelastic(p);const r=viscoelasticLogs(p,0),E=p.elasticModulus,omega=p.angularFrequency;
 const maxwellStorage=shift(mul(q(E),mul(r.x,r.x)),-r.denLog),maxwellLoss=shift(mul(q(E),r.x),-r.denLog),kelvinLoss=q(p.viscosity,omega);
 return{r,E,omega,maxwellStorage,maxwellLoss,kelvinLoss};
}
function observeModuli(m:ReturnType<typeof modulusTerms>){return{dimensionlessFrequency:value(m.r.x),kelvinStorage:m.E,kelvinLoss:value(m.kelvinLoss),maxwellStorage:value(m.maxwellStorage),logMaxwellStorage:log(m.maxwellStorage),maxwellLoss:value(m.maxwellLoss),logMaxwellLoss:log(m.maxwellLoss)};}
export function viscoelasticModuli(p:ViscoelasticParameters){return observeModuli(modulusTerms(p));}
export function viscoelasticCycle(p:ViscoelasticParameters){
 const terms=modulusTerms(p),{r,E,omega,maxwellLoss,kelvinLoss}=terms;
 const kelvinWork=mul(q(Math.PI,p.strainAmplitude,p.strainAmplitude),kelvinLoss),maxwellWork=mul(q(Math.PI,p.strainAmplitude,p.strainAmplitude),maxwellLoss);
 const integrated=omega===0?null:quadrature(phase=>{const sine=q(Math.sin(phase)),cosine=q(Math.cos(phase)),strainPhaseRate=mul(q(p.strainAmplitude),cosine),kv=add(mul(q(E,p.strainAmplitude),sine),mul(q(p.viscosity,omega,p.strainAmplitude),cosine)),m=add(mul(r.a,sine),mul(r.b,cosine));return[mul(kv,strainPhaseRate),mul(m,strainPhaseRate)];},0,2*Math.PI,16);
 const period=omega===0?null:2*Math.PI/omega,finitePeriod=period!==null&&Number.isFinite(period)?period:null;
 return{...observeModuli(terms),
  period:finitePeriod,periodStatus:omega===0?'no-cycle' as const:finitePeriod===null?'outside-arithmetic-range' as const:'finite-period' as const,
  kelvinSteadyCycleWork:omega===0?null:value(kelvinWork),maxwellSteadyCycleWork:omega===0?null:value(maxwellWork),logMaxwellCycleWork:omega===0?null:log(maxwellWork),
  integratedKelvinSteadyWork:integrated===null?null:value(integrated[0]),integratedMaxwellSteadyWork:integrated===null?null:value(integrated[1]),
  actualFirstCycle:finitePeriod!==null&&finitePeriod<=16?viscoelasticWindow(p,0,finitePeriod):null,
  firstCycleStatus:omega===0?'no-cycle' as const:finitePeriod===null?'outside-arithmetic-range' as const:finitePeriod>16?'outside-teaching-time-horizon' as const:'available' as const};
}

export interface NeoHookeanParameters{shearModulus:number;logStretch:number;rotation:number;normalAngle:number;referenceArea:number;referenceLength:number;}
function validateNeo(p:NeoHookeanParameters){bound(p.shearModulus,.25,4,'shear modulus');bound(p.logStretch,-Math.log(4),Math.log(4),'log stretch');bound(p.rotation,-Math.PI,Math.PI,'rotation');bound(p.normalAngle,-Math.PI,Math.PI,'normal angle');bound(p.referenceArea,.25,4,'reference area');bound(p.referenceLength,.25,4,'reference length');}
function neoPrincipal(mu:number,input:number|Q){
 const ellQ=typeof input==='number'?q(input):input,ell=value(ellQ)??0,stretch=Math.exp(ell),change=expm1(ellQ),triple=expm1(mul(q(3),ellQ));
 const nominal=shift(mul(q(mu),triple),-2*ell),cauchy=shift(mul(q(mu),triple),-ell),energy=mul(q(.5*mu*(stretch+2)/stretch),mul(change,change));
 return{stretch,change,nominal,cauchy,energy,tangent:mu*(1+2*Math.exp(-3*ell))};
}
export function neoHookeanAt(p:NeoHookeanParameters){
 validateNeo(p);const r=neoPrincipal(p.shearModulus,p.logStretch),lambda=r.stretch,transverse=Math.exp(-p.logStretch/2),c=Math.cos(p.rotation),s=Math.sin(p.rotation),axis=[c,s,0],normal=[Math.cos(p.normalAngle),Math.sin(p.normalAngle),0],tangent=[-normal[1],normal[0],0];
 const F=[[c*lambda,-s*transverse,0],[s*lambda,c*transverse,0],[0,0,transverse]],B=F.map(row=>F.map(other=>row.reduce((sum,v,i)=>sum+v*other[i],0))),pressure=p.shearModulus*Math.exp(-p.logStretch);
 const stressLogs=axis.map(a=>axis.map(b=>mul(r.cauchy,q(a,b)))),stress=stressLogs.map(row=>row.map(value)),firstPiola=axis.map(a=>[value(mul(r.nominal,q(a))),0,0]);
 const projection=Math.cos(p.normalAngle-p.rotation),tractionLogs=axis.map(a=>mul(r.cauchy,q(a,projection))),traction=tractionLogs.map(value),normalStress=mul(r.cauchy,q(projection,projection)),shearStress=mul(r.cauchy,q(projection,Math.sin(p.rotation-p.normalAngle)));
 const force=mul(r.nominal,q(p.referenceArea)),stored=mul(r.energy,q(p.referenceArea,p.referenceLength)),displacement=mul(r.change,q(p.referenceLength));
 return{logStretch:p.logStretch,stretch:lambda,transverseStretch:transverse,stableStretchChange:value(r.change),roundedStretchChange:lambda-1,F,B,declaredJacobian:1,roundedJacobian:(F[0][0]*F[1][1]-F[0][1]*F[1][0])*F[2][2],pressure,
  nominalStress:value(r.nominal),nominalStressSign:r.nominal.sign,logNominalStressMagnitude:log(r.nominal),cauchyStress:value(r.cauchy),cauchyStressSign:r.cauchy.sign,logCauchyStressMagnitude:log(r.cauchy),energyDensity:value(r.energy),logEnergyDensity:log(r.energy),nominalTangent:r.tangent,
  roundedEnergyDensity:p.shearModulus/2*(lambda*lambda+2/lambda-3),roundedCauchyStress:p.shearModulus*(lambda*lambda-1/lambda),roundedNominalStress:p.shearModulus*(lambda-1/(lambda*lambda)),
  stress,logStressMagnitude:stressLogs.map(row=>row.map(log)),firstPiola,roundedDirectStress:B.map((row,i)=>row.map((b,j)=>p.shearModulus*b-(i===j?pressure:0))),axis,normal,tangent,traction,normalStress:value(normalStress),signedShearStress:value(shearStress),
  referenceArea:p.referenceArea,currentArea:p.referenceArea*Math.exp(-p.logStretch),referenceLength:p.referenceLength,currentLength:p.referenceLength*lambda,stableDisplacement:value(displacement),logDisplacementMagnitude:log(displacement),referenceVolume:p.referenceArea*p.referenceLength,
  force:value(force),forceSign:force.sign,logForceMagnitude:log(force),forceVector:axis.map(a=>value(mul(force,q(a)))),storedEnergy:value(stored),logStoredEnergy:log(stored),
  wrongSameAreaForce:value(mul(r.cauchy,q(p.referenceArea))),smallStrainNominalControl:value(q(3*p.shearModulus,Math.expm1(p.logStretch))),
  residualMeaning:'uniaxial-family-not-a-general-stability-certificate' as const};
}
function neoEnergyChange(mu:number,start:number,end:number):Q{
 if(start===end)return zero;if(start===0)return neoPrincipal(mu,end).energy;if(end===0)return neg(neoPrincipal(mu,start).energy);
 const mid=q(start+end,.5),half=q(end-start,.5),m=value(mid)??0;
 const bracket=add(expm1(mul(q(3),mid)),shift(mul(q(2),mul(sinh(mul(half,q(.5))),sinh(mul(half,q(.5))))),3*m));
 return shift(mul(q(2*mu),mul(sinh(half),bracket)),-m);
}
export function neoHookeanWork(p:Omit<NeoHookeanParameters,'logStretch'>,start:number,end:number){
 validateNeo({...p,logStretch:start});validateNeo({...p,logStretch:end});const panels=Math.max(1,Math.ceil(Math.abs(end-start)*4));
 const mid=q(start+end,.5),half=q(end-start,.5),m=value(mid)??0,expMinusOne=expm1(mul(q(3),mid));
 const sums=quadrature(u=>{
  const offset=mul(half,q(u)),pair=add(mul(expMinusOne,q(Math.cosh(2*(value(offset)??0)))),mul(q(2),mul(sinh(mul(offset,q(1.5))),sinh(mul(offset,q(.5))))));
  const wrongPair=add(mul(expMinusOne,q(Math.cosh(3*(value(offset)??0)))),mul(q(2),mul(sinh(mul(offset,q(1.5))),sinh(mul(offset,q(1.5))))));
  return[mul(half,shift(mul(q(2*p.shearModulus),pair),-m)),mul(half,mul(q(2*p.shearModulus),wrongPair))];
 },0,1,panels),change=neoEnergyChange(p.shearModulus,start,end),volume=q(p.referenceArea,p.referenceLength);
 const work=value(sums[0]),energyChange=value(change);
 return{start,end,panels,workDensity:work,workSign:sums[0].sign,logWorkDensityMagnitude:log(sums[0]),energyDensityChange:energyChange,energyChangeSign:change.sign,logEnergyChangeMagnitude:log(change),work:value(mul(sums[0],volume)),storedEnergyChange:value(mul(change,volume)),wrongCauchyWorkDensity:value(sums[1]),balanceResidual:difference(work,energyChange),integrationMeaning:'reference-volume-path-quadrature-not-a-certified-error-bound' as const};
}
