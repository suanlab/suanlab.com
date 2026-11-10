/** Lumped RC closure and passive finite-area coils in prescribed smooth local fields.
 * Analytic identities are not independent certificates or whole-device Maxwell solutions.
 */
import {maxwellMedium}from './learning-maxwell';
import type {StaticVector}from './learning-electro-magnetostatics';
type Value=number|null;
function bound(x:number,lo:number,hi:number,name:string){if(!Number.isFinite(x)||x<lo||x>hi)throw new RangeError(`${name} outside [${lo},${hi}]`);}
function mul(...xs:Value[]):Value{if(xs.includes(0))return 0;if(xs.includes(null))return null;let v=1;for(const x of xs){v*=x!;if(!Number.isFinite(v)||v===0)return null;}return v||0;}
function add(...xs:Value[]):Value{if(xs.includes(null))return null;let s=0,e=0;for(const x of xs){const t=s+x!;e+=Math.abs(s)>=Math.abs(x!)?(s-t)+x!:(x!-t)+s;s=t;}const r=s+e;return Number.isFinite(r)?r||0:null;}
function div(a:Value,b:Value):Value{if(a===null||b===null||b===0)return null;const q=a/b;return !Number.isFinite(q)||a!==0&&q===0?null:q||0;}
const neg=(x:Value)=>mul(-1,x),vec=(xs:Value[]):StaticVector|null=>xs.includes(null)?null:xs as StaticVector;
const status=(v:Value)=>v===null?'arithmetic-range' as const:'finite' as const;
export interface RcPlateCircuit{resistance:number;plateRadius:number;gap:number;supplyVoltage:number;initialVoltage:number;permittivity?:number;permeability?:number;}
export function rcPlateState(s:RcPlateCircuit,theta:number,radiusFraction:number){
 bound(s.resistance,1,1e10,'Resistance ohm');bound(s.plateRadius,.05,2,'Plate radius m');bound(s.gap,.01,1,'Gap m');bound(s.supplyVoltage,-4,4,'Supply V');bound(s.initialVoltage,-4,4,'Initial V');bound(theta,0,16,'Time t/RC');bound(radiusFraction,0,1,'Radius fraction');
 const m=maxwellMedium(s),eps=m.permittivity,mu=m.permeability,area=Math.PI*s.plateRadius**2,C=eps*area/s.gap,T=s.resistance*C,delta=s.supplyVoltage-s.initialVoltage,decay=Math.exp(-theta),progress=-Math.expm1(-theta),change=mul(delta,progress),voltage=add(s.initialVoltage,change),roundedVoltage=s.initialVoltage+delta*(1-decay);
 const current=mul(div(delta,s.resistance),decay),voltageRate=mul(div(delta,T),decay),voltageAcceleration=mul(div(-delta,T*T),decay),radius=mul(s.plateRadius,radiusFraction);
 const charge=mul(C,voltage),chargeChange=mul(C,change),initialEnergy=mul(.5,C,s.initialVoltage,s.initialVoltage),electricEnergy=mul(.5,C,voltage,voltage),storedEnergyChange=mul(.5,C,change,add(2*s.initialVoltage,change));
 const supplyWork=mul(s.supplyVoltage,chargeChange),heat=mul(.5,C,delta,delta,-Math.expm1(-2*theta));
 const sourcePower=mul(s.supplyVoltage,current),resistorPower=mul(s.resistance,current,current),electricStorageRate=mul(C,voltage,voltageRate);
 const electricZ=div(voltage,s.gap),magneticPhi=mul(.5,mu,eps,radius,div(voltageRate,s.gap)),faradayPhi=mul(.5,mu,eps,radius,div(voltageAcceleration,s.gap));
 const gapMagneticEnergy=mul(mu,eps,eps,Math.PI,s.plateRadius**4,voltageRate,voltageRate,1/(16*s.gap));
 return{...m,resistance:s.resistance,plateRadius:s.plateRadius,gap:s.gap,theta,radiusFraction,radius,area,capacitance:C,timeConstant:T,time:mul(theta,T),voltage,roundedVoltage,stableVoltageChange:change,naiveVoltageChange:roundedVoltage-s.initialVoltage,voltageRate,voltageAcceleration,current,charge,chargeChange,initialEnergy,electricEnergy,storedEnergyChange,supplyWork,resistorHeat:heat,sourcePower,resistorPower,electricStorageRate,
  kvlResidual:add(voltage,mul(s.resistance,current),-s.supplyVoltage),powerResidual:add(sourcePower,neg(resistorPower),neg(electricStorageRate)),integratedEnergyResidual:add(supplyWork,neg(heat),neg(storedEnergyChange)),
  field:{electricZ,magneticPhi,faradayPhi,ampereZ:0,bulkCharge:0,bulkConductionCurrent:0,displacementCurrent:mul(eps,area,div(voltageRate,s.gap)),magneticCirculation:mul(2*Math.PI,radius,magneticPhi),gapMagneticEnergy,observedMagneticElectricEnergyRatio:div(gapMagneticEnergy,electricEnergy)},
  retardationParameter:s.plateRadius/(m.speed*T),fringingParameter:s.gap/s.plateRadius,reconstructedFieldFamilyExactlyMaxwell:delta===0,changeStatus:status(change),radialGeometryStatus:status(radius),
  diagnosticMeaning:'lumped-positive-RC-and-local-quasistatic-gap-reconstruction-not-exact-global-device-fields' as const};
}
export interface CoilField{uniformField:number;uniformRate:number;uniformAcceleration:number;curvature:number;curvatureRate:number;curvatureAcceleration:number;permittivity?:number;permeability?:number;}
function fieldParameters(s:CoilField){for(const k of['uniformField','uniformRate','uniformAcceleration','curvature','curvatureRate','curvatureAcceleration']as const)bound(s[k],-4,4,k);return maxwellMedium(s);}
function coefficients(s:CoilField,time:number){bound(time,-4,4,'Time s');return{B0:add(s.uniformField,mul(s.uniformRate,time),mul(.5,s.uniformAcceleration,time,time)),B0t:add(s.uniformRate,mul(s.uniformAcceleration,time)),g:add(s.curvature,mul(s.curvatureRate,time),mul(.5,s.curvatureAcceleration,time,time)),gt:add(s.curvatureRate,mul(s.curvatureAcceleration,time))};}
export function coilFieldState(s:CoilField,position:StaticVector,time:number){
 const m=fieldParameters(s);if(!Array.isArray(position)||position.length!==3)throw new RangeError('Three coordinates required');for(let i=0;i<3;i++)bound(position[i],-32,32,'Coordinate m');const[x,y]=position,r2=add(mul(x,x),mul(y,y)),c=coefficients(s,time),Bz=add(c.B0,mul(c.g,r2)),f=add(mul(.5,c.B0t),mul(.25,c.gt,r2)),ft=add(.5*s.uniformAcceleration,mul(.25,s.curvatureAcceleration,r2));
 const electric=vec([mul(f,y),mul(-1,f,x),0]),magnetic=vec([0,0,Bz]),Jfactor=add(div(mul(2,c.g),m.permeability),neg(mul(m.permittivity,ft))),current=vec([mul(Jfactor,y),mul(-1,Jfactor,x),0]);
 const magneticTimeRate=add(c.B0t,mul(c.gt,r2));
 return{...m,position:[...position]as StaticVector,time,...c,electric,magnetic,current,charge:0,magneticTimeRate,curlElectric:vec([0,0,neg(magneticTimeRate)]),faradayResidual:[0,0,0]as StaticVector,ampereResidual:[0,0,0]as StaticVector,magneticDivergence:0,sourceContinuity:0,diagnosticMeaning:'prescribed-polynomial-field-with-required-distributed-current-not-source-free-radiation' as const};
}
export interface CoilProbe{radius:number;turns:number;center:[number,number];velocity:[number,number];}
function probeParameters(p:CoilProbe){bound(p.radius,0,2,'Coil radius m');bound(p.turns,0,64,'Turns');if(!Number.isInteger(p.turns))throw new RangeError('Turns must be integer');for(const v of[p.center,p.velocity]){if(!Array.isArray(v)||v.length!==2)throw new RangeError('Two probe coordinates required');for(let i=0;i<2;i++)bound(v[i],-4,4,'Probe position or velocity');}}
export function finiteAreaCoilState(s:CoilField,p:CoilProbe,time:number){
 fieldParameters(s);probeParameters(p);const c=coefficients(s,time),increments=p.velocity.map(v=>mul(v,time)),center=p.center.map((x,i)=>add(x,increments[i])),r2=add(mul(center[0],center[0]),mul(center[1],center[1])),area=mul(Math.PI,p.radius,p.radius),halfRadiusSquared=mul(.5,p.radius,p.radius),weight=add(r2,halfRadiusSquared),motion=mul(2,add(mul(center[0],p.velocity[0]),mul(center[1],p.velocity[1]))),gain=mul(p.turns,area);
 const pointField=add(c.B0,mul(c.g,r2)),analyticDiskAverage=add(c.B0,mul(c.g,weight)),fluxPerTurn=mul(area,analyticDiskAverage),fluxLinkage=mul(p.turns,fluxPerTurn);
 const transformerEmf=mul(-1,gain,add(c.B0t,mul(c.gt,weight))),motionalEmf=mul(-1,gain,c.g,motion),emf=add(transformerEmf,motionalEmf),pointTransformerEmf=mul(-1,gain,add(c.B0t,mul(c.gt,r2))),finiteAreaCorrection=mul(-1,gain,c.gt,halfRadiusSquared);
 const positionStatus=center.includes(null)?'arithmetic-range' as const:increments.some((v,i)=>v!==0&&center[i]===p.center[i])?'motion-below-coordinate-resolution' as const:'finite' as const;
 const geometryStatus=p.radius===0?'zero-radius' as const:area===null||halfRadiusSquared===null||center.includes(null)?'arithmetic-range' as const:center.some(x=>x!+p.radius===x||x!-p.radius===x)?'radius-below-coordinate-resolution' as const:'finite' as const;
 return{time,radius:p.radius,turns:p.turns,initialCenter:[...p.center]as[number,number],velocity:[...p.velocity]as[number,number],center,trajectoryIncrements:increments,positionStatus,geometryStatus,area,...c,pointField,analyticDiskAverage,observedDiskAverage:div(fluxPerTurn,area),fluxPerTurn,fluxLinkage,transformerEmf,motionalEmf,emf,negativeFluxLinkageRate:emf,pointTransformerEmf,finiteAreaCorrection,
  normalizedEmfObservation:div(neg(emf),gain),emfObservationRow:[gain,mul(gain,weight),mul(gain,motion)],normalizedEmfRow:[1,weight,motion],
  fixedProbe:p.velocity[0]===0&&p.velocity[1]===0,absoluteUniformStaticFieldIdentifiable:false,diagnosticMeaning:'ideal-passive-translating-material-loop-emf-not-point-B-or-loaded-sensor' as const};
}
/** Exact binary-input quadratic contrast. BigInt aligns powers before one final rounding.
 * Exact structural rank remains distinct from representable observations/solves.
 */
function exactBinarySum(inputs:{a:number;b:number;sign:number;half?:number}[]){
 const ZERO=BigInt(0),ONE=BigInt(1),MASK=(ONE<<BigInt(52))-ONE,view=new DataView(new ArrayBuffer(8));
 const decode=(x:number)=>{view.setFloat64(0,x,false);const bits=view.getBigUint64(0,false),e=Number((bits>>BigInt(52))&BigInt(2047)),m=bits&MASK;return{m:((bits>>BigInt(63))!==ZERO?-ONE:ONE)*(e===0?m:m+(ONE<<BigInt(52))),e:e===0?-1074:e-1075};};
 const terms=inputs.map(t=>{const a=decode(t.a),b=decode(t.b);return{m:BigInt(t.sign)*a.m*b.m,e:a.e+b.e-(t.half??0)};});
 const exponent=Math.min(...terms.filter(t=>t.m!==ZERO).map(t=>t.e));if(!Number.isFinite(exponent))return{exactZero:true,value:0};
 let n=terms.reduce((sum,t)=>sum+(t.m===ZERO?ZERO:t.m<<BigInt(t.e-exponent)),ZERO);if(n===ZERO)return{exactZero:true,value:0};const sign=n<ZERO?-1:1;if(n<ZERO)n=-n;
 const bits=n.toString(2).length,quantum=Math.max(exponent,exponent+bits-53,-1074),shift=quantum-exponent;
 let mantissa=n>>BigInt(shift);if(shift>0){const tail=n-(mantissa<<BigInt(shift)),half=ONE<<BigInt(shift-1);if(tail>half||tail===half&&(mantissa&ONE)!==ZERO)mantissa+=ONE;}
 const value=sign*Number(mantissa)*2**quantum;return{exactZero:false,value:!Number.isFinite(value)||value===0?null:value};
}
function exactGeometryContrast(a:CoilProbe,b:CoilProbe){return exactBinarySum([...a.center.map(x=>({a:x,b:x,sign:-1})),...b.center.map(x=>({a:x,b:x,sign:1})),{a:a.radius,b:a.radius,sign:-1,half:1},{a:b.radius,b:b.radius,sign:1,half:1}]);}
export interface FixedCoilMeasurement{probe:CoilProbe;emf:number;}
export function inferFixedCoilRates(samples:FixedCoilMeasurement[],lengthScale=1,voltageErrorBound=0){
 if(!Array.isArray(samples)||samples.length!==2)throw new RangeError('Exactly two fixed-coil measurements required');bound(lengthScale,.1,4,'Inference length scale m');bound(voltageErrorBound,0,1,'Voltage error bound V');
 for(const m of samples){if(!m||!m.probe)throw new RangeError('Complete measurements required');probeParameters(m.probe);bound(m.emf,-1e12,1e12,'Measured emf V');if(m.probe.velocity.some(v=>v!==0))throw new RangeError('Fixed-coil inference does not accept moving probes');}
 const rows=samples.map(({probe:p,emf})=>{const area=mul(Math.PI,p.radius,p.radius),gain=mul(p.turns,area),weight=add(mul(p.center[0],p.center[0]),mul(p.center[1],p.center[1]),mul(.5,p.radius,p.radius));return{gain,weight,normalizedWeight:div(weight,lengthScale**2),observation:div(-emf,gain),noise:div(voltageErrorBound,gain),active:p.turns!==0&&p.radius!==0};});
 const contrast=exactGeometryContrast(samples[0].probe,samples[1].probe),active=rows.filter(r=>r.active).length,structuralRank=active<2?active:contrast.exactZero?1:2;
 const d=div(contrast.value,lengthScale**2),roundedContrast=rows[0].weight===null||rows[1].weight===null?null:rows[1].weight-rows[0].weight;
 const rawContrast=active===2&&rows.every(r=>r.gain!==null)?exactBinarySum([{a:samples[0].emf,b:rows[1].gain!,sign:1},{a:samples[1].emf,b:rows[0].gain!,sign:-1}]):null;
 const stableObservationContrast=rawContrast?div(rawContrast.value,mul(rows[0].gain,rows[1].gain)):null,roundedObservationContrast=add(rows[1].observation,neg(rows[0].observation));
 const slope=structuralRank===2?div(stableObservationContrast,d):null,uniform=structuralRank===2?add(rows[0].observation,neg(mul(slope,rows[0].normalizedWeight))):null,curvatureRate=div(slope,lengthScale**2),ad=d===null?null:Math.abs(d);
 const inverseNorm=ad===null||ad===0?null:div(Math.max(2,add(rows[0].normalizedWeight,rows[1].normalizedWeight)??Infinity),ad),conditionInfinity=mul(add(1,Math.max(rows[0].normalizedWeight??Infinity,rows[1].normalizedWeight??Infinity)),inverseNorm);
 const uniformError=structuralRank===2?div(add(mul(rows[1].normalizedWeight,rows[0].noise),mul(rows[0].normalizedWeight,rows[1].noise)),ad):null,curvatureError=structuralRank===2?div(add(rows[0].noise,rows[1].noise),mul(ad,lengthScale**2)):null;
 let measurementCompatible:boolean|null=samples.some((m,i)=>!rows[i].active&&Math.abs(m.emf)>voltageErrorBound)?false:true;
 if(measurementCompatible&&active===2&&structuralRank===1){
  if(rows.some(r=>r.gain===null))measurementCompatible=null;
  else{const cross=exactBinarySum([{a:samples[0].emf,b:rows[1].gain!,sign:1},{a:samples[1].emf,b:rows[0].gain!,sign:-1}]);
   if(voltageErrorBound===0)measurementCompatible=cross.exactZero;
   else{const allowance=mul(voltageErrorBound,add(rows[0].gain,rows[1].gain));if(cross.exactZero)measurementCompatible=true;else if(cross.value===null||allowance===null)measurementCompatible=null;else{const difference=Math.abs(cross.value)-allowance;measurementCompatible=Math.abs(difference)<=8*Number.EPSILON*Math.max(Math.abs(cross.value),allowance)?null:difference<0;}}
  }
 }
 const solveStatus=measurementCompatible===false?'inconsistent' as const:structuralRank!==2?'underdetermined' as const:uniform===null||curvatureRate===null?'arithmetic-range' as const:'unique' as const;
 return{structuralRank,solveStatus,measurementCompatible,stableObservationContrast,roundedObservationContrast,uniformRate:uniform,curvatureRate,exactGeometricContrast:contrast.value,roundedGeometricContrast:roundedContrast,physicalDesignRows:rows.map(r=>[r.gain,mul(r.gain,r.weight)]),normalizedRows:rows.map(r=>r.gain===0||r.gain===null?null:[1,r.normalizedWeight]),normalizedObservations:rows.map(r=>r.observation),conditionInfinity,voltageErrorBound,measurementErrorBounds:{uniformRate:uniformError,curvatureRate:curvatureError},lengthScale,absoluteUniformStaticFieldIdentifiable:false,diagnosticMeaning:'two-fixed-emf-rates-with-known-geometry-and-explicit-column-scaling-not-absolute-field-identification-or-arithmetic-error-certificate' as const};
}
