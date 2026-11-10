/** Independent Hermitian-matrix/algebraic certificates. Never runs the producer or sqrtRange.
 * Worker/CLI only: observation formatting primitives are shared, not the numerical model. */
import {snapshotQuantumInformationInput,quantumInformationInputKey}from'./learning-quantum-information-input';
import {snapshotManyBodyWire}from'./learning-many-body-owned-wire';
import {f,zero,one,two,add,sub,mul,div,neg,eq,le,observe,type F,type R}from'./learning-quantum-hilbert-core';
import type{QuantumInformationCalculation}from'./learning-quantum-information-calculation';
type Reply=Extract<QuantumInformationCalculation,{kind:'discrimination'}>;
type A={a:F;b:F;c:F};type C={re:F;im:F};type AC={re:A;im:A};
const sign=(x:F)=>x.n>BigInt(0)?1:x.n<BigInt(0)?-1:0;
const absolute=(x:F)=>x.n<BigInt(0)?neg(x):x;
const half=(x:F)=>div(x,two),rat=(s:string)=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const alg=(a:F,b=zero,c=zero):A=>({a,b,c});
function aa(x:A,y:A):A{
 if(eq(x.b,zero))return alg(add(x.a,y.a),y.b,y.c);
 if(eq(y.b,zero))return alg(add(x.a,y.a),x.b,x.c);
 if(!eq(x.c,y.c))throw new Error('Different algebraic fields');return alg(add(x.a,y.a),add(x.b,y.b),x.c);
}
const an=(x:A)=>alg(neg(x.a),neg(x.b),x.c),as=(x:A,y:A)=>aa(x,an(y));
const ar=(x:A,r:F)=>alg(mul(x.a,r),mul(x.b,r),x.c);
function am(x:A,y:A):A{
 if(eq(x.b,zero))return ar(y,x.a);if(eq(y.b,zero))return ar(x,y.a);
 if(!eq(x.c,y.c))throw new Error('Different algebraic fields');return alg(add(mul(x.a,y.a),mul(mul(x.b,y.b),x.c)),add(mul(x.a,y.b),mul(x.b,y.a)),x.c);
}
function ai(x:A):A{const denominator=sub(mul(x.a,x.a),mul(mul(x.b,x.b),x.c));if(eq(denominator,zero))throw new Error('Zero algebraic denominator');return alg(div(x.a,denominator),neg(div(x.b,denominator)),x.c);}
function algebraicSign(x:A):number{
 if(sign(x.c)<0)throw new Error('Negative radicand');
 const a=sign(x.a),b=sign(x.b);if(!b||!sign(x.c))return a;if(!a||a===b)return b;
 const difference=sign(sub(mul(x.a,x.a),mul(mul(x.b,x.b),x.c)));return difference===0?0:difference>0?a:b;
}
const ac=(re:A,im=alg(zero)):AC=>({re,im});
const cc=(re:F,im=zero):C=>({re,im});
const cm=(x:C,y:C):C=>cc(sub(mul(x.re,y.re),mul(x.im,y.im)),add(mul(x.re,y.im),mul(x.im,y.re)));
const determinant=(m:C[][])=>sub(mul(m[0][0].re,m[1][1].re),add(mul(m[0][1].re,m[0][1].re),mul(m[0][1].im,m[0][1].im)));
const density=(r:F[])=>[[cc(half(add(one,r[2]))),cc(half(r[0]),neg(half(r[1])))],[cc(half(r[0]),half(r[1])),cc(half(sub(one,r[2])))] ];
const traceProduct=(x:C[][],y:C[][])=>x.flatMap((row,i)=>row.map((z,j)=>cm(z,y[j][i]).re)).reduce(add,zero);
function traceEffect(effect:AC[][],rho:C[][]):A{
 return effect.flatMap((row,i)=>row.map((z,j)=>as(ar(z.re,rho[j][i].re),ar(z.im,rho[j][i].im)))).reduce(aa,alg(zero));
}
class Invalid extends Error{constructor(readonly path:string,reason:string){super(reason);}}
function requireValue(condition:unknown,path:string,reason:string):asserts condition{if(!condition)throw new Invalid(path,reason);}
function record(v:unknown,keys:readonly string[],path:string):Record<string,unknown>{
 requireValue(v&&typeof v==='object'&&!Array.isArray(v),path,'Record required');const d=v as Record<string,unknown>;
 requireValue(Object.keys(d).length===keys.length&&keys.every(k=>Object.hasOwn(d,k)),path,'Exact fields required');return d;
}
function list(v:unknown,size:number,path:string):unknown[]{requireValue(Array.isArray(v)&&v.length===size,path,'Exact array shape required');return v;}
function same(x:unknown,y:unknown):boolean{
 if(Object.is(x,y))return true;if(!x||!y||typeof x!=='object'||typeof y!=='object'||Array.isArray(x)!==Array.isArray(y))return false;
 const a=x as Record<string,unknown>,b=y as Record<string,unknown>,keys=Object.keys(a);return keys.length===Object.keys(b).length&&keys.every(k=>Object.hasOwn(b,k)&&same(a[k],b[k]));
}
function endpoint(value:unknown,path:string):F{
 const d=record(value,['numerator','denominator'],path);
 const binary=(v:unknown,name:string)=>{
  const p=record(v,['coefficient','exponent'],name);
  requireValue(typeof p.coefficient==='string'&&p.coefficient.length<=4096&&/^(?:0|-?[1-9]\d*)$/.test(p.coefficient),name,'Bounded canonical integer coefficient required');
  requireValue(typeof p.exponent==='number'&&Number.isSafeInteger(p.exponent)&&Math.abs(p.exponent)<=8192,name,'Bounded binary exponent required');
  return {coefficient:BigInt(p.coefficient),exponent:p.exponent};
 };
 const n=binary(d.numerator,`${path}.numerator`),den=binary(d.denominator,`${path}.denominator`);
 requireValue(den.coefficient>BigInt(0),path,'Positive denominator required');const shift=n.exponent-den.exponent;
 return shift>=0?f(n.coefficient<<BigInt(shift),den.coefficient):f(n.coefficient,den.coefficient<<BigInt(-shift));
}
const contract='Exact rational Bloch inputs and physicality/classification. General square roots use directed512-bit rational brackets; matrix effects may contain algebraic intervals. A zero eigenvalue makes the optimal effect nonunique. The uninformative case chooses the identity effect canonically. Interval bounds and positive-gain classifications are retained separately from rounded success. No entropy, accessible-information optimization, channel discrimination, public input reader, Worker or UI is supplied.';
export type DiscriminationValidation={ok:true;response:Reply;checkedObservations:number}|{ok:false;path:string;reason:string};
export function validateQuantumDiscriminationResponse(request:unknown,response:unknown):DiscriminationValidation{
 let checkedObservations=0;
 try{
  const input=snapshotQuantumInformationInput(request);requireValue(input.kind==='discrimination','$request','Discrimination request required');
  const owned=snapshotManyBodyWire(response),envelope=record(owned,['schemaVersion','kind','input','result'],'$');
  requireValue(envelope.schemaVersion===1&&envelope.kind==='discrimination','$','Wrong response version or family');
  requireValue(quantumInformationInputKey(envelope.input)===quantumInformationInputKey(input),'$.input','Original request differs');
  const fields=['schemaVersion','dimension','prior','otherPrior','density0','density1','bloch0','bloch1','weightedDifference','weightedTrace','weightedBloch','weightedRadiusSquared','weightedDeterminant','eigenvalues','traceNorm','optimalSuccess','priorOnlySuccess','gainAbovePriorOnly','strictlyBeatsPriorOnly','strategy','uniqueOptimalEffect','optimalGuessZeroEffect','measurementAxis','conditionalOutcomeProbabilities','achievedSuccess','successResidual','traceDistanceSquared','traceDistance','squaredUhlmannFidelity','fidelityConvention','exactStateEquality','exactOrthogonalSupports','approximationContract'];
  const result=record(envelope.result,fields,'$.result');
  const check=(value:unknown,expected:A,path:string,exact=false)=>{
   const d=record(value,['value','sign','logMagnitude','logResolution','resolution','bounds'],path),bounds=record(d.bounds,['lower','upper'],`${path}.bounds`);
   const range:R={lower:endpoint(bounds.lower,`${path}.bounds.lower`),upper:endpoint(bounds.upper,`${path}.bounds.upper`)};
   requireValue(le(range.lower,range.upper),path,'Reversed interval');
   requireValue(algebraicSign(as(expected,alg(range.lower)))>=0&&algebraicSign(as(expected,alg(range.upper)))<=0,path,'Algebraic value outside interval');
   if(exact)requireValue(eq(range.lower,range.upper),path,'Exact rational field requires equal endpoints');
   // With32-digit raw coefficients, root conditioning permits this absolute
   // width bound even for nearly identical states. A valid alternate enclosure
   // is allowed; byte identity with the producer's root algorithm is not required.
   const magnitude=add(one,le(absolute(range.lower),absolute(range.upper))?absolute(range.upper):absolute(range.lower));
   requireValue(le(sub(range.upper,range.lower),div(magnitude,f(BigInt(1)<<BigInt(256)))),path,'Interval too wide');
   requireValue(same(value,observe(range)),path,'Display/resolution metadata disagrees with bounds');checkedObservations++;
  };
  const rational=(name:string,value:F)=>check(result[name],alg(value),`$.result.${name}`,true);
  const vector=(name:string,values:F[])=>list(result[name],values.length,`$.result.${name}`).forEach((v,i)=>check(v,alg(values[i]),`$.result.${name}[${i}]`,true));
  const matrix=(name:string,values:AC[][],exact=false)=>list(result[name],2,`$.result.${name}`).forEach((row,i)=>list(row,2,`$.result.${name}[${i}]`).forEach((v,j)=>{
   const z=record(v,['real','imaginary'],`$.result.${name}[${i}][${j}]`);check(z.real,values[i][j].re,`$.result.${name}[${i}][${j}].real`,exact);check(z.imaginary,values[i][j].im,`$.result.${name}[${i}][${j}].imaginary`,exact);
  }));
  const r0=input.bloch0.map(rat),r1=input.bloch1.map(rat),rho0=density(r0),rho1=density(r1),p=rat(input.prior),other=sub(one,p);
  for(const [index,rho]of [rho0,rho1].entries())requireValue(le(zero,rho[0][0].re)&&le(zero,rho[1][1].re)&&le(zero,determinant(rho)),`$request.bloch${index}`,'Nonphysical qubit');
  const delta=rho0.map((row,i)=>row.map((z,j)=>cc(sub(mul(p,z.re),mul(other,rho1[i][j].re)),sub(mul(p,z.im),mul(other,rho1[i][j].im)))));
  const tr=add(delta[0][0].re,delta[1][1].re),det=determinant(delta),discriminant=sub(mul(tr,tr),mul(f(BigInt(4)),det));
  const root=alg(zero,one,discriminant),low=ar(as(alg(tr),root),half(one)),high=ar(aa(alg(tr),root),half(one)),loSign=algebraicSign(low),hiSign=algebraicSign(high);
  const projector=loSign<0&&hiSign>0,uninformative=loSign===0&&hiSign===0;
  const strategy=uninformative?'uninformative':projector?'spectral-projector':loSign>=0?'always-zero':'always-one';
  const constant=strategy==='always-one'?zero:one;
  const effect:AC[][]=projector?delta.map((row,i)=>row.map((z,j)=>ac(am(as(alg(z.re),i===j?low:alg(zero)),ai(root)),am(alg(z.im),ai(root)))))
   :[[ac(alg(constant)),ac(alg(zero))],[ac(alg(zero)),ac(alg(constant))]];
  const norm=aa(loSign<0?an(low):low,hiSign<0?an(high):high),success=ar(aa(alg(one),norm),half(one)),baseline=le(p,other)?other:p;
  const conditional=[traceEffect(effect,rho0),traceEffect(effect,rho1)],achieved=aa(ar(conditional[0],p),ar(as(alg(one),conditional[1]),other));
  const difference=rho0.map((row,i)=>row.map((z,j)=>cc(sub(z.re,rho1[i][j].re),sub(z.im,rho1[i][j].im))));
  const distanceSquared=neg(determinant(difference));
  requireValue(result.schemaVersion===1&&result.dimension===2,'$.result','Wrong model dimension/version');
  rational('prior',p);rational('otherPrior',other);vector('bloch0',r0);vector('bloch1',r1);
  for(const[name,m]of [['density0',rho0],['density1',rho1],['weightedDifference',delta]]as const)matrix(name,m.map(row=>row.map(z=>ac(alg(z.re),alg(z.im)))),true);
  rational('weightedTrace',tr);rational('weightedRadiusSquared',discriminant);rational('weightedDeterminant',det);
  vector('weightedBloch',[add(delta[0][1].re,delta[1][0].re),sub(delta[1][0].im,delta[0][1].im),sub(delta[0][0].re,delta[1][1].re)]);
  list(result.eigenvalues,2,'$.result.eigenvalues').forEach((x,i)=>check(x,i===0?low:high,`$.result.eigenvalues[${i}]`));
  check(result.traceNorm,norm,'$.result.traceNorm');check(result.optimalSuccess,success,'$.result.optimalSuccess');rational('priorOnlySuccess',baseline);
  check(result.gainAbovePriorOnly,as(success,alg(baseline)),'$.result.gainAbovePriorOnly');
  requireValue(result.strategy===strategy&&result.strictlyBeatsPriorOnly===projector,'$.result.strategy','Wrong optimal strategy');
  requireValue(result.uniqueOptimalEffect===(loSign!==0&&hiSign!==0),'$.result.uniqueOptimalEffect','Wrong zero-eigenvalue classification');
  matrix('optimalGuessZeroEffect',effect);
  if(projector){const axis=[aa(effect[0][1].re,effect[1][0].re),as(effect[1][0].im,effect[0][1].im),as(effect[0][0].re,effect[1][1].re)];list(result.measurementAxis,3,'$.result.measurementAxis').forEach((x,i)=>check(x,axis[i],`$.result.measurementAxis[${i}]`));}
  else requireValue(result.measurementAxis===null,'$.result.measurementAxis','Constant decision has no measurement axis');
  list(result.conditionalOutcomeProbabilities,2,'$.result.conditionalOutcomeProbabilities').forEach((row,i)=>list(row,2,'$.result.conditionalOutcomeProbabilities').forEach((x,j)=>check(x,j===0?conditional[i]:as(alg(one),conditional[i]),`$.result.conditionalOutcomeProbabilities[${i}][${j}]`)));
  check(result.achievedSuccess,achieved,'$.result.achievedSuccess');check(result.successResidual,as(achieved,success),'$.result.successResidual');
  rational('traceDistanceSquared',distanceSquared);check(result.traceDistance,alg(zero,one,distanceSquared),'$.result.traceDistance');
  const fidelity=alg(traceProduct(rho0,rho1),two,mul(determinant(rho0),determinant(rho1)));
  check(result.squaredUhlmannFidelity,fidelity,'$.result.squaredUhlmannFidelity');
  if(projector)requireValue((result.gainAbovePriorOnly as {sign:unknown}).sign===1,'$.result.gainAbovePriorOnly','Positive gain must remain resolved');
  requireValue(result.fidelityConvention==='squared-uhlmann-fidelity','$.result.fidelityConvention','Wrong fidelity convention');
  requireValue(result.exactStateEquality===rho0.every((row,i)=>row.every((z,j)=>eq(z.re,rho1[i][j].re)&&eq(z.im,rho1[i][j].im))),'$.result.exactStateEquality','Wrong exact equality');
  requireValue(result.exactOrthogonalSupports===eq(distanceSquared,one),'$.result.exactOrthogonalSupports','Wrong orthogonality');
  requireValue(result.approximationContract===contract,'$.result.approximationContract','Wrong approximation contract');
  return {ok:true,response:owned as Reply,checkedObservations};
 }catch(error){return {ok:false,path:error instanceof Invalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid response'};}
}
