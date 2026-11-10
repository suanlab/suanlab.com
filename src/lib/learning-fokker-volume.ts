/** Finite-volume Fokker–Planck moments. Whole-grid semigroup enclosures,
 * exact explicit/implicit Euler and boundary accounting, not a continuum proof. */
import {markovFraction as f,exactMarkovObservation as exact,type MarkovFraction as F} from '@/lib/learning-finite-markov-exact';
import {createTranscendentalContext} from '@/lib/learning-interval-transcendentals';
import {partitionObservationFromRange} from '@/lib/learning-partition-exact-core';
export interface FokkerVolumeParameters {
 cells:number;halfWidthNumerator:number;spaceDenominator:number;
 relaxationNumerator:number;windNumerator:number;diffusionNumerator:number;coefficientDenominator:number;
 timeNumerator:number;timeDenominator:number;eulerSteps:number;poissonCutoff:number;
 leftBoundary:'reflecting'|'absorbing';rightBoundary:'reflecting'|'absorbing';initialWeights:number[];
}
const Z=BigInt(0),O=BigInt(1),zero=f(Z),one=f(O),two=f(BigInt(2));
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d);
const sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n);
const sum=(a:readonly F[])=>a.reduce(add,zero),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
const max=(a:F,b:F)=>le(a,b)?b:a,min=(a:F,b:F)=>le(a,b)?a:b;
const positive=(a:F)=>max(zero,a),output=(a:F)=>exact(a.n,a.d),binary=(coefficient:bigint)=>({coefficient,exponent:0});
interface Range {lower:F;upper:F}
const scale=(a:Range,b:F):Range=>b.n<Z?{lower:mul(a.upper,b),upper:mul(a.lower,b)}:{lower:mul(a.lower,b),upper:mul(a.upper,b)};
function observe(a:Range){if(a.lower.n*a.upper.d===a.upper.n*a.lower.d)return output(a.lower);return partitionObservationFromRange({lower:{n:binary(a.lower.n),d:binary(a.lower.d)},upper:{n:binary(a.upper.n),d:binary(a.upper.d)}});}
const rowTimes=(row:readonly F[],matrix:readonly (readonly F[])[])=>matrix[0].map((_,j)=>sum(row.map((x,i)=>mul(x,matrix[i][j]))));
function invert(matrix:readonly (readonly F[])[]){
 const n=matrix.length,a=matrix.map((row,i)=>[...row,...Array.from({length:n},(_,j)=>i===j?one:zero)]);
 for(let col=0;col<n;col++){
  const pivot=a.findIndex((row,i)=>i>=col&&row[col].n!==Z);if(pivot<0)throw new RangeError('Singular exact matrix');
  [a[col],a[pivot]]=[a[pivot],a[col]];const d=a[col][col];a[col]=a[col].map(x=>div(x,d));
  for(let i=0;i<n;i++)if(i!==col){const q=a[i][col];a[i]=a[i].map((x,j)=>sub(x,mul(q,a[col][j])));}
 }
 return a.map(row=>row.slice(n));
}
const integer=(x:unknown,min:number,max=1000000):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
export function isFokkerVolumeParameters(value:unknown):value is FokkerVolumeParameters {
 if(!value||typeof value!=='object'||Array.isArray(value))return false;const p=value as Record<string,unknown>;
 const keys=['cells','halfWidthNumerator','spaceDenominator','relaxationNumerator','windNumerator','diffusionNumerator','coefficientDenominator','timeNumerator','timeDenominator','eulerSteps','poissonCutoff','leftBoundary','rightBoundary','initialWeights'];
 const shapeValid=Object.keys(p).length===keys.length&&keys.every(k=>Object.prototype.hasOwnProperty.call(p,k))
 &&integer(p.cells,2,8)&&['halfWidthNumerator','spaceDenominator','coefficientDenominator','timeDenominator'].every(k=>integer(p[k],1))
 &&['relaxationNumerator','diffusionNumerator','timeNumerator'].every(k=>integer(p[k],0))&&integer(p.windNumerator,-1000000)
 &&integer(p.eulerSteps,1,128)&&integer(p.poissonCutoff,0,256)
 &&['reflecting','absorbing'].includes(p.leftBoundary as string)&&['reflecting','absorbing'].includes(p.rightBoundary as string)
 &&2*(p.halfWidthNumerator as number)>=(p.spaceDenominator as number)&&(p.halfWidthNumerator as number)<=4*(p.spaceDenominator as number)
 &&(p.relaxationNumerator as number)<=2*(p.coefficientDenominator as number)&&Math.abs(p.windNumerator as number)<=2*(p.coefficientDenominator as number)
 &&(p.diffusionNumerator as number)<=(p.coefficientDenominator as number)&&(p.timeNumerator as number)<=10*(p.timeDenominator as number)
 &&Array.isArray(p.initialWeights)&&p.initialWeights.length===p.cells&&p.initialWeights.every(x=>integer(x,0))&&p.initialWeights.some(x=>x>0);
 if(!shapeValid)return false;
 // Check the joint workload bound with rational arithmetic, before any solve.
 const v=p as unknown as FokkerVolumeParameters,q=(a:number,b:number)=>f(BigInt(a),BigInt(b));
 const L=q(v.halfWidthNumerator,v.spaceDenominator),dx=div(mul(two,L),f(BigInt(v.cells)));
 const D=q(v.diffusionNumerator,v.coefficientDenominator),gamma=q(v.relaxationNumerator,v.coefficientDenominator),wind=q(v.windNumerator,v.coefficientDenominator),diff=div(D,mul(dx,dx));
 const b=(i:number)=>sub(wind,mul(gamma,add(neg(L),mul(dx,f(BigInt(i))))));
 const exit=Array.from({length:v.cells},(_,i)=>add(
  i>0?add(div(positive(neg(b(i))),dx),diff):v.leftBoundary==='absorbing'?add(div(positive(neg(b(0))),dx),mul(two,diff)):zero,
  i<v.cells-1?add(div(positive(b(i+1)),dx),diff):v.rightBoundary==='absorbing'?add(div(positive(b(v.cells)),dx),mul(two,diff)):zero));
 return le(mul(exit.reduce(max,zero),q(v.timeNumerator,v.timeDenominator)),f(BigInt(64)));
}
export function fokkerVolumeModel(input:FokkerVolumeParameters){
 if(!isFokkerVolumeParameters(input))throw new RangeError('Owned bounded rational grid, drift, diffusion, time, steps, cutoff, two boundaries and positive weights; max exit rate times time at most64 required');
 const p=Object.freeze({...input,initialWeights:Object.freeze([...input.initialWeights])}),n=p.cells,size=n+2,q=(a:number,b:number)=>f(BigInt(a),BigInt(b));
 const L=q(p.halfWidthNumerator,p.spaceDenominator),dx=div(mul(two,L),f(BigInt(n))),gamma=q(p.relaxationNumerator,p.coefficientDenominator),wind=q(p.windNumerator,p.coefficientDenominator),D=q(p.diffusionNumerator,p.coefficientDenominator);
 const time=q(p.timeNumerator,p.timeDenominator),dt=div(time,f(BigInt(p.eulerSteps))),diffusionRate=div(D,mul(dx,dx));
 const faces=Array.from({length:n+1},(_,i)=>add(neg(L),mul(dx,f(BigInt(i))))),centers=faces.slice(0,n).map(x=>add(x,div(dx,two)));
 const velocity=faces.map(x=>sub(wind,mul(gamma,x))),forward=velocity.slice(1,n).map(b=>add(div(positive(b),dx),diffusionRate)),backward=velocity.slice(1,n).map(b=>add(div(positive(neg(b)),dx),diffusionRate));
 const leftRate=p.leftBoundary==='reflecting'?zero:add(div(positive(neg(velocity[0])),dx),mul(two,diffusionRate));
 const rightRate=p.rightBoundary==='reflecting'?zero:add(div(positive(velocity[n]),dx),mul(two,diffusionRate));
 const generator=Array.from({length:size},()=>Array<F>(size).fill(zero));
 for(let i=0;i<n-1;i++){generator[i][i+1]=forward[i];generator[i+1][i]=backward[i];}
 generator[0][n]=leftRate;generator[n-1][n+1]=rightRate;
 for(let i=0;i<n;i++)generator[i][i]=neg(sum(generator[i]));
 const exits=generator.map((row,i)=>neg(row[i])),clock=exits.reduce(max,zero),mu=mul(clock,time);
 if(!le(mu,f(BigInt(64))))throw new RangeError('For this preparation, maximum exit rate times physical time must be at most64; lower T or adjust the grid');
 const kernel=generator.map((row,i)=>row.map((x,j)=>clock.n===Z?(i===j?one:zero):add(i===j?one:zero,div(x,clock))));
 const explicitKernel=generator.map((row,i)=>row.map((x,j)=>add(i===j?one:zero,mul(dt,x))));
 const implicitKernel=invert(generator.map((row,i)=>row.map((x,j)=>sub(i===j?one:zero,mul(dt,x)))));
 const weights=p.initialWeights.map(BigInt),weightSum=weights.reduce((a,b)=>a+b,Z),initial=[...weights.map(w=>f(w,weightSum)),zero,zero];
 const flux=(law:readonly F[])=>[neg(mul(leftRate,law[0])),...forward.map((r,i)=>sub(mul(r,law[i]),mul(backward[i],law[i+1]))),mul(rightRate,law[n-1])];
 const record=(step:number,e:readonly F[],im:readonly F[])=>({step,time:output(mul(dt,f(BigInt(step)))),explicitLaw:e.map(output),implicitLaw:im.map(output),
  explicitDensity:e.slice(0,n).map(x=>output(div(x,dx))),implicitDensity:im.slice(0,n).map(x=>output(div(x,dx))),
  explicitFaceFlux:flux(e).map(output),implicitFaceFlux:flux(im).map(output),
  explicitInteriorMass:output(sum(e.slice(0,n))),implicitInteriorMass:output(sum(im.slice(0,n))),
  explicitGlobalMassResidual:output(sub(sum(e),one)),implicitGlobalMassResidual:output(sub(sum(im),one)),
  explicitLawIsProbability:e.every(x=>x.n>=Z),implicitLawIsProbability:im.every(x=>x.n>=Z)});
 let e=[...initial],im=[...initial];const history=[record(0,e,im)];
 for(let step=1;step<=p.eulerSteps;step++){e=rowTimes(e,explicitKernel);im=rowTimes(im,implicitKernel);history.push(record(step,e,im));}
 // A positive Poisson tail, rather than1 minus a rounded retained mass.
 const ctx=createTranscendentalContext(256),exp=ctx.expNegativeRatio(binary(mu.n),binary(mu.d)),exponential={lower:f(exp.lower,ctx.scale),upper:f(exp.upper,ctx.scale)};
 const coefficients:F[]=[one];let last=0,remainder=zero;
 if(mu.n===Z){while(coefficients.length<=p.poissonCutoff+1)coefficients.push(zero);last=p.poissonCutoff+1;}
 else for(let k=0;k<1024;k++){
  const next=div(mul(coefficients[k],mu),f(BigInt(k+1))),ratio=div(mu,f(BigInt(k+2)));
  if(k>=p.poissonCutoff+1&&le(ratio,one)&&ratio.n!==ratio.d){remainder=div(next,sub(one,ratio));if(le(remainder,mul(coefficients[p.poissonCutoff+1],f(O,O<<BigInt(256))))){last=k;break;}}
  coefficients.push(next);last=k+1;if(k===1023)throw new RangeError('Poisson tail work limit');
 }
 const suffix:F[]=Array(last+2).fill(zero);for(let k=last;k>=0;k--)suffix[k]=add(coefficients[k],suffix[k+1]);
 let law=[...initial],weighted=Array<F>(size).fill(zero),bareMass=zero;const mixtures=[];
 for(let k=0;k<=p.poissonCutoff;k++){
  weighted=weighted.map((x,i)=>add(x,mul(coefficients[k],law[i])));bareMass=add(bareMass,coefficients[k]);
  const partial=weighted.map(x=>scale(exponential,x)),tail={lower:mul(exponential.lower,suffix[k+1]),upper:min(one,mul(exponential.upper,add(suffix[k+1],remainder)))};
  const enclosures=partial.map(x=>({lower:x.lower,upper:min(one,add(x.upper,tail.upper))}));
  mixtures.push({cutoff:k,conditionalLaw:law.map(output),partialLaw:partial.map(observe),partialMass:observe(scale(exponential,bareMass)),omittedPoissonMass:observe(tail),exactGridLawEnclosures:enclosures.map(observe)});
  law=rowTimes(law,kernel);
 }
 const reference=mixtures[mixtures.length-1],ranges=reference.exactGridLawEnclosures;
 // Keep error intervals signed; no normalization of the truncated reference.
 const finalPartial=weighted.map(x=>scale(exponential,x)),tailUpper=mul(exponential.upper,add(suffix[p.poissonCutoff+1],remainder));
 const errors=(values:readonly F[])=>values.map((x,i)=>observe({lower:sub(x,min(one,add(finalPartial[i].upper,tailUpper))),upper:sub(x,finalPartial[i].lower)}));
 let stationary:F[]|null=null;
 if(p.leftBoundary==='reflecting'&&p.rightBoundary==='reflecting'&&D.n>Z){stationary=[one];for(let i=0;i<n-1;i++)stationary.push(mul(stationary[i],div(forward[i],backward[i])));const total=sum(stationary);stationary=stationary.map(x=>div(x,total));}
 return{parameters:p,precisionBits:256,stateOrder:'interior cells, escaped-left, escaped-right',cellWidth:output(dx),faces:faces.map(output),cellCenters:centers.map(output),faceDrift:velocity.map(output),
  initialLaw:initial.map(output),initialDensity:initial.slice(0,n).map(x=>output(div(x,dx))),generator:generator.map(row=>row.map(output)),
  leftEscapeRate:output(leftRate),rightEscapeRate:output(rightRate),maximumExitRate:output(clock),poissonMean:output(mu),stepTime:output(dt),
  explicitStepKernel:explicitKernel.map(row=>row.map(output)),implicitStepKernel:implicitKernel.map(row=>row.map(output)),
  diffusiveClosedStationaryLaw:stationary?.map(output)??null,
  classification:{explicitKernelIsStochastic:le(mul(dt,clock),one),explicitHistoryIsProbability:history.every(r=>r.explicitLawIsProbability),
   implicitKernelIsStochastic:implicitKernel.every(row=>row.every(x=>x.n>=Z)&&sum(row).n===sum(row).d),
   initialLawIsStationary:rowTimes(initial,generator).every(x=>x.n===Z),bothBoundariesReflecting:p.leftBoundary==='reflecting'&&p.rightBoundary==='reflecting',
   positiveOmittedPoissonMass:mu.n>Z,absorbingBoundaryHasPositiveEscape:leftRate.n>Z||rightRate.n>Z},
  history,current:history[history.length-1],mixtures,reference,
  signedExplicitTerminalError:errors(e),signedImplicitTerminalError:errors(im),exactGridTerminalLaw:ranges,
  approximationContract:'Uniform finite-volume upwind drift/centered diffusion on[-L,L], b=wind−gamma x. Interior state variables are cell masses, density=mass/dx. Reflecting faces have exactly zero flux. Absorbing faces use zero exterior inflow and zero boundary density at half-cell distance, giving diffusive escape2D/dx². Separate escaped-left/right bookkeeping states preserve global mass while interior mass may decrease. The exact-grid semigroup is enclosed by a positive unnormalized Poisson mixture with an explicit tail, not a continuum Fokker–Planck solution. Explicit and implicit Euler are exact rational discrete schemes. Explicit positivity requires h times max exit rate≤1; signed failures are retained. Implicit Euler is stochastic for nonnegative h but still has time and space error. Grid/initialization/boundary changes define different finite models. No sampled trajectories, conditional-survival renormalization, heat bath, entropy production, thermal environment or sampled particle implementation.'};
}
