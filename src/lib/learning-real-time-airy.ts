/** Canonical fold integral across the complete physical u/hbar input range.
 * Central ODE series and real-axis asymptotics have explicit analytic tails.
 * Positive-argument Airy decay is retained as a scaled value plus exponent.
 */
import {readRealTimePhaseInput} from './learning-real-time-phase-input';
import {realTimeFraction as read} from './learning-real-time-phase-exact';
import {mul,div,neg,type F} from './learning-quantum-hilbert-core';
import {createRealTimeIntervalContext,type RealTimeInterval as I,type RealTimeRange} from './learning-real-time-interval';
import {certifiedRealTimeAiryOrigin} from './learning-real-time-airy-origin';
const ctx=createRealTimeIntervalContext(),Z=BigInt(0),O=BigInt(1),T=BigInt(2);
const maxAbs=(v:I)=>{const a=v.lower<Z?-v.lower:v.lower,b=v.upper<Z?-v.upper:v.upper;return a>b?a:b;};
const absF=(x:F)=>x.n<Z?neg(x):x;
const signedRemainder=(sum:I,bound:bigint,negative:boolean)=>ctx.add(sum,negative?{lower:-bound,upper:Z}:{lower:Z,upper:bound});
function central(z:I,z3:I) {
  const origin=certifiedRealTimeAiryOrigin(),target=O<<BigInt(ctx.workBits-2048);
  let a=ctx.one,b=z,sa=a,sb=b;
  for(let k=0;k<2048;k++) {
    const da=BigInt(3*k+3)*BigInt(3*k+2),db=BigInt(3*k+4)*BigInt(3*k+3);
    const na=ctx.divideInteger(ctx.multiply(a,z3),da),nb=ctx.divideInteger(ctx.multiply(b,z3),db);
    if(maxAbs(na)<target&&maxAbs(nb)<target&&da*ctx.S>=T*maxAbs(z3)) {
      // Subsequent absolute ratios decrease and are <=1/2. Each full
      // tail is at most twice its first omitted absolute term.
      const ta=T*maxAbs(na),tb=T*maxAbs(nb);
      const fa={lower:sa.lower-ta,upper:sa.upper+ta},fb={lower:sb.lower-tb,upper:sb.upper+tb};
      return {value:ctx.add(ctx.multiply(origin.value,fa),ctx.multiply(origin.derivative,fb)),terms:k+1};
    }
    sa=ctx.add(sa,na);sb=ctx.add(sb,nb);a=na;b=nb;
  }
  throw new RangeError('Central Airy tail did not reach its certified target');
}
function nextTerm(previous:I,j:number,zeta:I):I {
  const k=BigInt(j),numerator=(BigInt(6)*k-BigInt(5))*(BigInt(6)*k-BigInt(3))*(BigInt(6)*k-O);
  const denominator=(T*k-O)*BigInt(216)*k;
  return ctx.divideInteger(ctx.scale(ctx.divide(previous,zeta),numerator),denominator);
}
function asymptotic(x:I,zeta:I,positiveArgument:boolean) {
  const prefactor=ctx.divide(ctx.one,ctx.multiply(ctx.root(ctx.pi(),2),ctx.root(x,4)));
  const target=O<<BigInt(ctx.workBits-640);
  if(positiveArgument) {
    let term=ctx.one,sum=ctx.one;
    for(let j=1;j<=1024;j++) {
      term=nextTerm(term,j,zeta);
      if(term.upper<=target) {
        const bounded=signedRemainder(sum,term.upper,j%2===1);
        return {value:ctx.divideInteger(ctx.multiply(prefactor,bounded),2),terms:j};
      }
      sum=j%2?ctx.sub(sum,term):ctx.add(sum,term);
    }
  } else {
    const terms=[ctx.one,nextTerm(ctx.one,1,zeta)];let even=terms[0],odd=terms[1];
    for(let k=1;k<=512;k++) {
      terms.push(nextTerm(terms[2*k-1],2*k,zeta));terms.push(nextTerm(terms[2*k],2*k+1,zeta));
      if(terms[2*k].upper<=target&&terms[2*k+1].upper<=target) {
        even=signedRemainder(even,terms[2*k].upper,k%2===1);odd=signedRemainder(odd,terms[2*k+1].upper,k%2===1);
        const phase=ctx.sinCos(zeta),halfRoot=ctx.root(ctx.divideInteger(ctx.one,2),2);
        const cosine=ctx.multiply(ctx.add(phase.cos,phase.sin),halfRoot),sine=ctx.multiply(ctx.sub(phase.sin,phase.cos),halfRoot);
        return {value:ctx.multiply(prefactor,ctx.add(ctx.multiply(cosine,even),ctx.multiply(sine,odd))),terms:2*k};
      }
      even=k%2?ctx.sub(even,terms[2*k]):ctx.add(even,terms[2*k]);
      odd=k%2?ctx.sub(odd,terms[2*k+1]):ctx.add(odd,terms[2*k+1]);
    }
  }
  throw new RangeError('Airy asymptotic tail did not reach its certified target');
}
const nonnegativeRange=(a:I):RealTimeRange=>{
  const observed=ctx.observe(a);return{lower:read(observed.lower).n<Z?'0':observed.lower,upper:observed.upper};
};
export type RealTimeAiryStrategy='auto'|'central'|'asymptotic';
export function calculateRealTimeFold(value:unknown,strategy:RealTimeAiryStrategy='auto') {
  const input=readRealTimePhaseInput(value);
  if(!input||input.mode!=='fold')throw new RangeError('Owned real-time fold input required');
  if(!['auto','central','asymptotic'].includes(strategy))throw new RangeError('Unknown Airy strategy');
  const u=read(input.control),h=read(input.hbar),uAbs=absF(u),uI=ctx.from(uAbs),hI=ctx.from(h);
  const hThird=ctx.root(hI,3),x=ctx.divide(uI,ctx.square(hThird)),positiveArgument=u.n<Z;
  const z=positiveArgument?x:ctx.neg(x);
  const z3Exact=neg(div(mul(mul(u,u),u),mul(h,h))),z3=ctx.from(z3Exact);
  const magnitudeCubed=absF(z3Exact),atMost=(bound:number)=>magnitudeCubed.n<=BigInt(bound)**BigInt(3)*magnitudeCubed.d;
  const centralSelected=strategy==='central'||strategy==='auto'&&atMost(64);
  if(strategy==='central'&&!atMost(72))throw new RangeError('Central overlap path is bounded by|z|<=72');
  if(strategy==='asymptotic'&&magnitudeCubed.n<BigInt(64)**BigInt(3)*magnitudeCubed.d)throw new RangeError('Asymptotic overlap path requires|z|>=64');
  const zeta=ctx.divideInteger(ctx.divide(ctx.scale(ctx.multiply(uI,ctx.root(uI,2)),2),hI),3);
  const decay=positiveArgument?zeta:ctx.zero;
  const evaluation=centralSelected?central(z,z3):asymptotic(x,zeta,positiveArgument);
  let scaledAi=evaluation.value;
  if(centralSelected&&positiveArgument)scaledAi=ctx.divide(scaledAi,ctx.expNegative(zeta));
  if(positiveArgument&&scaledAi.lower<=Z)throw Error('Positive Airy scaled enclosure lost positivity');
  const rawAi=positiveArgument?ctx.multiply(scaledAi,ctx.expNegative(zeta)):scaledAi;
  const factor=ctx.multiply(ctx.scale(ctx.pi(),2),hThird);
  const scaledIntegral=ctx.multiply(factor,scaledAi),integral=positiveArgument?ctx.multiply(scaledIntegral,ctx.expNegative(zeta)):scaledIntegral;
  const scaledObserved=positiveArgument?nonnegativeRange(scaledIntegral):ctx.observe(scaledIntegral);
  const scaledSign=read(scaledObserved.lower).n>Z?'positive':read(scaledObserved.upper).n<Z?'negative':'unresolved';
  const absolute=scaledSign==='negative'?ctx.neg(scaledIntegral):scaledIntegral;
  const logMagnitude=scaledSign==='unresolved'?null:ctx.observe(ctx.sub(ctx.log(absolute),decay));
  let stationaryApproximation:RealTimeRange|null=null;
  let stationaryPoints:{position:RealTimeRange;action:RealTimeRange;curvature:RealTimeRange}[]=[];
  let complexStationaryPoints:{position:{real:RealTimeRange;imaginary:RealTimeRange};action:{real:RealTimeRange;imaginary:RealTimeRange};curvature:{real:RealTimeRange;imaginary:RealTimeRange};contributesToRealContour:boolean}[]=[];
  if(u.n>Z) {
    const root=ctx.root(uI,2),action=ctx.divideInteger(ctx.scale(ctx.multiply(uI,root),2),3),curvature=ctx.scale(root,2);
    stationaryPoints=[{position:ctx.observe(ctx.neg(root)),action:ctx.observe(action),curvature:ctx.observe(ctx.neg(curvature))},{position:ctx.observe(root),action:ctx.observe(ctx.neg(action)),curvature:ctx.observe(curvature)}];
    const phase=ctx.sinCos(zeta),cosine=ctx.multiply(ctx.add(phase.cos,phase.sin),ctx.root(ctx.divideInteger(ctx.one,2),2));
    const amplitude=ctx.scale(ctx.root(ctx.divide(ctx.multiply(ctx.pi(),hI),root),2),2);
    stationaryApproximation=ctx.observe(ctx.multiply(amplitude,cosine));
  } else if(u.n===Z)stationaryPoints=[{position:{lower:'0',upper:'0'},action:{lower:'0',upper:'0'},curvature:{lower:'0',upper:'0'}}];
  else {
    const root=ctx.root(uI,2),action=ctx.divideInteger(ctx.scale(ctx.multiply(uI,root),2),3),curvature=ctx.scale(root,2),zero={lower:'0',upper:'0'};
    complexStationaryPoints=[{position:{real:zero,imaginary:ctx.observe(root)},action:{real:zero,imaginary:ctx.observe(action)},curvature:{real:zero,imaginary:ctx.observe(curvature)},contributesToRealContour:true},
      {position:{real:zero,imaginary:ctx.observe(ctx.neg(root))},action:{real:zero,imaginary:ctx.observe(ctx.neg(action))},curvature:{real:zero,imaginary:ctx.observe(ctx.neg(curvature))},contributesToRealContour:false}];
  }
  return {kind:'canonical-fold' as const,input,method:centralSelected?'central-ode-series' as const:positiveArgument?'positive-scaled-asymptotic' as const:'oscillatory-asymptotic' as const,
    regime:u.n>Z?'two-real-stationary-points' as const:u.n===Z?'merged-stationary-point' as const:'no-real-stationary-points' as const,
    argument:ctx.observe(z),phaseScale:ctx.observe(zeta),airy:positiveArgument?nonnegativeRange(rawAi):ctx.observe(rawAi),
    scaledAiry:positiveArgument?nonnegativeRange(scaledAi):ctx.observe(scaledAi),integral:positiveArgument?nonnegativeRange(integral):ctx.observe(integral),
    scaledIntegral:scaledObserved,negativeExponential:ctx.observe(decay),logMagnitude,scaledSign,
    rawValueBelowObservationQuantum:positiveArgument&&integral.upper<(O<<BigInt(ctx.workBits-512)),
    stationaryPoints,complexStationaryPoints,stationaryApproximation,terms:evaluation.terms,workingBits:ctx.workBits,observationBits:512,
    relativeLogResolved:scaledSign!=='unresolved',exactZeroClaimed:false as const};
}
/** Expose origin observations for independent tests and teaching annotations. */
export function observeRealTimeAiryOrigin() {
  const origin=certifiedRealTimeAiryOrigin();return{value:ctx.observe(origin.value),derivative:ctx.observe(origin.derivative)};
}
