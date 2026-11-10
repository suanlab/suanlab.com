/** Finite quartic-chain saddle certificates and one-coordinate well regimes.
 * Worker-side only. Exact residuals certify rounded Newton candidates; directed
 * normalized LDL solves enclose the Hessian at the true stationary point.
 */
import { readSaddleFluctuationInput, type SaddleChainInput, type SaddleWellInput } from './learning-saddle-fluctuation-input';
import { f, zero, one, two, add, sub, mul, div, neg, le, sum, type F } from './learning-quantum-hilbert-core';
import { createQuantumEvolutionContext, type EvolutionInterval as I } from './learning-quantum-evolution-interval';

export interface SaddleRange { lower: string; upper: string; }
const Z = BigInt(0), O = BigInt(1), T = BigInt(2), ctx = createQuantumEvolutionContext();
const workScale = O << BigInt(ctx.workBits), outputBits = 512, outputScale = O << BigInt(outputBits), shrink = O << BigInt(ctx.workBits-outputBits);
const integer = (n: number) => f(BigInt(n)), square = (x: F) => mul(x, x), cube = (x: F) => mul(square(x), x);
const read = (s: string): F => { const [n, d='1'] = s.split('/'); return f(BigInt(n), BigInt(d)); };
const text = (v: F) => v.d === O ? String(v.n) : `${v.n}/${v.d}`;
const abs = (v: F): F => v.n < Z ? neg(v) : v;
const maximum = (a: F, b: F) => le(a, b) ? b : a, minimum = (a: F, b: F) => le(a, b) ? a : b;
const floor = (n: bigint, d: bigint): bigint => n >= Z ? n/d : -((-n+d-O)/d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
const midpoint = (v: I) => (v.lower+v.upper)/T;
const sumI = (vs: I[]) => vs.reduce(ctx.add, ctx.zero);
const scale = (v: I, factor: F): I => ctx.multiply(v, ctx.from(factor));
const nonnegative = (v: I): I => ctx.intersect(v, Z, v.upper);
const nonpositive = (v: I): I => ctx.intersect(v, v.lower, Z);
const exactRange = (v: F): SaddleRange => ({ lower: text(v), upper: text(v) });
/** One output quantum of extra padding keeps a higher-precision independent
 * enclosure inside this observation, including at an output-grid boundary.
 * Analytic zeros are preserved exactly; no unknown small value becomes zero.
 */
function out(v: I, positive = false, negative = false): SaddleRange {
  if (v.lower === Z && v.upper === Z) return exactRange(zero);
  let lower=floor(v.lower,shrink)-O, upper=ceil(v.upper,shrink)+O;
  if(positive && lower<Z) lower=Z;
  if(negative && upper>Z) upper=Z;
  if(lower>upper) throw Error('Unordered saddle observation');
  return {lower:text(f(lower,outputScale)),upper:text(f(upper,outputScale))};
}
function parameters(input: Readonly<SaddleChainInput>) {
  const n=input.intervals,d=n-1,time=read(input.time),mass=read(input.mass),hbar=read(input.hbar),omega=read(input.omegaSquared),quartic=read(input.quartic),start=read(input.start),end=read(input.end),delta=div(time,integer(n));
  const s=div(mass,delta),v=mul(omega,square(delta)),w=div(mul(quartic,square(delta)),mass),muA=add(div(one,integer(n*n)),v);
  return {n,d,time,mass,hbar,omega,quartic,start,end,delta,s,v,w,muA,mu:mul(s,muA)};
}
type Parameters = ReturnType<typeof parameters>;
function gradient(p: Parameters, q: F[]): F[] {
  const x=[p.start,...q,p.end];
  return q.map((v,i)=>add(sub(sub(mul(two,v),x[i]),x[i+2]),add(mul(p.v,v),p.w.n===Z?zero:mul(p.w,cube(v)))));
}
function hessianDiagonal(p: Parameters, q: F[]): F[] {
  return q.map(x=>add(add(two,p.v),p.w.n===Z?zero:mul(integer(3),mul(p.w,square(x)))));
}
function potential(p: Parameters, x: F): F {
  return add(div(mul(mul(p.mass,p.omega),square(x)),two),p.quartic.n===Z?zero:div(mul(p.quartic,square(square(x))),integer(4)));
}
function action(p: Parameters, q: F[]): F {
  const x=[p.start,...q,p.end],kinetic=div(mul(p.s,sum(x.slice(1).map((v,i)=>square(sub(v,x[i]))))),two);
  return add(kinetic,mul(p.delta,add(div(add(potential(p,p.start),potential(p,p.end)),two),sum(q.map(v=>potential(p,v))))));
}
/** A=H/(m/delta), off diagonals -1. Its LDL pivots are >=(i+2)/(i+1).
 * Factoring A avoids underflow in det(H) for very small physical masses.
 */
function factor(diagonal: I[]) {
  const pivots: I[]=[], lower: I[]=[];
  for(let i=0;i<diagonal.length;i++) {
    if(i>0) lower.push(ctx.neg(ctx.divide(ctx.one,pivots[i-1])));
    let value=i ? ctx.sub(diagonal[i],ctx.divide(ctx.one,pivots[i-1])) : diagonal[i];
    value=ctx.intersect(value,ctx.from(f(BigInt(i+2),BigInt(i+1))).lower,diagonal[i].upper);
    if(value.lower<=Z) throw Error('Unresolved normalized positive pivot');
    pivots.push(value);
  }
  const solve=(rhs:I[])=>{
    const forward:I[]=[],answer:I[]=[];
    for(let i=0;i<diagonal.length;i++) { forward.push(i?ctx.sub(rhs[i],ctx.multiply(lower[i-1],forward[i-1])):rhs[i]); answer.push(ctx.divide(forward[i],pivots[i])); }
    for(let i=diagonal.length-2;i>=0;i--) answer[i]=ctx.sub(answer[i],ctx.multiply(lower[i],answer[i+1]));
    return answer;
  };
  return {pivots,lower,solve,determinant:pivots.reduce(ctx.multiply,ctx.one)};
}
function exactLinearStationary(p: Parameters): F[] {
  const diagonal:F[]=[],lower:F[]=[],rhs=Array.from({length:p.d},()=>zero);
  rhs[0]=add(rhs[0],p.start);rhs[p.d-1]=add(rhs[p.d-1],p.end);
  for(let i=0;i<p.d;i++) {
    if(i>0) lower.push(neg(div(one,diagonal[i-1])));
    diagonal.push(i?sub(add(two,p.v),div(one,diagonal[i-1])):add(two,p.v));
  }
  const y:F[]=[],answer:F[]=[];
  for(let i=0;i<p.d;i++){y.push(i?sub(rhs[i],mul(lower[i-1],y[i-1])):rhs[i]);answer.push(div(y[i],diagonal[i]));}
  for(let i=p.d-2;i>=0;i--) answer[i]=sub(answer[i],mul(lower[i],answer[i+1]));
  return answer;
}
const norm = (vs: F[]) => vs.reduce((s,v)=>maximum(s,abs(v)),zero);
function stationary(p: Parameters) {
  if(p.quartic.n===Z) return {candidate:exactLinearStationary(p),iterations:0,backtracks:0};
  let values=Array.from({length:p.d},(_,i)=>midpoint(ctx.from(add(p.start,mul(sub(p.end,p.start),f(BigInt(i+1),BigInt(p.n)))))));
  let q=values.map(v=>f(v,workScale)),g=gradient(p,q),rho=norm(g),backtracks=0;
  const target=f(O,O<<BigInt(672));
  for(let iteration=0;iteration<128;iteration++) {
    if(le(div(mul(integer(p.d),rho),p.muA),target)) {
      // Recognize exactly representable roots without pretending all roots are rational.
      for(const bits of [16,32,64,128,256,384,512]) {
        const denominator=O<<BigInt(bits),rounded=q.map(v=>f(floor(v.n*denominator*T+v.d,v.d*T),denominator));
        if(gradient(p,rounded).every(v=>v.n===Z)) return {candidate:rounded,iterations:iteration,backtracks};
      }
      return {candidate:q,iterations:iteration,backtracks};
    }
    const decomposition=factor(hessianDiagonal(p,q).map(ctx.from)),step=decomposition.solve(g.map(v=>ctx.from(neg(v)))).map(midpoint);
    let accepted=false;
    for(let trial=0;trial<64;trial++) {
      const denominator=O<<BigInt(trial),next=values.map((v,i)=>v+step[i]/denominator),candidate=next.map(v=>f(v,workScale));
      const nextGradient=gradient(p,candidate),nextNorm=norm(nextGradient),threshold=mul(rho,sub(one,f(O,BigInt(4)*denominator)));
      if(le(nextNorm,threshold)) {values=next;q=candidate;g=nextGradient;rho=nextNorm;backtracks+=trial;accepted=true;break;}
    }
    if(!accepted) throw Error('Saddle Newton residual line search did not resolve');
  }
  throw Error('Saddle Newton iteration budget exhausted');
}
function moments(p: Parameters, q: I[], r: I[][]) {
  const diagonals=r.map((row,i)=>row[i]),m1=scale(sumI(diagonals.map(ctx.square)),integer(3));
  let cubic=ctx.zero,m2=ctx.zero;
  for(let i=0;i<p.d;i++) for(let j=0;j<p.d;j++) {
    const cross=r[i][j],crossSquared=ctx.square(cross),same=ctx.multiply(diagonals[i],diagonals[j]);
    const sixth=ctx.add(scale(ctx.multiply(same,cross),integer(9)),scale(ctx.multiply(crossSquared,cross),integer(6)));
    cubic=ctx.add(cubic,ctx.multiply(ctx.multiply(q[i],q[j]),sixth));
    const eighth=ctx.add(ctx.add(scale(ctx.square(same),integer(9)),scale(ctx.multiply(same,crossSquared),integer(72))),scale(ctx.square(crossSquared),integer(24)));
    m2=ctx.add(m2,eighth);
  }
  const a=mul(p.delta,p.quartic),c3=nonnegative(scale(cubic,div(square(a),two))),c4=nonpositive(ctx.neg(scale(m1,div(a,integer(4)))));
  return {firstMoment:nonnegative(m1),secondMoment:nonnegative(m2),cubic:c3,quartic:c4,first:ctx.add(c3,c4)};
}
function calculateChain(input: Readonly<SaddleChainInput>) {
  const p=parameters(input),solved=stationary(p),q=solved.candidate,gA=gradient(p,q),rhoA=norm(gA),rho=mul(p.s,rhoA);
  const radius=div(mul(integer(p.d),rhoA),p.muA),actionError=div(mul(integer(p.d),square(rho)),mul(two,p.mu)),candidateAction=action(p,q);
  const lowerPosition=minimum(minimum(p.start,p.end),zero),upperPosition=maximum(maximum(p.start,p.end),zero);
  const position=q.map(v=>({lower:maximum(lowerPosition,sub(v,radius)),upper:minimum(upperPosition,add(v,radius))}));
  if(position.some(v=>!le(v.lower,v.upper))) throw Error('Stationary certificate missed the maximum-principle bracket');
  const qRange=position.map(ctx.range),aDiagonal=qRange.map(v=>ctx.add(ctx.from(add(two,p.v)),scale(ctx.square(v),mul(integer(3),p.w))));
  const decomposition=factor(aDiagonal),columns=Array.from({length:p.d},(_,j)=>decomposition.solve(Array.from({length:p.d},(__,i)=>i===j?ctx.one:ctx.zero)));
  const inverse=Array.from({length:p.d},(_,i)=>Array.from({length:p.d},(__,j)=>{
    const a=columns[j][i],b=columns[i][j],free=f(BigInt((Math.min(i,j)+1)*(p.n-Math.max(i,j)-1)),BigInt(p.n));
    return ctx.intersect({lower:a.lower>b.lower?a.lower:b.lower,upper:a.upper<b.upper?a.upper:b.upper},Z,ctx.from(free).upper);
  }));
  const r=inverse.map(row=>row.map(v=>nonnegative(scale(v,div(one,p.s))))),physical=r.map(row=>row.map(v=>nonnegative(scale(v,p.hbar))));
  const full=(matrix:I[][])=>Array.from({length:p.n+1},(_,i)=>Array.from({length:p.n+1},(__,j)=>i>0&&i<p.n&&j>0&&j<p.n?out(matrix[i-1][j-1],true):exactRange(zero)));
  const hessian=aDiagonal.map((v,i)=>Array.from({length:p.d},(_,j)=>out(i===j?scale(v,p.s):Math.abs(i-j)===1?ctx.from(neg(p.s)):ctx.zero)));
  const correction=moments(p,qRange,r),epsilon=div(mul(mul(p.hbar,p.delta),p.quartic),integer(4));
  const quadratic=p.quartic.n===Z,zeroSaddle=rhoA.n===Z&&q.every(v=>v.n===Z);
  let ratio: {status:'exact-quadratic'|'certified-zero-saddle'|'not-applicable-shifted-cubic';lower:string|null;upper:string|null};
  if(quadratic) ratio={status:'exact-quadratic',lower:'1',upper:'1'};
  else if(zeroSaddle) {
    const first=scale(correction.firstMoment,epsilon),upper=ctx.add(ctx.sub(ctx.one,first),scale(correction.secondMoment,div(square(epsilon),two)));
    const lo=ctx.sub(ctx.one,first).lower,hi=upper.upper;
    ratio={status:'certified-zero-saddle',lower:text(f(lo>Z?floor(lo,shrink):Z,outputScale)),upper:text(f(hi<workScale?ceil(hi,shrink):outputScale,outputScale))};
  } else ratio={status:'not-applicable-shifted-cubic',lower:null,upper:null};
  const endCost=mul(div(p.delta,two),add(potential(p,p.start),potential(p,p.end))),freeLower=add(div(mul(p.mass,square(sub(p.end,p.start))),mul(two,p.time)),endCost),minimumLower=maximum(freeLower,sub(candidateAction,actionError));
  const hessianError=mul(mul(integer(3),mul(p.delta,p.quartic)),mul(radius,add(mul(two,norm(q)),radius)));
  return {
    kind:'chain' as const,input,timeStep:text(p.delta),precision:{workBits:ctx.workBits,outputBits,enclosurePaddingQuanta:1},
    stationary:{candidate:[text(p.start),...q.map(text),text(p.end)],position:[exactRange(p.start),...position.map(v=>({lower:text(v.lower),upper:text(v.upper)})),exactRange(p.end)],
      gradient:gA.map(v=>text(mul(p.s,v))),residualMaximum:text(rho),curvatureLowerBound:text(p.mu),positionErrorUpper:text(radius),candidateAction:text(candidateAction),minimumAction:{lower:text(minimumLower),upper:text(candidateAction)},actionErrorUpper:text(actionError),hessianErrorUpper:text(hessianError),inverseErrorUpper:text(div(hessianError,square(p.mu)))},
    gaussian:{hessian,inverseCurvature:full(r),covariance:full(physical),normalizedPivots:decomposition.pivots.map(v=>out(v,true)),
      determinant:{scale:text(p.s),power:p.d,normalized:out(decomposition.determinant,true)},prefactorSquaredTimesPi:out(ctx.divide(ctx.from(div(p.s,mul(two,p.hbar))),decomposition.determinant),true)},
    correction:{cubic:out(correction.cubic,true),quartic:out(correction.quartic,false,true),first:out(correction.first),hbarFirst:out(scale(correction.first,p.hbar)),gaussianFourthSum:out(correction.firstMoment,true),gaussianFourthSumSquared:out(correction.secondMoment,true)},
    ratio,
    scope:{fullMatrixIntervals:p.n,integrationDimension:p.d,finiteRule:'trapezoidal-potential' as const,normalizedHessian:'H=(m/delta)A;A off-diagonals -1' as const,stationaryMeanInterpretation:quadratic?'exact-Gaussian-mean' as const:zeroSaddle?'zero-mean-by-inversion-symmetry' as const:'not-a-mean-calculation' as const,gaussianCovarianceIsExactQuarticCovariance:quadratic,
      generalQuarticIntegralEvaluated:false,generalHigherOrderErrorBoundComputed:false,actualRatioEnclosureProvided:ratio.lower!==null,ratioCertificateCondition:'quartic-zero or certified-zero-stationary-path' as const,randomPathsGenerated:false},
  };
}
function calculateWell(input: Readonly<SaddleWellInput>) {
  const hbar=read(input.hbar),kappa=read(input.kappa),positive=kappa.n>Z,negative=kappa.n<Z,curvature=positive?mul(two,kappa):neg(kappa);
  const root=positive?ctx.sqrt(kappa):ctx.zero,minimumAction=positive?neg(div(square(kappa),integer(4))):zero;
  const minima=positive?2:1,coefficient=kappa.n===Z?null:div(positive?integer(3):integer(-3),mul(integer(4),square(kappa)));
  const minimum=(position:I)=>({position:out(position),classification:kappa.n===Z?'degenerate-minimum' as const:'minimum' as const,action:text(minimumAction),curvature:text(curvature),gaussianVariance:kappa.n===Z?null:text(div(hbar,curvature))});
  const criticalScale=(()=>{const first=ctx.rationalRange(ctx.sqrt(hbar));return{lower:ctx.sqrt(first.lower).lower,upper:ctx.sqrt(first.upper).upper};})();
  return {kind:'well' as const,input,regime:positive?'double-well' as const:negative?'single-well' as const:'degenerate-quartic' as const,
    stationaryPoints:positive?[minimum(ctx.neg(root)),{position:exactRange(zero),classification:'maximum' as const,action:'0',curvature:text(neg(kappa)),gaussianVariance:null},minimum(root)]:[minimum(ctx.zero)],
    minimumAction:text(minimumAction),minimaIncluded:minima,gaussianAvailable:kappa.n!==Z,firstCoefficient:coefficient?text(coefficient):null,hbarFirst:coefficient?text(mul(hbar,coefficient)):null,
    smallParameter:kappa.n===Z?null:text(div(hbar,square(kappa))),sumPrefactorSquaredDividedByPi:kappa.n===Z?null:text(div(mul(integer(positive?4:2),hbar),abs(kappa))),
    criticalQuarticWidthScale:out(criticalScale,true),criticalWidthPower:'1/4' as const,
    scope:{coordinateDimension:1,integralEvaluated:false,asymptoticRemainderBoundComputed:false,fullQuantumDoubleWellComputed:false,tunnellingRateComputed:false},
  };
}
export function calculateSaddleFluctuation(value: unknown) {
  const input=readSaddleFluctuationInput(value);
  if(!input) throw RangeError('Invalid owned saddle-fluctuation input');
  return input.mode==='chain'?calculateChain(input):calculateWell(input);
}
export type SaddleFluctuationResult=ReturnType<typeof calculateSaddleFluctuation>;
