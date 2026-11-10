/** Certificate-based independent response verification, Worker-side only.
 * The bounded dyadic stationary vector is an untrusted proof witness, never a
 * model parameter. Its exact gradient, residual and convexity proof are checked
 * before it seeds a separate 1280-bit Green-function Newton refinement. All
 * observable intervals must enclose that independently certified result.
 */
import { readSaddleFluctuationInput, type SaddleChainInput, type SaddleWellInput } from './learning-saddle-fluctuation-input';
import { snapshotQubitWire } from './learning-qubit-wire';
import type { SaddleFluctuationResult } from './learning-saddle-fluctuation';
import { Q,Z,O,integer,compare,abs,max,min,square,sum,NIL,UNIT,exact,interval,zero,one,add,sub,neg,mul,div,sq,scale,total,positive,negative,rational,green,root,readBounded,verifyObservation,type Band } from './learning-saddle-fluctuation-validation-math';
class Observation { constructor(readonly reference: Band) {} }
class RatioEndpoint { constructor(readonly reference:Q,readonly side:'lower'|'upper') {} }
const observation=(value:Band)=>new Observation(value),point=(x:Q)=>({lower:x.text(),upper:x.text()}),two=integer(2);
function same(actual:unknown,expected:unknown):boolean {
  if(expected instanceof Observation)return verifyObservation(actual,expected.reference);
  if(expected instanceof RatioEndpoint){
    const x=readBounded(actual),denominator=UNIT<<BigInt(512),error=expected.side==='lower'?expected.reference.sub(x):x.sub(expected.reference);
    return compare(x,Z)>=0&&compare(x,O)<=0&&x.d<=denominator&&(x.d&(x.d-UNIT))===NIL&&compare(error,Z)>=0&&compare(error,new Q(UNIT,UNIT<<BigInt(480)))<=0;
  }
  if(expected===null||typeof expected!=='object')return actual===expected;
  if(!actual||typeof actual!=='object'||Array.isArray(actual)!==Array.isArray(expected))return false;
  if(Array.isArray(expected))return Array.isArray(actual)&&actual.length===expected.length&&expected.every((x,i)=>same(actual[i],x));
  const a=actual as Record<string,unknown>,b=expected as Record<string,unknown>,keys=Object.keys(b);
  return Object.keys(a).length===keys.length&&keys.every(k=>Object.hasOwn(a,k)&&same(a[k],b[k]));
}
function norm(xs:Q[]){return xs.map(abs).reduce(max,Z);}
function chainParameters(input:Readonly<SaddleChainInput>) {
  const n=input.intervals,d=n-1,tau=Q.read(input.time),m=Q.read(input.mass),h=Q.read(input.hbar),omega=Q.read(input.omegaSquared),lambda=Q.read(input.quartic),a=Q.read(input.start),b=Q.read(input.end),delta=tau.div(integer(n)),s=m.div(delta),v=omega.mul(square(delta)),w=lambda.mul(square(delta)).div(m),muA=O.div(integer(n*n)).add(v),mu=s.mul(muA);
  const gradient=(q:Q[])=>{const path=[a,...q,b];return q.map((x,i)=>x.mul(two).sub(path[i]).sub(path[i+2]).add(v.mul(x)).add(w.mul(x).mul(x).mul(x)));};
  const diagonal=(q:Band[])=>q.map(x=>add(exact(two.add(v)),scale(sq(x),integer(3).mul(w))));
  const potential=(x:Q)=>m.mul(omega).mul(square(x)).div(two).add(lambda.mul(square(square(x))).div(integer(4)));
  const action=(q:Q[])=>{const path=[a,...q,b];return s.mul(sum(path.slice(1).map((x,i)=>square(x.sub(path[i]))))).div(two).add(delta.mul(sum(q.map(potential)).add(potential(a).add(potential(b)).div(two))));};
  return {n,d,tau,m,h,omega,lambda,a,b,delta,s,v,w,muA,mu,gradient,diagonal,potential,action};
}
/** Enumerate bivariate Gaussian MGF coefficients, independently of the
 * producer's explicit sixth/eighth-moment polynomials. */
const factorial=(n:number):number=>n<=1?1:n*factorial(n-1);
function power(a:Band,n:number):Band{let answer=one;for(let i=0;i<n;i++)answer=mul(answer,a);return answer;}
function gaussianMoment(a:number,b:number,xx:Band,yy:Band,xy:Band):Band {
  const terms:Band[]=[];
  for(let k=0;k<=Math.min(a,b);k++){
    if((a-k)%2||(b-k)%2)continue;
    const u=(a-k)/2,v=(b-k)/2,coefficient=factorial(a)*factorial(b)/(factorial(k)*factorial(u)*factorial(v)*2**(u+v));
    terms.push(scale(mul(mul(power(xx,u),power(yy,v)),power(xy,k)),integer(coefficient)));
  }
  return total(terms);
}
function reconstructChain(input:Readonly<SaddleChainInput>,value:unknown) {
  const p=chainParameters(input);
  const candidate=(value as {stationary?:{candidate?:unknown}})?.stationary?.candidate;
  if(!Array.isArray(candidate)||candidate.length!==p.n+1)throw Error('Wrong stationary witness dimension');
  const path=candidate.map(x=>readBounded(x)),q=path.slice(1,-1),quadratic=p.lambda.n===NIL;
  if(path[0].text()!==p.a.text()||path[p.n].text()!==p.b.text())throw Error('Wrong fixed endpoints');
  if(q.some(x=>compare(abs(x),integer(17))>0||(!quadratic&&(x.d>(UNIT<<BigInt(768))||(x.d&(x.d-UNIT))!==NIL))))throw Error('Out-of-budget stationary witness');
  const gA=p.gradient(q),rhoA=norm(gA),rho=p.s.mul(rhoA),radius=integer(p.d).mul(rhoA).div(p.muA);
  if(compare(radius,new Q(UNIT,UNIT<<BigInt(672)))>0||(quadratic&&rho.n!==NIL))throw Error('Uncertified root witness');
  const action=p.action(q),actionError=integer(p.d).mul(square(rho)).div(two.mul(p.mu)),eta=integer(3).mul(p.delta).mul(p.lambda).mul(radius).mul(norm(q).mul(two).add(radius));
  const bracketLo=min(Z,min(p.a,p.b)),bracketHi=max(Z,max(p.a,p.b));
  const position=q.map(x=>({lower:max(bracketLo,x.sub(radius)).text(),upper:min(bracketHi,x.add(radius)).text()}));
  if(position.some(x=>compare(Q.read(x.lower),Q.read(x.upper))>0))throw Error('Empty maximum-principle bracket');
  const freeLower=p.m.mul(square(p.b.sub(p.a))).div(two.mul(p.tau)).add(p.delta.mul(p.potential(p.a).add(p.potential(p.b))).div(two));
  // Refine only a witness already proven close to the unique root. A different
  // precision and continuant inverse replace the producer's LDL line search.
  let refined=q;
  if(!quadratic&&rhoA.n!==NIL)for(let iteration=0;iteration<3;iteration++){
    const g=p.gradient(refined),inverse=green(p.diagonal(refined.map(exact))).inverse;
    refined=refined.map((x,i)=>{
      const step=total(inverse[i].map((r,j)=>scale(r,g[j]))),next=sub(exact(x),step);
      return rational((next.lo+next.hi)/BigInt(2));
    });
  }
  const refinedError=integer(p.d).mul(norm(p.gradient(refined))).div(p.muA);
  if(compare(refinedError,new Q(UNIT,UNIT<<BigInt(1100)))>0)throw Error('Independent refinement did not certify');
  const qStar=refined.map(x=>interval(max(bracketLo,x.sub(refinedError)),min(bracketHi,x.add(refinedError))));
  const diagonal=p.diagonal(qStar),g=green(diagonal),r=g.inverse.map(row=>row.map(x=>scale(x,O.div(p.s))));
  const full=(matrix:Band[][])=>Array.from({length:p.n+1},(_,i)=>Array.from({length:p.n+1},(__,j)=>i>0&&j>0&&i<p.n&&j<p.n?observation(matrix[i-1][j-1]):point(Z)));
  const fourth=total(r.map((row,i)=>gaussianMoment(4,0,row[i],row[i],row[i])));
  let sixth=zero,eighth=zero;
  for(let i=0;i<p.d;i++)for(let j=0;j<p.d;j++){
    sixth=add(sixth,mul(mul(qStar[i],qStar[j]),gaussianMoment(3,3,r[i][i],r[j][j],r[i][j])));
    eighth=add(eighth,gaussianMoment(4,4,r[i][i],r[j][j],r[i][j]));
  }
  const cubic=positive(scale(sixth,square(p.delta.mul(p.lambda)).div(two))),quartic=negative(scale(fourth,p.delta.mul(p.lambda).div(integer(-4)))),first=add(cubic,quartic);
  const zeroSaddle=rhoA.n===NIL&&q.every(x=>x.n===NIL),epsilon=p.h.mul(p.delta).mul(p.lambda).div(integer(4));
  let ratio:unknown;
  if(quadratic)ratio={status:'exact-quadratic',lower:'1',upper:'1'};
  else if(zeroSaddle){
    const firstTerm=scale(fourth,epsilon),lower=sub(one,firstTerm),upper=add(lower,scale(eighth,square(epsilon).div(two)));
    ratio={status:'certified-zero-saddle',lower:new RatioEndpoint(max(Z,rational(lower.lo)),'lower'),upper:new RatioEndpoint(min(O,rational(upper.hi)),'upper')};
  }else ratio={status:'not-applicable-shifted-cubic',lower:null,upper:null};
  return {kind:'chain',input,timeStep:p.delta.text(),precision:{workBits:768,outputBits:512,enclosurePaddingQuanta:1},
    stationary:{candidate:path.map(x=>x.text()),position:[point(p.a),...position,point(p.b)],gradient:gA.map(x=>p.s.mul(x).text()),residualMaximum:rho.text(),curvatureLowerBound:p.mu.text(),positionErrorUpper:radius.text(),candidateAction:action.text(),minimumAction:{lower:max(freeLower,action.sub(actionError)).text(),upper:action.text()},actionErrorUpper:actionError.text(),hessianErrorUpper:eta.text(),inverseErrorUpper:eta.div(square(p.mu)).text()},
    gaussian:{hessian:diagonal.map((x,i)=>diagonal.map((_,j)=>i===j?observation(scale(x,p.s)):Math.abs(i-j)===1?observation(exact(p.s.neg())):point(Z))),inverseCurvature:full(r),covariance:full(r.map(row=>row.map(x=>scale(x,p.h)))),normalizedPivots:g.pivots.map(observation),determinant:{scale:p.s.text(),power:p.d,normalized:observation(g.determinant)},prefactorSquaredTimesPi:observation(div(exact(p.s.div(two.mul(p.h))),g.determinant))},
    correction:{cubic:observation(cubic),quartic:observation(quartic),first:observation(first),hbarFirst:observation(scale(first,p.h)),gaussianFourthSum:observation(fourth),gaussianFourthSumSquared:observation(eighth)},ratio,
    scope:{fullMatrixIntervals:p.n,integrationDimension:p.d,finiteRule:'trapezoidal-potential',normalizedHessian:'H=(m/delta)A;A off-diagonals -1',stationaryMeanInterpretation:quadratic?'exact-Gaussian-mean':zeroSaddle?'zero-mean-by-inversion-symmetry':'not-a-mean-calculation',gaussianCovarianceIsExactQuarticCovariance:quadratic,generalQuarticIntegralEvaluated:false,generalHigherOrderErrorBoundComputed:false,actualRatioEnclosureProvided:quadratic||zeroSaddle,ratioCertificateCondition:'quartic-zero or certified-zero-stationary-path',randomPathsGenerated:false}};
}
function reconstructWell(input:Readonly<SaddleWellInput>) {
  const h=Q.read(input.hbar),k=Q.read(input.kappa),positiveK=k.n>NIL,nonzero=k.n!==NIL,curvature=positiveK?k.mul(two):k.neg(),minimum=positiveK?square(k).div(integer(-4)):Z,coefficient=nonzero?integer(positiveK?3:-3).div(integer(4).mul(square(k))):null;
  const stationary=(position:unknown)=>({position,classification:nonzero?'minimum':'degenerate-minimum',action:minimum.text(),curvature:curvature.text(),gaussianVariance:nonzero?h.div(curvature).text():null});
  const positiveRoot=positiveK?root(k,2):zero;
  return {kind:'well',input,regime:positiveK?'double-well':nonzero?'single-well':'degenerate-quartic',stationaryPoints:positiveK?[stationary(observation(neg(positiveRoot))),{position:point(Z),classification:'maximum',action:'0',curvature:k.neg().text(),gaussianVariance:null},stationary(observation(positiveRoot))]:[stationary(point(Z))],minimumAction:minimum.text(),minimaIncluded:positiveK?2:1,gaussianAvailable:nonzero,firstCoefficient:coefficient?.text()??null,hbarFirst:coefficient?.mul(h).text()??null,smallParameter:nonzero?h.div(square(k)).text():null,sumPrefactorSquaredDividedByPi:nonzero?integer(positiveK?4:2).mul(h).div(abs(k)).text():null,criticalQuarticWidthScale:observation(root(h,4)),criticalWidthPower:'1/4',scope:{coordinateDimension:1,integralEvaluated:false,asymptoticRemainderBoundComputed:false,fullQuantumDoubleWellComputed:false,tunnellingRateComputed:false}};
}
const references=new Map<string,unknown>();
export function validateSaddleFluctuationResult(source:unknown,value:unknown):SaddleFluctuationResult|null {
  try {
    const input=readSaddleFluctuationInput(source);if(!input)return null;
    const owned=snapshotQubitWire(value);
    const witness=input.mode==='chain'?(owned as {stationary?:{candidate?:unknown}})?.stationary?.candidate:null;
    const key=JSON.stringify([input,witness]);
    let expected=references.get(key);
    if(!expected){
      expected=input.mode==='chain'?reconstructChain(input,owned):reconstructWell(input);
      if(references.size===4)references.delete(references.keys().next().value!);
      references.set(key,expected);
    }
    return same(owned,expected)?owned as SaddleFluctuationResult:null;
  }catch{return null;}
}
