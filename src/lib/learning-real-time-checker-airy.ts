/** Independent fold reference: exact rational Taylor terms andfactorial-form
 * asymptotic coefficients,with explicit analytic remainders. No producer math.
 */
import {readRealTimePhaseInput,type RealTimeFoldInput} from './learning-real-time-phase-input';
import {CheckerRational as Q} from './learning-real-time-validation-rational';
import {createRealTimeCheckerInterval,checkerRealTimeRotation,checkerFloor,type CheckerInterval as I} from './learning-real-time-checker-interval';
import {checkerRealTimeAiryOrigin} from './learning-real-time-checker-airy-origin';
const Z=BigInt(0),O=BigInt(1),T=BigInt(2);
const ctx=createRealTimeCheckerInterval(6144);
const absolute=(x:I)=>[x.lower<Z?-x.lower:x.lower,x.upper<Z?-x.upper:x.upper].reduce((a,b)=>a>b?a:b);
const signedTail=(sum:I,bound:bigint,negative:boolean)=>negative?{lower:sum.lower-bound,upper:sum.upper}:{lower:sum.lower,upper:sum.upper+bound};
export type CheckerAiryMethod='central-ode-series'|'positive-scaled-asymptotic'|'oscillatory-asymptotic';
function parameters(input:RealTimeFoldInput) {
  const u=Q.input(input.control),h=Q.input(input.hbar),a=u.n<Z?u.neg():u;
  const cube=a.mul(a).mul(a).div(h.mul(h));
  const z3=u.n>Z?cube.neg():cube,zetaSquared=cube.mul(Q.of(BigInt(4),BigInt(9)));
  return{u,h,a,cube,z3,zetaSquared};
}
function rotation(angle:I) {
  const r=checkerRealTimeRotation({n:angle.lower+angle.upper,d:T*ctx.S},1536),lift=O<<BigInt(ctx.bits-r.bits);
  const radius=-checkerFloor(-(angle.upper-angle.lower),T);
  const widen=(x:I):I=>({lower:x.lower*lift-radius< -ctx.S?-ctx.S:x.lower*lift-radius,upper:x.upper*lift+radius>ctx.S?ctx.S:x.upper*lift+radius});
  return{cos:widen(r.cos),sin:widen(r.sin)};
}
function central(z:I,z3:Q) {
  const origin=checkerRealTimeAiryOrigin(),target=O<<BigInt(ctx.bits-2304);
  let numerator=O,denominator=O,df=O,dg=O,sa=ctx.one,sb=ctx.one;
  for(let k=1;k<=2048;k++) {
    const j=BigInt(3*k);numerator*=z3.n;denominator*=z3.d;df*=j*(j-O);dg*=j*(j+O);
    const a=ctx.from({n:numerator,d:denominator*df}),b=ctx.from({n:numerator,d:denominator*dg});
    if(absolute(a)<target&&absolute(ctx.multiply(z,b))<target&&j*(j-O)*z3.d>=T*(z3.n<Z?-z3.n:z3.n)) {
      const fa={lower:sa.lower-T*absolute(a),upper:sa.upper+T*absolute(a)},fb={lower:sb.lower-T*absolute(b),upper:sb.upper+T*absolute(b)};
      return ctx.add(ctx.multiply(origin.value,fa),ctx.multiply(origin.derivative,ctx.multiply(z,fb)));
    }
    sa=ctx.add(sa,a);sb=ctx.add(sb,b);
  }
  throw Error('Checker central Airy remainder budget');
}
/** u_j=(6j)!/[864^j(3j)!(2j)!]. Even powers ofzeta are exact rationals;
 * odd powers need only one directed division bysqrt(zetaSquared).
 */
function termStream(zetaSquared:Q,zeta:I) {
  let f6=O,f3=O,f2=O,p864=O,pn=O,pd=O,last=-1;
  return(j:number):I=>{
    if(j!==last+1)throw Error('Sequential checker Airy coefficients');last=j;
    if(j) {
      for(let r=6*j-5;r<=6*j;r++)f6*=BigInt(r);
      for(let r=3*j-2;r<=3*j;r++)f3*=BigInt(r);
      f2*=BigInt(2*j-1)*BigInt(2*j);p864*=BigInt(864);
      if(j%2===0){pn*=zetaSquared.n;pd*=zetaSquared.d;}
    }
    const q=ctx.from({n:f6*pd,d:p864*f3*f2*pn});return j%2?ctx.divide(q,zeta):q;
  };
}
function asymptotic(x:I,zetaSquared:Q,zeta:I,positive:boolean) {
  const term=termStream(zetaSquared,zeta),prefactor=ctx.divide(ctx.one,ctx.multiply(ctx.root(ctx.pi(),2),ctx.root(x,4)));
  const target=O<<BigInt(ctx.bits-768);let even=term(0),odd=ctx.zero,pending=ctx.zero;
  for(let j=1;j<=1024;j++) {
    const t=term(j);
    if(positive) {
      if(t.upper<target)return ctx.divideInteger(ctx.multiply(prefactor,signedTail(even,t.upper,j%2===1)),2);
      even=j%2?ctx.sub(even,t):ctx.add(even,t);
    }else if(j===1)odd=t;
    else if(j%2===0)pending=t;
    else {
      const k=(j-1)/2;
      if(pending.upper<target&&t.upper<target) {
        const e=signedTail(even,pending.upper,k%2===1),o=signedTail(odd,t.upper,k%2===1);
        const phase=rotation(ctx.sub(zeta,ctx.divideInteger(ctx.pi(),4)));
        return ctx.multiply(prefactor,ctx.add(ctx.multiply(phase.cos,e),ctx.multiply(phase.sin,o)));
      }
      even=k%2?ctx.sub(even,pending):ctx.add(even,pending);odd=k%2?ctx.sub(odd,t):ctx.add(odd,t);
    }
  }
  throw Error('Checker asymptotic Airy remainder budget');
}

/** The work count is checked as a valid truncation witness,not an execution
 * transcript of a particular implementation's first stopping decision.
 */
export function checkerRealTimeAiryTerms(source:unknown,method:CheckerAiryMethod,count:number) {
  try {
    const input=readRealTimePhaseInput(source);if(!input||input.mode!=='fold'||!Number.isSafeInteger(count)||count<1)return false;
    const p=parameters(input);
    if(method==='central-ode-series') {
      if(count>2048||p.cube.n>BigInt(72)**BigInt(3)*p.cube.d)return false;
      let df=O,dg=O;for(let k=1;k<=count;k++){const j=BigInt(3*k);df*=j*(j-O);dg*=j*(j+O);}
      const n=p.z3.n**BigInt(count),d=p.z3.d**BigInt(count),z=ctx.root(ctx.from(p.cube),3),threshold=O<<BigInt(ctx.bits-2047),j=BigInt(3*count);
      return absolute(ctx.from({n,d:d*df}))<=threshold&&absolute(ctx.multiply(z,ctx.from({n,d:d*dg})))<=threshold&&j*(j-O)*p.cube.d>=T*p.cube.n;
    }
    if(count>1024||p.cube.n<BigInt(64)**BigInt(3)*p.cube.d||
      method!==(p.u.n<Z?'positive-scaled-asymptotic':'oscillatory-asymptotic')||p.u.n===Z)return false;
    if(p.u.n>Z&&(count<2||count%2!==0))return false;
    const zeta=ctx.root(ctx.from(p.zetaSquared),2),term=termStream(p.zetaSquared,zeta),threshold=O<<BigInt(ctx.bits-639);
    let first=ctx.zero,second=ctx.zero;
    for(let j=0;j<=count+(p.u.n>Z?1:0);j++){const t=term(j);if(j===count)first=t;else second=t;}
    return first.upper<=threshold&&(p.u.n<Z||second.upper<=threshold);
  }catch{return false;}
}

export function checkerRealTimeFoldReference(source:unknown,method:CheckerAiryMethod) {
  const input=readRealTimePhaseInput(source);if(!input||input.mode!=='fold')throw Error('Checker physical fold input');
  const p=parameters(input),positive=p.u.n<Z;
  const centralMethod=method==='central-ode-series';
  if(centralMethod?p.cube.n>BigInt(72)**BigInt(3)*p.cube.d:
    method!==(positive?'positive-scaled-asymptotic':'oscillatory-asymptotic')||p.u.n===Z||p.cube.n<BigInt(64)**BigInt(3)*p.cube.d)throw Error('Checker Airy method domain');
  const x=ctx.root(ctx.from(p.cube),3),z=positive?x:ctx.neg(x),zeta=ctx.root(ctx.from(p.zetaSquared),2),hThird=ctx.root(ctx.from(p.h),3);
  const exponential=positive?ctx.expNegative(zeta):ctx.one,decay=positive?zeta:ctx.zero;
  const evaluated=centralMethod?central(z,p.z3):asymptotic(x,p.zetaSquared,zeta,positive);
  const scaledAiry=centralMethod&&positive?ctx.divide(evaluated,exponential):evaluated;
  if(positive&&scaledAiry.lower<=Z)throw Error('Checker positive Airy lost positivity');
  const airy=positive?ctx.multiply(scaledAiry,exponential):scaledAiry,factor=ctx.multiply(ctx.scale(ctx.pi(),2),hThird);
  const scaledIntegral=ctx.multiply(factor,scaledAiry),integral=positive?ctx.multiply(scaledIntegral,exponential):scaledIntegral;
  const root=ctx.root(ctx.from(p.a),2),action=ctx.divideInteger(ctx.scale(ctx.multiply(ctx.from(p.a),root),2),3),curvature=ctx.scale(root,2);
  const stationary=p.u.n>Z?[
    {position:ctx.neg(root),action,curvature:ctx.neg(curvature)},
    {position:root,action:ctx.neg(action),curvature}]:p.u.n===Z?[{position:ctx.zero,action:ctx.zero,curvature:ctx.zero}]:[];
  const complexStationary=p.u.n<Z?[
    {position:{real:ctx.zero,imaginary:root},action:{real:ctx.zero,imaginary:action},curvature:{real:ctx.zero,imaginary:curvature},contributesToRealContour:true},
    {position:{real:ctx.zero,imaginary:ctx.neg(root)},action:{real:ctx.zero,imaginary:ctx.neg(action)},curvature:{real:ctx.zero,imaginary:ctx.neg(curvature)},contributesToRealContour:false}]:[];
  const stationaryApproximation=p.u.n>Z?ctx.multiply(ctx.scale(ctx.root(ctx.divide(ctx.multiply(ctx.pi(),ctx.from(p.h)),root),2),2),rotation(ctx.sub(zeta,ctx.divideInteger(ctx.pi(),4))).cos):null;
  const absoluteIntegral=scaledIntegral.upper<Z?ctx.neg(scaledIntegral):scaledIntegral;
  const logMagnitude=absoluteIntegral.lower>Z?ctx.sub(ctx.log(absoluteIntegral),decay):null;
  return{ctx,input,argument:z,phaseScale:zeta,airy,scaledAiry,integral,scaledIntegral,negativeExponential:decay,logMagnitude,
    stationaryPoints:stationary,complexStationaryPoints:complexStationary,stationaryApproximation};
}
