/** Undamped hardening Duffing oscillator. Certified AGM/nome Fourier reference,
 * finite Lindstedt polynomials and unwrapped fundamental phase diagnostics.
 */
import {f,zero,one,add,sub,mul,div,neg,eq,le,polyAdd,polyScale,polyMul,sqrtRange,type F,type P} from './learning-quantum-hilbert-core';
import {createQuantumEvolutionContext,type EvolutionInterval as I} from './learning-quantum-evolution-interval';
import {createTranscendentalContext} from './learning-interval-transcendentals';
import {directedPi} from './learning-directed-pi';
import {readDuffingSecularInput,type DuffingSecularInput} from './learning-duffing-secular-input';
const B=BigInt,outputBits=192,outputScale=B(1)<<B(outputBits),N=64;
const integer=(n:number)=>f(B(n)),parse=(s:string)=>{const[n,d='1']=s.split('/');return f(B(n),B(d));},text=(a:F)=>a.d===B(1)?String(a.n):`${a.n}/${a.d}`;
const abs=(a:F)=>a.n<B(0)?neg(a):a,min=(a:F,b:F)=>le(a,b)?a:b,max=(a:F,b:F)=>le(a,b)?b:a;
const floor=(n:bigint,d:bigint)=>n>=B(0)?n/d:-((-n+d-B(1))/d),ceil=(n:bigint,d:bigint)=>-floor(-n,d);
export function calculateDuffingSecular(value:DuffingSecularInput){
 const input=readDuffingSecularInput(value);if(!input)throw new RangeError('Invalid Duffing input');
 const lambda=parse(input.lambda),A=parse(input.amplitude),H=parse(input.horizon),probeTime=mul(H,parse(input.probeFraction)),g=mul(lambda,mul(A,A)),g2=mul(g,g),c=createQuantumEvolutionContext(),q=(a:F)=>c.from(a),z=c.zero,o=c.one;
 const times=c.multiply,plus=c.add,minus=c.sub,scaled=(v:I,a:F)=>times(v,q(a)),power=(v:I,n:number)=>{let r=o;for(let j=0;j<n;j++)r=times(r,v);return r;};
 const root=(v:I):I=>{const r=c.rationalRange(v);return c.range({lower:sqrtRange(r.lower).lower,upper:sqrtRange(r.upper).upper});};
 const wire=(v:I)=>{const r=c.rationalRange(v);return{lower:text(f(floor(r.lower.n*outputScale,r.lower.d),outputScale)),upper:text(f(ceil(r.upper.n*outputScale,r.upper.d),outputScale))};};
 const absolute=(v:I):I=>({lower:v.lower<=B(0)&&v.upper>=B(0)?B(0):minBig(absBig(v.lower),absBig(v.upper)),upper:maxBig(absBig(v.lower),absBig(v.upper))});
 const distance=(a:I,b:I)=>wire(absolute(minus(a,b)));
 const piRaw=directedPi(512),pi=c.range({lower:f(piRaw.lower,B(1)<<B(512)),upper:f(piRaw.upper,B(1)<<B(512))});
 const agm=(parameter:F)=>{let a=o,b=root(q(sub(one,parameter)));for(let j=0;j<18;j++){const next=scaled(plus(a,b),f(B(1),B(2)));b=root(times(a,b));a=next;}return{lower:b.lower,upper:a.upper};};
 const mode=eq(A,zero)?'equilibrium' as const:eq(g,zero)?'harmonic' as const:'nonlinear-periodic' as const;
 const m=div(g,mul(integer(2),add(one,g))),M=eq(g,zero)?o:agm(m),omega=eq(g,zero)?o:times(root(q(add(one,g))),M);
 let nome:I|null=null,prefactor=o,positionTail=z,velocityTail=z;const coefficients:I[]=[];
 if(!eq(g,zero)){
  const complement=agm(sub(one,m)),exponent=times(pi,c.divide(M,complement)),r=c.rationalRange(exponent),ex=createTranscendentalContext(512),e=(v:F)=>ex.expNegativeRatio({coefficient:v.n,exponent:0},{coefficient:v.d,exponent:0});
  nome=c.range({lower:f(e(r.upper).lower,B(1)<<B(512)),upper:f(e(r.lower).upper,B(1)<<B(512))});
  if(nome.lower<=B(0)||nome.upper>q(f(B(1),B(16))).lower)throw Error('Positive nome bounded by1/16 required');
  prefactor=scaled(times(M,root(c.divide(nome,q(m)))),integer(4));
  let qn=o;for(let n=0;n<N;n++){coefficients.push(c.divide(qn,plus(o,times(times(qn,qn),nome))));qn=times(qn,nome);}
  const den=minus(o,nome);positionTail=c.divide(times(prefactor,qn),den);
  velocityTail=times(times(times(prefactor,omega),qn),plus(c.divide(q(integer(2*N+1)),den),c.divide(scaled(nome,integer(2)),times(den,den))));
 }
 const omega1=add(one,mul(f(B(3),B(8)),g)),omega2=sub(omega1,mul(f(B(21),B(256)),g2));
 const shapes=[{order:1,frequency:omega1,amplitudes:[sub(one,div(g,integer(32))),div(g,integer(32)),zero]}, {order:2,frequency:omega2,amplitudes:[add(sub(one,div(g,integer(32))),mul(f(B(23),B(1024)),g2)),sub(div(g,integer(32)),mul(f(B(24),B(1024)),g2)),div(g2,integer(1024))]}];
 const chebyshev:P[]=[[zero,one],[zero,integer(-3),zero,integer(4)],[zero,integer(5),zero,integer(-20),zero,integer(16)]];
 const residuals=shapes.map(method=>{const polynomial=method.amplitudes.reduce<P>((r,a,j)=>polyAdd(r,polyScale(chebyshev[j],a)),[zero]),second=method.amplitudes.reduce<P>((r,a,j)=>polyAdd(r,polyScale(chebyshev[j],mul(integer(-((2*j+1)**2)),a))),[zero]);const coefficients=polyAdd(polyAdd(polyScale(second,mul(method.frequency,method.frequency)),polynomial),polyScale(polyMul(polyMul(polynomial,polynomial),polynomial),g));return{order:method.order,frequency:text(method.frequency),positiveFrequency:method.frequency.n>B(0),waveformCoefficients:method.amplitudes.map(text),normalizedResidualPolynomialInCos:coefficients.map(text),normalizedResidualUniformBound:text(coefficients.reduce((r,a)=>add(r,abs(a)),zero)),physicalResidualUniformBound:text(mul(A,coefficients.reduce((r,a)=>add(r,abs(a)),zero))),boundScope:'Interior residual bound, not a trajectory-error bound.'};});
 const potentialEnergy=add(div(mul(A,A),integer(2)),div(mul(lambda,mul(mul(A,A),mul(A,A))),integer(4)));
 const energy=(position:I,velocity:I)=>plus(scaled(c.square(velocity),f(B(1),B(2))),plus(scaled(c.square(position),f(B(1),B(2))),scaled(c.square(c.square(position)),div(lambda,integer(4)))));
 const exact=(time:F)=>{
  if(eq(A,zero))return{position:z,velocity:z};if(eq(time,zero))return{position:q(A),velocity:z};
  const phase=times(omega,q(time)),trig=c.sinCos(c.rationalRange(phase));if(eq(g,zero))return{position:scaled(trig.cos,A),velocity:scaled(c.neg(trig.sin),A)};
  const cos2=minus(scaled(c.square(trig.cos),integer(2)),o),factor=scaled(cos2,integer(2));let cm=trig.cos,sm=trig.sin,cn=times(trig.cos,minus(factor,o)),sn=times(trig.sin,plus(factor,o)),sum=z,derivative=z;
  for(let n=0;n<N;n++){const co=n===0?cm:cn,si=n===0?sm:sn;sum=plus(sum,times(coefficients[n],co));derivative=plus(derivative,scaled(times(coefficients[n],si),integer(2*n+1)));if(n>0){const nextC=minus(times(factor,cn),cm),nextS=minus(times(factor,sn),sm);cm=cn;cn=nextC;sm=sn;sn=nextS;}}
  const widen=(v:I,r:I):I=>({lower:v.lower-r.upper,upper:v.upper+r.upper}),position=scaled(widen(times(prefactor,sum),positionTail),A),velocity=scaled(widen(c.neg(times(times(prefactor,omega),derivative)),velocityTail),A),limit=c.sqrt(mul(mul(A,A),add(one,div(g,integer(2))))).upper;
  return{position:c.intersect(position,-q(A).upper,q(A).upper),velocity:c.intersect(velocity,-limit,limit)};
 };
 const approximate=(time:F)=>{
  if(eq(A,zero))return[{position:z,velocity:z,residual:z},...shapes.map(()=>({position:z,velocity:z,residual:z}))];
  const t=q(time),base=c.sinCos({lower:time,upper:time}),third=c.sinCos({lower:mul(integer(3),time),upper:mul(integer(3),time)}),correction=minus(scaled(minus(third.cos,base.cos),f(B(1),B(32))),scaled(times(t,base.sin),f(B(3),B(8)))),naive=plus(base.cos,scaled(correction,g));
  const naiveVelocity=plus(c.neg(base.sin),scaled(minus(scaled(plus(scaled(third.sin,integer(-3)),base.sin),f(B(1),B(32))),scaled(plus(base.sin,times(t,base.cos)),f(B(3),B(8)))),g));
  const methods=[{position:scaled(naive,A),velocity:scaled(naiveVelocity,A),residual:scaled(scaled(minus(power(naive,3),power(base.cos,3)),g),A)}];
  for(const method of shapes){let y=z,dy=z,ddy=z;for(let j=0;j<3;j++){if(eq(method.amplitudes[j],zero))continue;const harmonic=integer(2*j+1),angle=mul(mul(method.frequency,time),harmonic),trig=c.sinCos({lower:angle,upper:angle});y=plus(y,scaled(trig.cos,method.amplitudes[j]));dy=plus(dy,scaled(trig.sin,neg(mul(method.amplitudes[j],mul(harmonic,method.frequency)))));ddy=plus(ddy,scaled(trig.cos,neg(mul(method.amplitudes[j],mul(mul(harmonic,harmonic),mul(method.frequency,method.frequency))))));}methods.push({position:scaled(y,A),velocity:scaled(dy,A),residual:scaled(plus(plus(ddy,y),scaled(power(y,3),g)),A)});}
  return methods;
 };
 const at=(time:F)=>{const reference=exact(time),methods=approximate(time);return{time:text(time),exact:reference,methods};};
 const sample=(time:F)=>{const r=at(time),initial=eq(time,zero),initialPosition={lower:text(A),upper:text(A)};return{time:r.time,exactPosition:initial?initialPosition:wire(r.exact.position),exactVelocity:wire(r.exact.velocity),naivePosition:initial?initialPosition:wire(r.methods[0].position),firstPosition:initial?initialPosition:wire(r.methods[1].position),secondPosition:initial?initialPosition:wire(r.methods[2].position)};};
 const p=at(probeTime),probe={time:p.time,exactPosition:eq(probeTime,zero)?{lower:text(A),upper:text(A)}:wire(p.exact.position),exactVelocity:wire(p.exact.velocity),methods:p.methods.map((r,j)=>({name:['naive','lindstedt-first','lindstedt-second'][j],position:eq(probeTime,zero)?{lower:text(A),upper:text(A)}:wire(r.position),velocity:wire(r.velocity),positionError:eq(probeTime,zero)||eq(g,zero)?wire(z):distance(r.position,p.exact.position),velocityError:eq(probeTime,zero)||eq(g,zero)?wire(z):distance(r.velocity,p.exact.velocity),residual:wire(r.residual),energy:eq(probeTime,zero)||eq(g,zero)?{lower:text(potentialEnergy),upper:text(potentialEnergy)}:wire(energy(r.position,r.velocity)),energyDefect:eq(probeTime,zero)||eq(g,zero)?wire(z):wire(minus(energy(r.position,r.velocity),q(potentialEnergy)))}))};
 const frequencyCeiling=ceil(omega.upper,c.from(one).upper),localWidth=min(H,div(integer(32),f(frequencyCeiling))),localStart=max(zero,min(sub(probeTime,div(localWidth,integer(2))),sub(H,localWidth)));
 const grid=Array.from({length:65},(_,j)=>sample(mul(H,f(B(j),B(64))))),local=Array.from({length:65},(_,j)=>sample(add(localStart,mul(localWidth,f(B(j),B(64))))));
 const phaseDrift=(frequency:F)=>wire(scaled(minus(q(frequency),omega),H));
 return{input,mode,effectiveCoupling:text(g),initialEnergy:text(potentialEnergy),frequency:mode==='equilibrium'?null:wire(omega),period:mode==='equilibrium'?null:wire(c.divide(scaled(pi,integer(2)),omega)),elliptic:mode==='nonlinear-periodic'?{parameter:text(m),agmIterations:18,fourierTerms:N,nome:wire(nome!),positionTailBound:wire(scaled(positionTail,A)),velocityTailBound:wire(scaled(velocityTail,A)),tailScope:'Enclosures of geometric tail upper-bound formulas, not lower bounds on actual truncation errors. Arithmetic intervals are propagated separately.'}:null,approximations:residuals,probe,fullSamples:grid,localSamples:local,localWindow:{start:text(localStart),width:text(localWidth)},horizonDiagnostics:{physicalSecularEnvelope:text(mul(A,mul(f(B(3),B(8)),mul(g,H)))),physicalNaiveCorrectionEnvelope:text(mul(A,mul(g,add(f(B(1),B(16)),mul(f(B(3),B(8)),H))))),normalizedSecularEnvelope:text(mul(f(B(3),B(8)),mul(g,H))),normalizedNaiveCorrectionEnvelope:text(mul(g,add(f(B(1),B(16)),mul(f(B(3),B(8)),H)))),phaseDrift:mode==='equilibrium'?null:{naive:phaseDrift(one),first:phaseDrift(omega1),second:phaseDrift(omega2)},fullGridResolvesEightPointsPerCycle:mode==='equilibrium'?null:times(omega,q(div(H,integer(64)))).upper<=scaled(pi,f(B(1),B(4))).lower,phaseMeaning:'Unwrapped exact fundamental phase versus formal approximation phases; nonpositive truncated frequencies are invalid physical frequency approximations. Not atan2(position,velocity) or position error.'},precision:{outputBits,workingTrigBits:c.workBits,rootBits:512,nomeBits:512},contract:'Finite undamped,unforced hardening oscillator at a turning point. Exact periodic reference is enclosed by AGM and nome Fourier series. Lindstedt residual bounds are not whole-trajectory guarantees; truncated frequencies can be nonpositive outside weak coupling. Zero amplitude is an equilibrium without a physical period.'};
}
const absBig=(n:bigint)=>n<BigInt(0)?-n:n,minBig=(a:bigint,b:bigint)=>a<b?a:b,maxBig=(a:bigint,b:bigint)=>a>b?a:b;
export type DuffingSecularResult=ReturnType<typeof calculateDuffingSecular>;
