/** Independent finite certificate checker,not a full response validator.
 * Physical kick/drift iteration uses homogeneous integer matrices. Prefix
 * certificates are checked on an integer lattice and their complete Green
 * matrices satisfy direct tridiagonal residual identities. Inertia comes from
 * 1x1/2x2 congruence,not the producer's Sturm sign sequence.
 */
import {readRealTimePhaseInput,type RealTimeChainInput,type RealTimePacketInput} from './learning-real-time-phase-input';
import {realTimePhaseData,realTimePhaseInputKey} from './learning-real-time-phase-protocol';
import {snapshotQubitWire} from './learning-qubit-wire';
import {CheckerRational as Q,checkerZero as zero,checkerOne as one,checkerInteger as integer,checkerLcm,checkerCanonicalReader,checkerExactDivision as exactDivide,
  gi,ga,gn,gm,gs,ge,type CheckerGaussianInteger as GI} from './learning-real-time-validation-rational';
const Z=BigInt(0),O=BigInt(1),two=integer(2);
type Fraction={n:bigint;d:bigint};
interface HomogeneousMap {numerators:bigint[][];denominator:bigint;}
const square=(x:Q)=>x.mul(x),total=(xs:Q[])=>xs.reduce((a,b)=>a.add(b),zero);
function record(value:unknown,keys:readonly string[]):Record<string,unknown>{const r=realTimePhaseData(value,keys);if(!r)throw Error('Wrong finite certificate record');return r;}
function array(value:unknown,length:number):readonly unknown[]{if(!Array.isArray(value)||value.length!==length)throw Error('Wrong finite certificate dimension');return value;}
function literal(actual:unknown,expected:unknown){if(!Object.is(actual,expected))throw Error('Wrong finite certificate metadata');}
/** A zero first pivot is eliminated with its neighboring2x2 block. Its
 * determinant is negative and inverse bottom-right entry zero,so the next
 * diagonal is unchanged. This is an exact congruence for a Jacobi matrix.
 */
function inertia(d:number,diagonal:Q) {
  let current=diagonal,positive=0,negative=0,zeros=0,k=0;
  while(k<d){if(current.n!==Z){if(current.n>Z)positive++;else negative++;k++;if(k<d)current=diagonal.sub(one.div(current));}
    else if(k+1<d){positive++;negative++;k+=2;current=diagonal;}
    else{zeros++;k++;}}
  return{positive,negative,zero:zeros};
}
function homogeneousStep(delta:Q,mass:Q,omega:Q):HomogeneousMap {
  const kick=mass.mul(omega).mul(delta).div(two).neg(),drift=delta.div(mass),diagonal=one.add(drift.mul(kick));
  const entries=[[diagonal,drift],[kick.mul(two.add(drift.mul(kick))),diagonal]];
  const denominator=entries.flat().reduce((d,q)=>checkerLcm(d,q.d),O);
  return{numerators:entries.map(row=>row.map(q=>q.n*(denominator/q.d))),denominator};
}
function canonicalFlow(n:number,time:Q,mass:Q,omega:Q):HomogeneousMap {
  const step=homogeneousStep(time.div(integer(n)),mass,omega),a=step.numerators;
  let current=[[O,Z],[Z,O]],denominator=O;
  for(let k=0;k<n;k++) {
    current=[[a[0][0]*current[0][0]+a[0][1]*current[1][0],a[0][0]*current[0][1]+a[0][1]*current[1][1]],
      [a[1][0]*current[0][0]+a[1][1]*current[1][0],a[1][0]*current[0][1]+a[1][1]*current[1][1]]];
    denominator*=step.denominator;
  }
  if(current[0][0]*current[1][1]-current[0][1]*current[1][0]!==denominator*denominator)throw Error('Independent canonical flow lost determinant one');
  return{numerators:current,denominator};
}
function realCertificate(theta:Q[],diagonal:Q) {
  const d=theta.length-1,L=diagonal.d,p=diagonal.n,powers=Array.from({length:d+1},(_,k)=>L**BigInt(k));
  const P=theta.map((q,k)=>exactDivide(q.n*powers[k],q.d));
  if(P[0]!==O||P[1]!==p)throw Error('Wrong real initial minors');
  for(let k=2;k<=d;k++)if(P[k]!==p*P[k-1]-L*L*P[k-2])throw Error('Wrong real minor certificate');
  const green=(i:number,j:number)=>P[Math.min(i,j)]*P[d-1-Math.max(i,j)]*powers[Math.abs(i-j)];
  if(P[d]!==Z)for(let i=0;i<d;i++)for(let j=0;j<d;j++) {
    const residual=p*green(i,j)-(i?L*green(i-1,j):Z)-(i+1<d?L*green(i+1,j):Z);
    if(residual!==(i===j?P[d]:Z))throw Error('Full real Green residual failed');
  }
  return{L,P,powers,green,determinant:P[d]};
}
export interface VerifiedRealTimeFiniteCore {
  scope:'finite-exact-core-only';fullResponseValidated:false;input:Readonly<RealTimeChainInput|RealTimePacketInput>;
  wire:Readonly<Record<string,unknown>>;inertia:{positive:number;negative:number;zero:number};map:Q[][];refinements:Q[][][];
  inverseEntry:((i:number,j:number)=>Fraction)|null;
  regulatedInverseEntry:((i:number,j:number)=>{real:Fraction;imaginary:Fraction})|null;
}
/** This deliberately accepts only an exact subrecord. Wave/Airy/range fields
 * remain unverified and cannot obtain a validated whole-response receipt.
 */
export function verifyRealTimeExactCore(source:unknown,value:unknown):VerifiedRealTimeFiniteCore|null {
  try {
    const input=readRealTimePhaseInput(source);if(!input||input.mode==='fold')return null;
    const owned=snapshotQubitWire(value),read=checkerCanonicalReader();
    const expect=(actual:unknown,expected:Fraction,maxBits=65536):Q=>{const q=read(actual,maxBits);if(!q.equals(expected))throw Error('Wrong exact certificate scalar');return q;};
    const vector=(actual:unknown,expected:Fraction[],bits=65536)=>array(actual,expected.length).map((q,i)=>expect(q,expected[i],bits));
    const checkMap=(value:unknown,expected:HomogeneousMap,n:number)=>array(value,2).map((row,i)=>array(row,2).map((q,j)=>expect(q,{n:expected.numerators[i][j],d:expected.denominator},148*n+130)));
    const n=input.intervals,d=n-1,time=Q.input(input.time),mass=Q.input(input.mass),h=Q.input(input.hbar),omega=Q.input(input.omegaSquared),delta=time.div(integer(n)),s=mass.div(delta),v=omega.mul(square(delta)),diagonal=two.sub(v),signed=inertia(d,diagonal);
    const keys=input.mode==='chain'?['kind','input','dimension','hessian','determinant','forcing','constant','inertia','stationaryStatus','stationary','stationaryAction','nullVector','compatible','causticIndex','inverse','map','refinements','regulator']:
      ['kind','input','variance','initialCovariance','finalCovariance','center','centralAction','initialWidth','finalWidth','negativeModes','causticIndex','map','refinements'];
    const wire=record(owned,keys),attributed=readRealTimePhaseInput(wire.input);
    if(!attributed||realTimePhaseInputKey(attributed)!==realTimePhaseInputKey(input))throw Error('Finite certificate attribution mismatch');
    literal(wire.kind,input.mode==='chain'?'finite-chain-core':'finite-packet-core');
    const map=checkMap(wire.map,canonicalFlow(n,time,mass,omega),n),refinements:Q[][][]=[];
    array(wire.refinements,4).forEach((row,i)=>{
      const data=record(row,['multiplier','intervals','map']),multiplier=[1,2,4,8][i];literal(data.multiplier,multiplier);literal(data.intervals,n*multiplier);
      refinements.push(checkMap(data.map,canonicalFlow(n*multiplier,time,mass,omega),n*multiplier));
    });
    const caustic=signed.zero?signed.negative+1:null;literal(wire.causticIndex,caustic);
    let inverseEntry:VerifiedRealTimeFiniteCore['inverseEntry']=null,regulatedInverseEntry:VerifiedRealTimeFiniteCore['regulatedInverseEntry']=null;
    if(input.mode==='packet') {
      const width=Q.input(input.width),variance=input.widthMode==='fixed-variance'?width:h.mul(width),chirp=Q.input(input.chirp),q0=Q.input(input.position),p0=Q.input(input.momentum);
      expect(wire.variance,variance,128);literal(wire.negativeModes,signed.negative);
      const initial=[[variance,chirp.mul(variance)],[chirp.mul(variance),square(chirp).mul(variance).add(square(h).div(integer(4).mul(variance)))]];
      array(wire.initialCovariance,2).forEach((row,i)=>vector(row,initial[i],512));
      const flow=canonicalFlow(n,time,mass,omega),M=flow.numerators,DS=initial.flat().reduce((den,q)=>checkerLcm(den,q.d),O),B=initial.map(row=>row.map(q=>q.n*(DS/q.d)));
      const expectedCov=Array.from({length:2},(_,i)=>Array.from({length:2},(_,j)=>({n:[0,1].reduce((out,k)=>out+[0,1].reduce((sum,l)=>sum+M[i][k]*B[k][l]*M[j][l],Z),Z),d:flow.denominator*flow.denominator*DS})));
      const covariance=array(wire.finalCovariance,2).map((row,i)=>vector(row,expectedCov[i],10752));
      if(covariance[0][0].n<=Z||covariance[1][1].n<=Z)throw Error('Packet covariance lost positivity');
      const centerExpected=map.map(row=>row[0].mul(q0).add(row[1].mul(p0))),center=vector(wire.center,centerExpected,5504);
      expect(wire.centralAction,center[0].mul(center[1]).sub(q0.mul(p0)).div(two),11264);
      const before=record(wire.initialWidth,['real','imaginary']),after=record(wire.finalWidth,['real','imaginary']);
      expect(before.real,chirp,64);expect(before.imaginary,h.div(two.mul(variance)),256);
      expect(after.real,covariance[0][1].div(covariance[0][0]),10752);const positive=expect(after.imaginary,h.div(two.mul(covariance[0][0])),10752);
      if(positive.n<=Z)throw Error('Packet complex width lost upper-half-plane property');
    } else {
      literal(wire.dimension,d);const hessian=record(wire.hessian,['diagonal','offDiagonal']);expect(hessian.diagonal,s.mul(diagonal),512);expect(hessian.offDiagonal,s.neg(),256);
      const actualInertia=record(wire.inertia,['positive','negative','zero']);for(const key of ['positive','negative','zero']as const)literal(actualInertia[key],signed[key]);
      const a=Q.input(input.start),b=Q.input(input.end),forcing=Array.from({length:d},()=>zero);forcing[0]=s.mul(a);forcing[d-1]=forcing[d-1].add(s.mul(b));vector(wire.forcing,forcing,256);
      const ends=square(a).add(square(b)),constant=s.mul(ends).div(two).sub(delta.mul(mass).mul(omega).mul(ends).div(integer(4)));expect(wire.constant,constant,512);
      const inverse=wire.inverse===null?null:record(wire.inverse,['layout','dimension','continuants','scale']);
      let theta:Q[];
      if(signed.zero){literal(wire.inverse,null);theta=array(wire.nullVector,d).map((q,i)=>read(q,Math.max(1,141*i)));theta.push(zero);}
      else{if(!inverse)throw Error('Missing real inverse certificate');literal(inverse.layout,'symmetric-continuant');literal(inverse.dimension,d);expect(inverse.scale,s,256);theta=array(inverse.continuants,d+1).map((q,i)=>read(q,Math.max(1,141*i)));literal(wire.nullVector,null);}
      const real=realCertificate(theta,diagonal),det=real.determinant;
      if((det===Z)!==Boolean(signed.zero))throw Error('Inertia/determinant certificate mismatch');
      expect(wire.determinant,{n:s.n**BigInt(d)*det,d:s.d**BigInt(d)*real.L**BigInt(d)},8192);
      if(det!==Z)inverseEntry=(i,j)=>{const numerator=real.L*s.d*real.green(i,j),denominator=s.n*det;return denominator<Z?{n:-numerator,d:-denominator}:{n:numerator,d:denominator};};
      const compatible=!signed.zero||b.equals(map[0][0].mul(a));literal(wire.compatible,compatible);
      if(signed.zero) {
        const parity=caustic!%2?-1:1;
        if(!map[0][0].equals(integer(parity))||!map[1][1].equals(integer(parity))||map[0][1].n!==Z||map[1][0].n!==Z)throw Error('Caustic canonical map is not the certified parity');
        const r=theta.slice(0,d);for(let i=0;i<d;i++)if(!diagonal.mul(r[i]).sub(i?r[i-1]:zero).sub(i+1<d?r[i+1]:zero).equals(zero))throw Error('Null vector residual failed');
      }
      literal(wire.stationaryStatus,signed.zero?(compatible?'affine-family':'no-stationary-path'):'unique');
      if(!compatible){literal(wire.stationary,null);literal(wire.stationaryAction,null);}
      else {
        const point=array(wire.stationary,d).map(q=>read(q,4608));
        if(signed.zero&&!point[0].equals(zero))throw Error('Affine representative must fix its first free coordinate');
        for(let i=0;i<d;i++)if(!s.mul(diagonal.mul(point[i]).sub(i?point[i-1]:zero).sub(i+1<d?point[i+1]:zero)).equals(forcing[i]))throw Error('Stationary witness residual failed');
        const action=constant.sub(total(forcing.map((q,i)=>q.mul(point[i]))).div(two));expect(wire.stationaryAction,action,5120);
        if(signed.zero&&action.n!==Z)throw Error('Caustic representative has nonzero action');
        if(!signed.zero){const generating=map[1][1].mul(square(b)).add(map[0][0].mul(square(a))).sub(two.mul(a).mul(b)).div(two.mul(map[0][1]));if(!action.equals(generating))throw Error('Stationary action disagrees with independent canonical flow');}
      }
      const epsilon=Q.input(input.damping);
      if(epsilon.n===Z)literal(wire.regulator,null);
      else {
        const regulator=record(wire.regulator,['layout','dimension','continuants','distancePowers','determinant','pivotLayout','pivotRootBranch','mean','exponent']);
        literal(regulator.layout,'symmetric-complex-continuant');literal(regulator.dimension,d);literal(regulator.pivotLayout,'successive-continuant-ratios');literal(regulator.pivotRootBranch,'positive-real-part-per-ratio');
        const Hdiag=s.mul(diagonal),L=checkerLcm(checkerLcm(epsilon.d,Hdiag.d),s.d),diag=gi(epsilon.n*(L/epsilon.d),-Hdiag.n*(L/Hdiag.d)),off=gi(Z,s.n*(L/s.d));
        const powers=Array.from({length:d+1},(_,k)=>L**BigInt(k));
        const complexValue=(value:unknown,bits:number)=>{const q=record(value,['real','imaginary']);return{real:read(q.real,bits),imaginary:read(q.imaginary,bits)};};
        const onLattice=(value:unknown,k:number):GI=>{const q=complexValue(value,Math.max(1,257*k));return gi(exactDivide(q.real.n*powers[k],q.real.d),exactDivide(q.imaginary.n*powers[k],q.imaginary.d));};
        const P=array(regulator.continuants,d+1).map(onLattice);
        if(!ge(P[0],gi(O))||!ge(P[1],diag))throw Error('Wrong complex initial minors');
        for(let k=2;k<=d;k++)if(!ge(P[k],ga(gm(diag,P[k-1]),gn(gm(gm(off,off),P[k-2])))))throw Error('Complex minor certificate failed');
        for(let k=1;k<=d;k++){const quotientNumerator=gm(P[k],gi(P[k-1].real,-P[k-1].imaginary));if(quotientNumerator.real<=Z)throw Error('Gaussian pivot root branch lacks positive real part');}
        const distances=[gi(O)];for(let k=1;k<d;k++)distances.push(gm(distances[k-1],gn(off)));
        array(regulator.distancePowers,d).forEach((q,k)=>{const v=complexValue(q,Math.max(1,85*k));if(v.real.n*powers[k]!==distances[k].real*v.real.d||v.imaginary.n*powers[k]!==distances[k].imaginary*v.imaginary.d)throw Error('Complex cofactor distance certificate failed');});
        const determinant=complexValue(regulator.determinant,257*d);
        if(determinant.real.n*powers[d]!==P[d].real*determinant.real.d||determinant.imaginary.n*powers[d]!==P[d].imaginary*determinant.imaginary.d)throw Error('Wrong complex determinant');
        const G=Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>gm(gm(P[Math.min(i,j)],P[d-1-Math.max(i,j)]),distances[Math.abs(i-j)])));
        for(let i=0;i<d;i++)for(let j=0;j<d;j++) {
          let residual=gm(diag,G[i][j]);if(i)residual=ga(residual,gm(off,G[i-1][j]));if(i+1<d)residual=ga(residual,gm(off,G[i+1][j]));
          if(!ge(residual,i===j?P[d]:gi(Z)))throw Error('Full complex Green residual failed');
        }
        const norm=P[d].real*P[d].real+P[d].imaginary*P[d].imaginary;if(norm===Z)throw Error('Positive regulator determinant vanished');
        regulatedInverseEntry=(i,j)=>{const z=gm(G[i][j],gi(P[d].real,-P[d].imaginary));return{real:{n:L*z.real,d:norm},imaginary:{n:L*z.imaginary,d:norm}};};
        const FD=forcing.reduce((den,q)=>checkerLcm(den,q.d),O),F=forcing.map(q=>q.n*(FD/q.d));
        const numerator=Array.from({length:d},(_,i)=>F.reduce((out,x,j)=>x===Z?out:ga(out,gs(G[i][j],x)),gi(Z)));
        const expectedMean=numerator.map(q=>gs(gm(gi(Z,-O),gm(q,gi(P[d].real,-P[d].imaginary))),L));
        array(regulator.mean,d).forEach((q,i)=>{const x=complexValue(q,16128);if(!x.real.equals({n:expectedMean[i].real,d:norm*FD})||!x.imaginary.equals({n:expectedMean[i].imaginary,d:norm*FD}))throw Error('Complex mean certificate failed');});
        const quadratic=F.reduce((out,x,i)=>x===Z?out:ga(out,gs(numerator[i],x)),gi(Z)),quotient=gs(gm(quadratic,gi(P[d].real,-P[d].imaginary)),L),denominator=norm*FD*FD;
        const exponent=complexValue(regulator.exponent,16896),real=Q.of(-quotient.real,denominator).div(two.mul(h)),imaginary=constant.div(h).sub(Q.of(quotient.imaginary,denominator).div(two.mul(h)));
        if(!exponent.real.equals(real)||!exponent.imaginary.equals(imaginary)||real.n>Z)throw Error('Regulated completed-square exponent failed');
      }
    }
    return Object.freeze({scope:'finite-exact-core-only',fullResponseValidated:false,input,wire:owned as Readonly<Record<string,unknown>>,inertia:signed,map,refinements,inverseEntry,regulatedInverseEntry});
  }catch{return null;}
}
