/** Semantic interval comparisons shared by independent response checks. */
import {checkerCanonicalReader} from './learning-real-time-validation-rational';
import {realTimePhaseData} from './learning-real-time-phase-protocol';
import type {createRealTimeCheckerInterval,CheckerInterval as I} from './learning-real-time-checker-interval';
const Z=BigInt(0),O=BigInt(1);
export function checkerObservationRecord(value:unknown,keys:string[]) {
  const r=realTimePhaseData(value,keys);if(!r)throw Error('Observation fields');return r;
}
export function createRealTimeObservationCheck(ctx:ReturnType<typeof createRealTimeCheckerInterval>) {
  const read=checkerCanonicalReader(),outScale=O<<BigInt(512);
  function range(value:unknown,expected:I,scale=0) {
    const r=checkerObservationRecord(value,['lower','upper']),lo=read(r.lower,32768),hi=read(r.upper,32768);
    for(const q of [lo,hi])if(q.d>outScale||(q.d&(q.d-O))!==Z)throw Error('Observation dyadic precision');
    if(lo.n*hi.d>hi.n*lo.d||(hi.n*lo.d-lo.n*hi.d)*outScale>BigInt(8)*lo.d*hi.d)throw Error('Observation resolution');
    const shifted=(q:typeof lo)=>scale>=0?{n:q.n<<BigInt(scale),d:q.d}:{n:q.n,d:q.d<<BigInt(-scale)};
    const l=shifted(lo),h=shifted(hi);
    if(l.n*ctx.S>expected.lower*l.d||h.n*ctx.S<expected.upper*h.d)throw Error('Observation misses independent enclosure');
  }
  function exponent(value:unknown):number {
    if(typeof value!=='number'||!Number.isSafeInteger(value)||Object.is(value,-0)||Math.abs(value)>32768)throw Error('Observation exponent');return value;
  }
  function normalization(values:I[],e:number) {
    const max=values.flatMap(x=>[x.lower<Z?-x.lower:x.lower,x.upper<Z?-x.upper:x.upper]).reduce((a,b)=>a>b?a:b);
    if(max===Z){if(e!==0)throw Error('Zero observation scale');return;}
    const n=e<0?max<<BigInt(-e):max,d=e<0?ctx.S:ctx.S<<BigInt(e);
    if(n*BigInt(4)<=d||n>=TWO*d)throw Error('Observation shared normalization');
  }
  const TWO=BigInt(2);
  function complex(value:unknown,expected:{real:I;imaginary:I},scale=0) {
    const r=checkerObservationRecord(value,['real','imaginary']);range(r.real,expected.real,scale);range(r.imaginary,expected.imaginary,scale);
  }
  function scaledPositive(value:unknown,expected:I) {
    const r=checkerObservationRecord(value,['scalePowerOfTwo','lower','upper']),e=exponent(r.scalePowerOfTwo);
    normalization([expected],e);range({lower:r.lower,upper:r.upper},expected,e);
  }
  function scaledComplex(value:unknown,expected:{real:I;imaginary:I}) {
    const r=checkerObservationRecord(value,['scalePowerOfTwo','real','imaginary']),e=exponent(r.scalePowerOfTwo);
    normalization([expected.real,expected.imaginary],e);range(r.real,expected.real,e);range(r.imaginary,expected.imaginary,e);
  }
  return{range,complex,scaledPositive,scaledComplex};
}
