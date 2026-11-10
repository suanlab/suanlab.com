/** Worker-side directed fixed-point arithmetic for real-time observations. */
import {f,type F} from './learning-quantum-hilbert-core';
import {createTranscendentalContext} from './learning-interval-transcendentals';
import {certifiedRealTimeRotation,realTimePiInterval} from './learning-real-time-phase-rotation';
export interface RealTimeInterval {lower:bigint;upper:bigint;}
export interface RealTimeRange {lower:string;upper:string;}
const Z=BigInt(0),O=BigInt(1),T=BigInt(2);
export const realTimeFloor=(n:bigint,d:bigint):bigint=>d<Z?realTimeFloor(-n,-d):n>=Z?n/d:-((-n+d-O)/d);
export const realTimeCeil=(n:bigint,d:bigint)=>-realTimeFloor(-n,d);
export function realTimeIntegerRoot(n:bigint,degree:2|3|4):bigint {
  if(n<Z)throw new RangeError('Nonnegative integer root required');if(n<T)return n;
  const k=BigInt(degree);let x=O<<BigInt(Math.ceil(n.toString(2).length/degree));
  for(;;){const y=((k-O)*x+n/(x**(k-O)))/k;if(y>=x)break;x=y;}
  while(x**k>n)x--;while((x+O)**k<=n)x++;return x;
}
export function createRealTimeIntervalContext(workBits=4096) {
  if(!Number.isInteger(workBits)||workBits<1024||workBits>8192)throw new RangeError('Real-time work precision1024..8192 required');
  const S=O<<BigInt(workBits),point=(x:bigint):RealTimeInterval=>({lower:x,upper:x});
  const from=(x:F):RealTimeInterval=>{if(x.d<=Z)throw new RangeError('Positive rational denominator required');return{lower:realTimeFloor(x.n*S,x.d),upper:realTimeCeil(x.n*S,x.d)};};
  const integer=(n:number|bigint)=>point(BigInt(n)*S),neg=(x:RealTimeInterval)=>({lower:-x.upper,upper:-x.lower});
  const add=(a:RealTimeInterval,b:RealTimeInterval)=>({lower:a.lower+b.lower,upper:a.upper+b.upper});
  const sub=(a:RealTimeInterval,b:RealTimeInterval)=>add(a,neg(b));
  const multiply=(a:RealTimeInterval,b:RealTimeInterval):RealTimeInterval=>{
    const p=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper];
    return{lower:realTimeFloor(p.reduce((a,b)=>a<b?a:b),S),upper:realTimeCeil(p.reduce((a,b)=>a>b?a:b),S)};
  };
  const divide=(a:RealTimeInterval,b:RealTimeInterval):RealTimeInterval=>{
    if(b.lower<=Z&&b.upper>=Z)throw new RangeError('Division interval contains zero');
    const ns=[a.lower,a.lower,a.upper,a.upper],ds=[b.lower,b.upper,b.lower,b.upper];
    return{lower:ns.map((x,i)=>realTimeFloor(x*S,ds[i])).reduce((a,b)=>a<b?a:b),upper:ns.map((x,i)=>realTimeCeil(x*S,ds[i])).reduce((a,b)=>a>b?a:b)};
  };
  const divideInteger=(a:RealTimeInterval,d:number|bigint):RealTimeInterval=>{const k=BigInt(d);if(k<=Z)throw new RangeError('Positive integer divisor required');return{lower:realTimeFloor(a.lower,k),upper:realTimeCeil(a.upper,k)};};
  const scale=(a:RealTimeInterval,n:number|bigint)=>{const k=BigInt(n);return k>=Z?{lower:a.lower*k,upper:a.upper*k}:{lower:a.upper*k,upper:a.lower*k};};
  const square=(a:RealTimeInterval):RealTimeInterval=>{
    const x=a.lower*a.lower,y=a.upper*a.upper;
    return{lower:a.lower<=Z&&a.upper>=Z?Z:realTimeFloor(x<y?x:y,S),upper:realTimeCeil(x>y?x:y,S)};
  };
  const root=(a:RealTimeInterval,degree:2|3|4):RealTimeInterval=>{
    if(a.lower<Z)throw new RangeError('Nonnegative root interval required');
    const factor=S**BigInt(degree-1),lo=realTimeIntegerRoot(a.lower*factor,degree),hi=realTimeIntegerRoot(a.upper*factor,degree);
    return{lower:lo,upper:hi**BigInt(degree)===a.upper*factor?hi:hi+O};
  };
  let transcendental:ReturnType<typeof createTranscendentalContext>|undefined;
  const trans=()=>transcendental??=createTranscendentalContext(workBits);
  const binary=(coefficient:bigint)=>({coefficient,exponent:0});
  const log=(a:RealTimeInterval):RealTimeInterval=>{
    if(a.lower<=Z)throw new RangeError('Positive logarithm interval required');
    return{lower:trans().logRatio(binary(a.lower),binary(S)).lower,upper:trans().logRatio(binary(a.upper),binary(S)).upper};
  };
  const expNegative=(a:RealTimeInterval):RealTimeInterval=>{
    if(a.lower<Z)throw new RangeError('Nonnegative decay exponent required');
    return{lower:trans().expNegativeRatio(binary(a.upper),binary(S)).lower,upper:trans().expNegativeRatio(binary(a.lower),binary(S)).upper};
  };
  const sinCos=(a:RealTimeInterval)=>{
    const midpoint=f(a.lower+a.upper,T*S),radius=realTimeCeil(a.upper-a.lower,T);
    const bits=1024,rotation=certifiedRealTimeRotation(midpoint,bits),lift=O<<BigInt(workBits-bits);
    const widen=(x:RealTimeInterval)=>({lower:x.lower*lift-radius < -S?-S:x.lower*lift-radius,upper:x.upper*lift+radius>S?S:x.upper*lift+radius});
    return{sin:widen(rotation.sin),cos:widen(rotation.cos)};
  };
  const observe=(a:RealTimeInterval,bits=512,pad=true):RealTimeRange=>{
    if(a.lower>a.upper||!Number.isInteger(bits)||bits<64||bits>workBits)throw new RangeError('Ordered interval and valid observation precision required');
    const divisor=O<<BigInt(workBits-bits),p=pad?O:Z,outS=O<<BigInt(bits);
    const text=(n:bigint)=>{const x=f(n,outS);return x.d===O?String(x.n):`${x.n}/${x.d}`;};
    if(a.lower===Z&&a.upper===Z)return{lower:'0',upper:'0'};
    return{lower:text(realTimeFloor(a.lower,divisor)-p),upper:text(realTimeCeil(a.upper,divisor)+p)};
  };
  return{workBits,S,zero:point(Z),one:point(S),point,from,integer,neg,add,sub,multiply,divide,divideInteger,scale,square,root,
    log,expNegative,sinCos,observe,pi:()=>realTimePiInterval(workBits)};
}
