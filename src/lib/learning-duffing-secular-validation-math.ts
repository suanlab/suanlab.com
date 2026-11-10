/** Independent rational/fixed-point verification arithmetic. No producer core,
 * trig, pi, sqrt or positive-exponential implementation is imported.
 */
import {Rational as Q} from './learning-graph-gauge-validation-math';
const B=BigInt,Z=B(0),O=B(1),bits=768,S=O<<B(bits),rootScale=O<<B(512);
export type F=Q;export type P=Q[];export type I={lower:bigint;upper:bigint};
export const f=(n:bigint,d=O)=>new Q(n,d),zero=f(Z),one=f(O),add=(a:Q,b:Q)=>a.add(b),sub=(a:Q,b:Q)=>a.sub(b),neg=(a:Q)=>a.neg(),mul=(a:Q,b:Q)=>a.mul(b),div=(a:Q,b:Q)=>a.div(b),eq=(a:Q,b:Q)=>a.n*b.d===b.n*a.d,le=(a:Q,b:Q)=>a.n*b.d<=b.n*a.d;
export const polyAdd=(a:P,b:P):P=>Array.from({length:Math.max(a.length,b.length)},(_,j)=>add(a[j]??zero,b[j]??zero)),polyScale=(a:P,b:Q)=>a.map(v=>mul(v,b)),polyMul=(a:P,b:P):P=>Array.from({length:a.length+b.length-1},(_,j)=>a.reduce((sum,v,k)=>j>=k&&j-k<b.length?add(sum,mul(v,b[j-k])):sum,zero));
const floor=(n:bigint,d:bigint):bigint=>d<Z?floor(-n,-d):n>=Z?n/d:-((-n+d-O)/d),ceil=(n:bigint,d:bigint)=>-floor(-n,d),min=(xs:bigint[])=>xs.reduce((a,b)=>a<b?a:b),max=(xs:bigint[])=>xs.reduce((a,b)=>a>b?a:b);
/** Restoring binary square root, independent of the producer Newton iteration. */
function isqrt(n:bigint){if(n<Z)throw Error('Nonnegative square required');let remainder=n,result=Z,bit=O<<B(Math.max(0,2*Math.floor((n.toString(2).length-1)/2)));while(bit){const trial=result+bit;if(remainder>=trial){remainder-=trial;result=(result>>O)+bit;}else result>>=O;bit>>=B(2);}return result;}
export function sqrtRange(x:Q){if(x.n<Z)throw Error('Nonnegative root required');const n=isqrt(x.n),d=isqrt(x.d);if(n*n===x.n&&d*d===x.d)return{lower:f(n,d),upper:f(n,d)};const scaled=x.n*rootScale*rootScale,r=isqrt(scaled/x.d);return{lower:f(r,rootScale),upper:f(r+(r*r*x.d===scaled?Z:O),rootScale)};}
/** Størmer identity π=48 atan(1/18)+32 atan(1/57)-20 atan(1/239).
 * Alternating rational tails, 96 guard bits; different from producer Machin.
 */
export function independentDuffingPi(precision:number){
 const work=O<<B(precision+96),atan=(k:bigint)=>{let lo=Z,hi=Z,p=k,j=Z;for(;;){const den=(B(2)*j+O)*p;if(den>work*B(256)){const tail=ceil(work,den);if(j%B(2))lo-=tail;else hi+=tail;return[lo,hi]as const;}const a=work/den,b=ceil(work,den);if(j%B(2)){lo-=b;hi-=a;}else{lo+=a;hi+=b;}p*=k*k;j++;}};
 const a=atan(B(18)),b=atan(B(57)),c=atan(B(239)),guard=O<<B(96);return{lower:floor(B(48)*a[0]+B(32)*b[0]-B(20)*c[1],guard),upper:ceil(B(48)*a[1]+B(32)*b[1]-B(20)*c[0],guard)};
}
/** Negative alternating exponential, not an inverted positive series. */
export function independentDuffingExponent(precision:number){const target=O<<B(precision);return{expNegativeRatio(n:{coefficient:bigint;exponent:number},d:{coefficient:bigint;exponent:number}){
 if(n.exponent!==0||d.exponent!==0||n.coefficient<Z||d.coefficient<=Z)throw Error('Rational decay required');if(n.coefficient===Z)return{lower:target,upper:target};let squarings=0;while(B(8)*n.coefficient>(d.coefficient<<B(squarings)))squarings++;
 for(let extra=96;extra<=1024;extra+=64){const scale=O<<B(precision+extra),den=d.coefficient<<B(squarings),x={lower:n.coefficient*scale/den,upper:ceil(n.coefficient*scale,den)},multiply=(a:I,b:I):I=>({lower:a.lower*b.lower/scale,upper:ceil(a.upper*b.upper,scale)});let term:I={lower:scale,upper:scale},lo=scale,hi=scale,upper=scale;const even=2*Math.ceil((precision+extra+12)/6);
  for(let j=1;j<=even+1;j++){const p=multiply(term,x);term={lower:p.lower/B(j),upper:ceil(p.upper,B(j))};if(j%2){lo-=term.upper;hi-=term.lower;}else{lo+=term.lower;hi+=term.upper;}if(j===even)upper=hi;}
  let r:I={lower:lo,upper};for(let j=0;j<squarings;j++)r=multiply(r,r);const lower=r.lower>>B(extra),higher=r.upper>>B(extra);if(lower===higher)return{lower,upper:lower+O};
 }throw Error('Independent exponential cell unresolved');
}};}
export function createIndependentDuffingContext(){
 const point=(x:bigint):I=>({lower:x,upper:x}),from=(x:Q):I=>({lower:floor(x.n*S,x.d),upper:ceil(x.n*S,x.d)}),negative=(a:I):I=>({lower:-a.upper,upper:-a.lower}),plus=(a:I,b:I):I=>({lower:a.lower+b.lower,upper:a.upper+b.upper}),minus=(a:I,b:I)=>plus(a,negative(b));
 const times=(a:I,b:I):I=>{const v=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper];return{lower:floor(min(v),S),upper:ceil(max(v),S)};},divide=(a:I,b:I):I=>{if(b.lower<=Z&&b.upper>=Z)throw Error('Zero interval denominator');const ns=[a.lower,a.lower,a.upper,a.upper],ds=[b.lower,b.upper,b.lower,b.upper];return{lower:min(ns.map((n,j)=>floor(n*S,ds[j]))),upper:max(ns.map((n,j)=>ceil(n*S,ds[j])))};};
 const square=(a:I):I=>({lower:a.lower<=Z&&a.upper>=Z?Z:floor(min([a.lower*a.lower,a.upper*a.upper]),S),upper:ceil(max([a.lower*a.lower,a.upper*a.upper]),S)}),intersect=(a:I,l:bigint,u:bigint):I=>{const out={lower:a.lower<l?l:a.lower,upper:a.upper>u?u:a.upper};if(out.lower>out.upper)throw Error('Disjoint known range');return out;};
 const range=(r:{lower:Q;upper:Q}):I=>({lower:from(r.lower).lower,upper:from(r.upper).upper}),rationalRange=(a:I)=>({lower:f(a.lower,S),upper:f(a.upper,S)}),cache=new Map<string,{sin:I;cos:I}>();
 function sinCosPoint(x:Q){const key=x.text(),old=cache.get(key);if(old)return old;if(x.n===Z)return{sin:point(Z),cos:point(S)};if((x.n<Z?-x.n:x.n)>(O<<B(64))*x.d)throw Error('Bounded angle required');
  const workBits=bits+128,scale=O<<B(workBits);let squarings=0,den=x.d;const n=x.n<Z?-x.n:x.n;while(B(8)*n>den){den*=B(2);squarings++;}const a:I={lower:n*scale/den,upper:ceil(n*scale,den)},mult=(a:I,b:I):I=>{const p=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper];return{lower:floor(min(p),scale),upper:ceil(max(p),scale)};},a2=mult(a,a),intDiv=(a:I,d:bigint):I=>({lower:floor(a.lower,d),upper:ceil(a.upper,d)});
  let st=a,ct=point(scale),sin=a,cos=point(scale);const terms=2*Math.ceil((workBits+24)/12);
  for(let j=1;j<=terms+1;j++){st=intDiv(mult(st,a2),B(2*j)*B(2*j+1));ct=intDiv(mult(ct,a2),B(2*j-1)*B(2*j));if(j%2){sin=minus(sin,st);cos=minus(cos,ct);}else{sin=plus(sin,st);cos=plus(cos,ct);}}
  const tail=ceil(scale,O<<B(6*terms+3));sin={lower:sin.lower-tail,upper:sin.upper+tail};cos={lower:cos.lower-tail,upper:cos.upper+tail};
  for(let j=0;j<squarings;j++){const next=mult(sin,cos),cc=mult(cos,cos),ss=mult(sin,sin);sin={lower:B(2)*next.lower,upper:B(2)*next.upper};cos=minus(cc,ss);sin={lower:sin.lower< -scale?-scale:sin.lower,upper:sin.upper>scale?scale:sin.upper};cos={lower:cos.lower< -scale?-scale:cos.lower,upper:cos.upper>scale?scale:cos.upper};}
  const convert=(v:I):I=>({lower:floor(v.lower,O<<B(128)),upper:ceil(v.upper,O<<B(128))}),result={sin:x.n<Z?negative(convert(sin)):convert(sin),cos:convert(cos)};cache.set(key,result);return result;
 }
 const sinCos=(r:{lower:Q;upper:Q})=>{if(eq(r.lower,r.upper))return sinCosPoint(r.lower);const center=div(add(r.lower,r.upper),f(B(2))),radius=from(div(sub(r.upper,r.lower),f(B(2)))).upper,v=sinCosPoint(center),widen=(x:I)=>intersect({lower:x.lower-radius,upper:x.upper+radius},-S,S);return{sin:widen(v.sin),cos:widen(v.cos)};};
 return{workBits:bits,from,range,rationalRange,zero:point(Z),one:point(S),neg:negative,add:plus,sub:minus,multiply:times,divide,square,intersect,sinCos,sqrt:(x:Q)=>range(sqrtRange(x))};
}
