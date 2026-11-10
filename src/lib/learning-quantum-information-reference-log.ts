/** Independent log2 enclosure: signed log(1+u) Taylor series, |u|<=1/2.
 * This never calls the producer's atanh/logarithm context. Worker/CLI arithmetic. */
import {f,zero,one,add,sub,mul,div,eq,le,type F,type R}from'./learning-quantum-hilbert-core';
const O=BigInt(1),Z=BigInt(0),workBits=608,terms=608,scale=O<<BigInt(workBits),outScale=O<<BigInt(512);
type I={lower:bigint;upper:bigint};
const floor=(n:bigint,d:bigint)=>n>=Z?n/d:-((-n+d-O)/d),ceil=(n:bigint,d:bigint)=>-floor(-n,d);
const sum=(a:I,b:I):I=>({lower:a.lower+b.lower,upper:a.upper+b.upper});
const difference=(a:I,b:I):I=>({lower:a.lower-b.upper,upper:a.upper-b.lower});
function product(a:I,b:I):I{
 const values=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper];
 return {lower:floor(values.reduce((x,y)=>x<y?x:y),scale),upper:ceil(values.reduce((x,y)=>x>y?x:y),scale)};
}
function lnOnePlus(u:F):I{
 if(!le(f(BigInt(-1),BigInt(4)),u)||!le(u,f(O,BigInt(2))))throw new RangeError('Reference log range reduction failed');
 if(eq(u,zero))return{lower:Z,upper:Z};
 const x={lower:floor(u.n*scale,u.d),upper:ceil(u.n*scale,u.d)};
 let power=x,total:I={lower:Z,upper:Z};
 for(let j=1;j<=terms;j++){
  const term={lower:floor(power.lower,BigInt(j)),upper:ceil(power.upper,BigInt(j))};
  total=j%2?sum(total,term):difference(total,term);power=product(power,x);
 }
 // Sum_{j>N}|u|^j/j <= 2^-N/(N+1), including the negative-u case.
 const tail=ceil(scale,(O<<BigInt(terms))*BigInt(terms+1));
 return{lower:total.lower-tail,upper:total.upper+tail};
}
let lnTwo:I|undefined;
export function independentInformationLog2(x:F):R{
 if(!x||typeof x.n!=='bigint'||typeof x.d!=='bigint'||x.n<=Z||x.d<=Z)throw new RangeError('Positive rational logarithm argument required');
 let k=x.n.toString(2).length-x.d.toString(2).length;
 if(k>=0?x.n<(x.d<<BigInt(k)):(x.n<<BigInt(-k))<x.d)k--;
 let y=k>=0?f(x.n,x.d<<BigInt(k)):f(x.n<<BigInt(-k),x.d);
 if(eq(y,one))return{lower:f(BigInt(k)),upper:f(BigInt(k))};
 if(!le(mul(f(BigInt(2)),y),f(BigInt(3)))){y=div(y,f(BigInt(2)));k++;}
 const numerator=lnOnePlus(sub(y,one));
 lnTwo??=difference(lnOnePlus(f(O,BigInt(2))),lnOnePlus(f(BigInt(-1),BigInt(4))));
 if(lnTwo.lower<=Z)throw new RangeError('Reference ln2 denominator unresolved');
 const quotients=[f(numerator.lower,lnTwo.lower),f(numerator.lower,lnTwo.upper),f(numerator.upper,lnTwo.lower),f(numerator.upper,lnTwo.upper)];
 const lower=add(f(BigInt(k)),quotients.reduce((a,b)=>le(a,b)?a:b)),upper=add(f(BigInt(k)),quotients.reduce((a,b)=>le(a,b)?b:a));
 return{lower:f(floor(lower.n*outScale,lower.d),outScale),upper:f(ceil(upper.n*outScale,upper.d),outScale)};
}
export function independentInformationEntropy(probabilities:readonly F[]):R{
 if(!eq(probabilities.reduce(add,zero),one)||probabilities.some(x=>!le(zero,x)))throw new RangeError('Probability spectrum required');
 return probabilities.filter(x=>x.n>Z).reduce((total,p)=>{
  const logarithm=independentInformationLog2(p);
  return{lower:sub(total.lower,mul(p,logarithm.upper)),upper:sub(total.upper,mul(p,logarithm.lower))};
 },{lower:zero,upper:zero});
}
