/** Small rational polynomial root isolation. Coefficients are ascending.
 * Sturm counts certify completeness; exact gcd chains certify multiplicity.
 * No distance tolerance merges nearby roots. Public users serialize BigInts.
 */
import { rationalGreatestCommonDivisor as gcd } from './learning-rational-gcd';
export interface ExactQ { n: bigint; d: bigint }
export type ExactPolynomial = ExactQ[];
const Z=BigInt(0),O=BigInt(1),T=BigInt(2);
const abs=(n:bigint)=>n<Z?-n:n;
export function exactQ(n:bigint,d=O):ExactQ {if(!d)throw new RangeError('Zero denominator');if(d<Z){n=-n;d=-d;}const g=gcd(abs(n),d);return {n:n/g,d:d/g};}
export const exactAdd=(a:ExactQ,b:ExactQ)=>exactQ(a.n*b.d+b.n*a.d,a.d*b.d);
export const exactNeg=(a:ExactQ):ExactQ=>({n:-a.n,d:a.d});
export const exactSub=(a:ExactQ,b:ExactQ)=>exactAdd(a,exactNeg(b));
export const exactMul=(a:ExactQ,b:ExactQ)=>exactQ(a.n*b.n,a.d*b.d);
export const exactDiv=(a:ExactQ,b:ExactQ)=>exactQ(a.n*b.d,a.d*b.n);
export const exactCompare=(a:ExactQ,b:ExactQ)=>a.n*b.d-b.n*a.d;
export const exactAbs=(a:ExactQ):ExactQ=>({n:abs(a.n),d:a.d});
const zero=()=>exactQ(Z);
function trim(p:ExactPolynomial):ExactPolynomial {const out=p.slice();while(out.length>1&&!out[out.length-1].n)out.pop();return out.length?out:[zero()];}
const isZero=(p:ExactPolynomial)=>p.length===1&&!p[0].n;
export function exactPolynomialEvaluate(p:ExactPolynomial,x:ExactQ):ExactQ {let v=zero();for(let i=p.length-1;i>=0;i--)v=exactAdd(exactMul(v,x),p[i]);return v;}
export function exactPolynomialDerivative(p:ExactPolynomial):ExactPolynomial {return trim(p.slice(1).map((x,i)=>exactMul(x,exactQ(BigInt(i+1)))));}
export function exactPolynomialDivide(p:ExactPolynomial,d:ExactPolynomial) {
 const divisor=trim(d);if(isZero(divisor))throw new RangeError('Zero polynomial divisor');let remainder=trim(p);
 const quotient=Array.from({length:Math.max(1,remainder.length-divisor.length+1)},zero);
 while(!isZero(remainder)&&remainder.length>=divisor.length){const shift=remainder.length-divisor.length,factor=exactDiv(remainder[remainder.length-1],divisor[divisor.length-1]);quotient[shift]=factor;
  for(let i=0;i<divisor.length;i++)remainder[i+shift]=exactSub(remainder[i+shift],exactMul(factor,divisor[i]));remainder=trim(remainder);
 }
 return {quotient:trim(quotient),remainder};
}
function monic(p:ExactPolynomial):ExactPolynomial {const t=trim(p);if(isZero(t))return t;return t.map(x=>exactDiv(x,t[t.length-1]));}
export function exactPolynomialGcd(a:ExactPolynomial,b:ExactPolynomial):ExactPolynomial {a=monic(a);b=monic(b);while(!isZero(b)){const r=exactPolynomialDivide(a,b).remainder;a=b;b=monic(r);}return monic(a);}
const squareFree=(p:ExactPolynomial)=>exactPolynomialDivide(p,exactPolynomialGcd(p,exactPolynomialDerivative(p))).quotient;
function sturm(p:ExactPolynomial) {
 const f=monic(p),sequence=[f,exactPolynomialDerivative(f)];
 if(f.length<=1)return [f];
 for(;;){const remainder=exactPolynomialDivide(sequence[sequence.length-2],sequence[sequence.length-1]).remainder;if(isZero(remainder))break;
  // Scale by a positive number only: a negative normalization changes variations.
  const scale=exactAbs(remainder[remainder.length-1]);sequence.push(remainder.map(x=>exactDiv(exactNeg(x),scale)));
 }
 return sequence;
}
function variations(sequence:ExactPolynomial[],x:ExactQ){let previous=0,count=0;for(const p of sequence){const n=exactPolynomialEvaluate(p,x).n,sign=n>Z?1:n<Z?-1:0;if(sign){if(previous&&previous!==sign)count++;previous=sign;}}return count;}
export interface ExactRootInterval { lower:ExactQ; upper:ExactQ; multiplicity:number }
export function isolateExactPolynomialRoots(coefficients:ExactPolynomial,options:{precisionBits?:number;maximumBisections?:number}={}) {
 const precision=options.precisionBits??112,maximum=options.maximumBisections??50000;
 if(!Number.isInteger(precision)||precision<32||precision>8192||!Number.isInteger(maximum)||maximum<1)throw new RangeError('Invalid polynomial isolation budget');
 const p=trim(coefficients.map(x=>exactQ(x.n,x.d)));if(isZero(p))throw new RangeError('The zero polynomial has no finite isolated root set');
 if(p.length===1)return {roots:[] as ExactRootInterval[],degree:0,distinctRootCount:0,bisections:0};
 const sf=squareFree(p),sequence=sturm(sf),scale=O<<BigInt(precision);
 let ratio=zero();for(const x of p.slice(0,-1)){const r=exactDiv(exactAbs(x),exactAbs(p[p.length-1]));if(exactCompare(r,ratio)>Z)ratio=r;}
 const bound=exactQ(O+(ratio.n+ratio.d-O)/ratio.d),left=exactNeg(bound),right=bound;
 // At a simple zero, ignoring the first zero entry gives V(x+).
 // Thus V(a)-V(b)-1_{sf(b)=0} counts roots strictly inside (a,b).
 const countOpen=(a:ExactQ,b:ExactQ)=>variations(sequence,a)-variations(sequence,b)-(exactPolynomialEvaluate(sf,b).n===Z?1:0);
 const total=countOpen(left,right),pending=[{a:left,b:right,count:total}],isolated:{lower:ExactQ;upper:ExactQ}[]=[];
 let bisections=0;
 while(pending.length){const item=pending.pop()!;const {a,b,count}=item;if(!count)continue;
  const sameSign=a.n>Z&&b.n>Z||a.n<Z&&b.n<Z,near=exactCompare(exactAbs(a),exactAbs(b))<Z?exactAbs(a):exactAbs(b);
  if(count===1&&sameSign&&exactCompare(exactMul(exactSub(b,a),exactQ(scale)),near)<=Z&&exactPolynomialEvaluate(sf,a).n!==Z&&exactPolynomialEvaluate(sf,b).n!==Z){isolated.push({lower:a,upper:b});continue;}
  if(++bisections>maximum)throw new RangeError('Polynomial isolation bisection budget exhausted');
  const mid=exactDiv(exactAdd(a,b),exactQ(T)),point=exactPolynomialEvaluate(sf,mid).n===Z;
  if(point)isolated.push({lower:mid,upper:mid});
  const leftCount=countOpen(a,mid),rightCount=countOpen(mid,b);
  if(leftCount+rightCount+(point?1:0)!==count)throw new Error('Sturm root count invariant violated');
  if(rightCount)pending.push({a:mid,b,count:rightCount});if(leftCount)pending.push({a,b:mid,count:leftCount});
 }
 isolated.sort((a,b)=>exactCompare(a.lower,b.lower)<Z?-1:1);
 const chains:{polynomial:ExactPolynomial;sequence:ExactPolynomial[]}[]=[];
 let repeated=exactPolynomialGcd(p,exactPolynomialDerivative(p));
 while(repeated.length>1){const f=squareFree(repeated);chains.push({polynomial:f,sequence:sturm(f)});repeated=exactPolynomialGcd(repeated,exactPolynomialDerivative(repeated));}
 const roots=isolated.map(r=>{let multiplicity=1;for(const chain of chains){const present=exactCompare(r.lower,r.upper)===Z?exactPolynomialEvaluate(chain.polynomial,r.lower).n===Z:variations(chain.sequence,r.lower)-variations(chain.sequence,r.upper)>0;if(present)multiplicity++;else break;}return {...r,multiplicity};});
 if(roots.length!==total)throw new Error('Incomplete polynomial root set');
 for(let i=1;i<roots.length;i++)if(exactCompare(roots[i-1].upper,roots[i].lower)>=Z)throw new Error('Root isolation intervals overlap');
 return {roots,degree:p.length-1,distinctRootCount:total,bisections};
}
