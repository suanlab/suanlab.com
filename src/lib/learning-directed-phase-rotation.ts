/** Directed sin/cos for bounded exact rational radians. Range halving,
 * outward fixed-point Taylor polynomials, analytic remainders, angle doubling. */
export interface DirectedPhaseRotation {sin:{lower:bigint;upper:bigint};cos:{lower:bigint;upper:bigint};scale:bigint;precisionBits:number}
const Z=BigInt(0),O=BigInt(1),TWO=BigInt(2);
export function directedPhaseRotation(n:bigint,d:bigint,precisionBits=256):DirectedPhaseRotation{
 if(typeof n!=='bigint'||typeof d!=='bigint'||d<=Z||!Number.isInteger(precisionBits)||precisionBits<64||precisionBits>512||n.toString(2).length>4096||d.toString(2).length>4096||n*n>(O<<BigInt(96))*d*d)throw new RangeError('Exact rational radians within[-2^48,2^48], positive denominator and64..512bits required');
 const scale=O<<BigInt(precisionBits);if(n===Z)return{sin:{lower:Z,upper:Z},cos:{lower:scale,upper:scale},scale,precisionBits};
 const guard=32,workBits=precisionBits+guard,S=O<<BigInt(workBits);
 const floor=(a:bigint,b:bigint)=>a>=Z?a/b:-((-a+b-O)/b),ceil=(a:bigint,b:bigint)=>-floor(-a,b);
 type I={lower:bigint;upper:bigint};const point=(a:bigint):I=>({lower:a,upper:a});
 const add=(a:I,b:I):I=>({lower:a.lower+b.lower,upper:a.upper+b.upper}),neg=(a:I):I=>({lower:-a.upper,upper:-a.lower});
 const mul=(a:I,b:I):I=>{const q=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper],lo=q.reduce((a,b)=>a<b?a:b),hi=q.reduce((a,b)=>a>b?a:b);return{lower:floor(lo,S),upper:ceil(hi,S)}};
 const divide=(a:I,b:bigint):I=>({lower:floor(a.lower,b),upper:ceil(a.upper,b)});
 const clamp=(a:I):I=>({lower:a.lower<-S?-S:a.lower,upper:a.upper>S?S:a.upper});
 let denominator=d,halvings=0;while((n<Z?-n:n)*BigInt(8)>denominator){denominator*=TWO;halvings++;}
 const angle={lower:floor(n*S,denominator),upper:ceil(n*S,denominator)},square=mul(angle,angle),terms=Math.ceil((workBits+16)/4);
 let st=angle,ct=point(S),sin=st,cos=ct;
 for(let j=1;j<terms;j++){
  st=neg(divide(mul(st,square),BigInt((2*j)*(2*j+1))));ct=neg(divide(mul(ct,square),BigInt((2*j-1)*(2*j))));
  sin=add(sin,st);cos=add(cos,ct);
 }
 // Both polynomials can be treated through degree2N−1 (zero coefficients
 // included). The Lagrange remainder is <=(1/8)^(2N)/(2N)!.
 let factorial=O;for(let j=2;j<=2*terms;j++)factorial*=BigInt(j);
 const tail=ceil(S,(O<<BigInt(6*terms))*factorial);
 sin=clamp({lower:sin.lower-tail,upper:sin.upper+tail});cos=clamp({lower:cos.lower-tail,upper:cos.upper+tail});
 for(let k=0;k<halvings;k++){
  const sc=mul(sin,cos),cc=mul(cos,cos),ss=mul(sin,sin);
  sin=clamp({lower:TWO*sc.lower,upper:TWO*sc.upper});cos=clamp(add(cc,neg(ss)));
 }
 const shrink=(a:I)=>({lower:floor(a.lower,O<<BigInt(guard)),upper:ceil(a.upper,O<<BigInt(guard))});
 return{sin:shrink(sin),cos:shrink(cos),scale,precisionBits};
}
