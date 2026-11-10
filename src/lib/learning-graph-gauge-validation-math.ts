/** Independent exact arithmetic for graph response verification; Worker-side only.
 * No producer arithmetic or graph calculator is imported.
 */
export class Rational {
 readonly n:bigint;readonly d:bigint;
 constructor(n:bigint,d=BigInt(1)){
  if(d===BigInt(0))throw Error('Zero denominator');if(d<BigInt(0)){n=-n;d=-d;}
  let a=n<BigInt(0)?-n:n,b=d;while(b)[a,b]=[b,a%b];this.n=n/a;this.d=d/a;
 }
 static read(s:string){const[n,d='1']=s.split('/');return new Rational(BigInt(n),BigInt(d));}
 add(x:Rational){return new Rational(this.n*x.d+x.n*this.d,this.d*x.d);}
 neg(){return new Rational(-this.n,this.d);}sub(x:Rational){return this.add(x.neg());}
 mul(x:Rational){return new Rational(this.n*x.n,this.d*x.d);}div(x:Rational){return new Rational(this.n*x.d,this.d*x.n);}
 text(){return this.d===BigInt(1)?String(this.n):`${this.n}/${this.d}`;}
}
export const Z=new Rational(BigInt(0)),O=new Rational(BigInt(1)),T=new Rational(BigInt(2));
export type Complex=readonly [Rational,Rational];
export const complex=(x:Rational,y=Z):Complex=>[x,y];
export const CZ=complex(Z),CO=complex(O),CI=complex(Z,O);
export const plus=(x:Complex,y:Complex):Complex=>[x[0].add(y[0]),x[1].add(y[1])];
export const conjugate=(x:Complex):Complex=>[x[0],x[1].neg()];
export const minus=(x:Complex,y:Complex):Complex=>[x[0].sub(y[0]),x[1].sub(y[1])];
export const times=(x:Complex,y:Complex):Complex=>[x[0].mul(y[0]).sub(x[1].mul(y[1])),x[0].mul(y[1]).add(x[1].mul(y[0]))];
export const norm=(x:Complex)=>x[0].mul(x[0]).add(x[1].mul(x[1]));
export const wire=(x:Complex)=>({real:x[0].text(),imaginary:x[1].text()});
export const sum=(xs:Rational[])=>xs.reduce((a,b)=>a.add(b),Z);
export const equal=(x:Complex,y:Complex)=>x[0].text()===y[0].text()&&x[1].text()===y[1].text();
export function unitPhase(p:{cayley:string;minusOne:boolean}):Complex{
 if(p.minusOne)return complex(O.neg());const t=Rational.read(p.cayley),tt=t.mul(t),den=O.add(tt);return complex(O.sub(tt).div(den),T.mul(t).div(den));
}
