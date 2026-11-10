/** Independent exact checker arithmetic. No producer arithmetic is imported.
 * Cross cancellation keeps certificate checks on a common integer lattice.
 */
const Z=BigInt(0),O=BigInt(1);
export function checkerGcd(a:bigint,b:bigint):bigint {a=a<Z?-a:a;b=b<Z?-b:b;while(b)[a,b]=[b,a%b];return a;}
export class CheckerRational {
  private constructor(readonly n:bigint,readonly d:bigint) {}
  static of(n:bigint,d=O):CheckerRational {
    if(d===Z)throw new RangeError('Zero checker denominator');if(d<Z){n=-n;d=-d;}
    const g=checkerGcd(n,d);return new CheckerRational(n/g,d/g);
  }
  static input(value:string) {const[n,d='1']=value.split('/');return CheckerRational.of(BigInt(n),BigInt(d));}
  add(x:CheckerRational):CheckerRational {
    const g=checkerGcd(this.d,x.d),a=this.d/g,b=x.d/g,n=this.n*b+x.n*a,h=checkerGcd(n,g);
    return new CheckerRational(n/h,a*(x.d/h));
  }
  neg(){return new CheckerRational(-this.n,this.d);}
  sub(x:CheckerRational){return this.add(x.neg());}
  mul(x:CheckerRational):CheckerRational {
    const a=checkerGcd(this.n,x.d),b=checkerGcd(x.n,this.d);
    return new CheckerRational((this.n/a)*(x.n/b),(this.d/b)*(x.d/a));
  }
  div(x:CheckerRational):CheckerRational {
    if(x.n===Z)throw new RangeError('Zero checker divisor');
    const n=x.n<Z?-x.n:x.n,sign=x.n<Z?-O:O,a=checkerGcd(this.n,n),b=checkerGcd(x.d,this.d);
    return new CheckerRational(sign*(this.n/a)*(x.d/b),(this.d/b)*(n/a));
  }
  equals(x:{n:bigint;d:bigint}){return this.n*x.d===x.n*this.d;}
  text(){return this.d===O?String(this.n):`${this.n}/${this.d}`;}
}
export const checkerZero=CheckerRational.of(Z),checkerOne=CheckerRational.of(O),checkerInteger=(n:number)=>CheckerRational.of(BigInt(n));
export const checkerLcm=(a:bigint,b:bigint)=>a/checkerGcd(a,b)*b;
/** Check bounds before gcd; cache is per verification,not global hostile data. */
export function checkerCanonicalReader() {
  const cache=new Map<string,CheckerRational>();
  return(value:unknown,maximumBits=65536):CheckerRational=>{
    if(typeof value!=='string'||value.length>24000||! /^(?:0|-?[1-9][0-9]*)(?:\/[1-9][0-9]*)?$/.test(value))throw new RangeError('Bounded canonical rational required');
    const cached=cache.get(value);if(cached){if((cached.n<Z?-cached.n:cached.n).toString(2).length>maximumBits||cached.d.toString(2).length>maximumBits)throw new RangeError('Checker rational bit budget');return cached;}
    const[n,d='1']=value.split('/'),numerator=BigInt(n),denominator=BigInt(d);
    if((numerator<Z?-numerator:numerator).toString(2).length>maximumBits||denominator.toString(2).length>maximumBits)throw new RangeError('Checker rational bit budget');
    const result=CheckerRational.of(numerator,denominator);if(result.text()!==value)throw new RangeError('Noncanonical checker rational');
    if(cache.size===256)cache.delete(cache.keys().next().value!);cache.set(value,result);return result;
  };
}
export interface CheckerGaussianInteger {real:bigint;imaginary:bigint;}
export const gi=(real:bigint,imaginary=Z):CheckerGaussianInteger=>({real,imaginary});
export const ga=(a:CheckerGaussianInteger,b:CheckerGaussianInteger)=>gi(a.real+b.real,a.imaginary+b.imaginary);
export const gn=(a:CheckerGaussianInteger)=>gi(-a.real,-a.imaginary);
export const gm=(a:CheckerGaussianInteger,b:CheckerGaussianInteger)=>gi(a.real*b.real-a.imaginary*b.imaginary,a.real*b.imaginary+a.imaginary*b.real);
export const gs=(a:CheckerGaussianInteger,n:bigint)=>gi(a.real*n,a.imaginary*n);
export const ge=(a:CheckerGaussianInteger,b:CheckerGaussianInteger)=>a.real===b.real&&a.imaginary===b.imaginary;
export function checkerExactDivision(n:bigint,d:bigint):bigint {if(d===Z||n%d!==Z)throw new RangeError('Certificate does not lie on its integer lattice');return n/d;}
