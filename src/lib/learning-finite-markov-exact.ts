/** Exact rational output for finite chains; no numerical Markov model runs here. */
import {binaryRatioObservation} from './learning-binary-products';
import {rationalGreatestCommonDivisor as gcd} from './learning-rational-gcd';
import type {PartitionObservation} from './learning-partition-exact-core';

export interface MarkovFraction {n:bigint;d:bigint}
const zero=BigInt(0),one=BigInt(1);

export function markovFraction(n:bigint,d=one):MarkovFraction {
 if(d===zero)throw new RangeError('Expected a nonzero rational denominator');
 if(d<zero){n=-n;d=-d;}
 const factor=gcd(n,d);
 return {n:n/factor,d:d/factor};
}

/** Equal rational endpoints certify exact values, including nonzero underflow.
 * Number and magnitude-log fields are display projections, not exact arithmetic.
 */
export function exactMarkovObservation(n:bigint,d=one):PartitionObservation {
 const q=markovFraction(n,d);
 const numerator={coefficient:q.n,exponent:0},denominator={coefficient:q.d,exponent:0};
 const projected=binaryRatioObservation(numerator,denominator);
 const bound={numerator:{coefficient:q.n.toString(),exponent:0},denominator:{coefficient:q.d.toString(),exponent:0}};
 return {...projected,logResolution:projected.sign===0?'not-applicable':projected.logMagnitude===null?'outside-number-range':'finite',
  resolution:projected.sign===0?'structural-zero':'exact-rational',bounds:{lower:bound,upper:bound}};
}
