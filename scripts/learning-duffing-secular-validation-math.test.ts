import test from 'node:test';
import assert from 'node:assert/strict';
import {Rational as Q} from '../src/lib/learning-graph-gauge-validation-math';
import {independentDuffingPi,createIndependentDuffingContext,sqrtRange,independentDuffingExponent} from '../src/lib/learning-duffing-secular-validation-math';
import {directedPi} from '../src/lib/learning-directed-pi';
import {createQuantumEvolutionContext} from '../src/lib/learning-quantum-evolution-interval';
const B=BigInt,le=(a:Q,b:Q)=>a.n*b.d<=b.n*a.d;
test('the independent Størmer identity has an exact positive pi/4 tangent and canonical pi cell',()=>{
 type C=[Q,Q];const mul=(a:C,b:C):C=>[a[0].mul(b[0]).sub(a[1].mul(b[1])),a[0].mul(b[1]).add(a[1].mul(b[0]))],power=(x:C,n:number)=>{let r:C=[Q.read('1'),Q.read('0')];for(let j=0;j<n;j++)r=mul(r,x);return r;};
 const product=mul(mul(power([Q.read('1'),Q.read('1/18')],12),power([Q.read('1'),Q.read('1/57')],8)),power([Q.read('1'),Q.read('-1/239')],5));assert.equal(product[0].text(),product[1].text());assert(product[0].n>B(0));const a=Q.read('1/18'),b=Q.read('1/57'),lower=(x:Q)=>x.sub(x.mul(x).mul(x).div(Q.read('3'))),angleLower=lower(a).mul(Q.read('12')).add(lower(b).mul(Q.read('8'))).sub(Q.read('5/239')),angleUpper=a.mul(Q.read('12')).add(b.mul(Q.read('8')));assert(angleLower.n>B(0));assert(le(angleUpper,Q.read('1')));assert.deepEqual(independentDuffingPi(512),directedPi(512));
});
test('restoring roots and independent signed-phase arithmetic preserve their enclosures',()=>{
 const rational=sqrtRange(Q.read('25/49'));assert.equal(rational.lower.text(),'5/7');assert.equal(rational.upper.text(),'5/7');const two=Q.read('2'),root=sqrtRange(two);assert(le(root.lower.mul(root.lower),two));assert(le(two,root.upper.mul(root.upper)));
 const independent=createIndependentDuffingContext(),producer=createQuantumEvolutionContext();
 for(const value of ['0','1/3','-1/7','999999999999999999','-6873594999990000000']){const x=Q.read(value),a=independent.sinCos({lower:x,upper:x}),b=producer.sinCos({lower:x,upper:x});for(const key of ['sin','cos']as const)assert(a[key].lower<=b[key].upper&&b[key].lower<=a[key].upper,value+key);}
 const e=independentDuffingExponent(512).expNegativeRatio({coefficient:B(0),exponent:0},{coefficient:B(1),exponent:0});assert.equal(e.lower,B(1)<<B(512));assert.equal(e.lower,e.upper);
});
