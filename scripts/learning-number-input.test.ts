import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {parseNumberDraft as parse} from '../src/lib/learning-number-input';
import {NumberField,VectorFields} from '../src/components/learning/lab-controls';
test('nonzero underflow, overflow and malformed drafts cannot become model values',()=>{
 for(const raw of ['1e-999','-1e-999','2.47e-324','1e999','NaN','Infinity','0x10','',' ','1.2.3'])assert.equal(parse(raw,-Number.MAX_VALUE,Number.MAX_VALUE),null,raw);
 assert.equal(parse('5e-324',0,1),Number.MIN_VALUE);assert.equal(parse('0e-999',0,1),0);assert.equal(parse('-0.0',-1,1),0);assert.equal(parse('-1e-310',-1,1),-1e-310);
});
test('integer count syntax is checked before fractional decimal digits disappear through rounding',()=>{
 for(const raw of ['2.0000000000000000001','1.9999999999999999999','12.3','123e-1'])assert.equal(parse(raw,0,100,true),null,raw);
 for(const raw of ['2','2.000','20e-1','2e0'])assert.equal(parse(raw,0,100,true),2,raw);
 assert.equal(parse('2.0000000000000000001',0,100),2);assert.equal(parse('120e-1',0,100,true),12);
});
test('standalone scalar and vector controls render accessible labels without a language provider or browser APIs',()=>{
 const html=renderToStaticMarkup(React.createElement(NumberField,{label:'Coefficient',value:.5,min:-2,max:2,onChange:()=>{throw Error('SSR callback');}}));assert.match(html,/aria-labelledby=/);assert.match(html,/aria-invalid="false"/);assert.doesNotMatch(html,/[가-힣]/);
 const vector=renderToStaticMarkup(React.createElement(VectorFields,{label:'v',value:[1,2,3],change:()=>{throw Error('SSR callback');}}));assert.equal((vector.match(/type="number"/g)??[]).length,3);
});
