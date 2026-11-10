import test from 'node:test';
import assert from 'node:assert/strict';
import {parseThermodynamicDraft as parse} from '../src/components/learning/thermodynamic-number-field';
test('nonzero decimal underflow cannot change an open connection or nonzero heat into physical zero',()=>{for(const raw of ['1e-999','-1e-999','1e-324','.0001e-999'])assert.equal(parse(raw,-10,10),null,raw);assert.equal(parse('5e-324',0,2),Number.MIN_VALUE);assert.equal(parse('0e-999',0,2),0);assert.equal(parse('-0.0',0,2),0);});
test('decimal and scientific physical inputs preserve small nonzero values, signs and declared bounds',()=>{assert.equal(parse('3.47e-14',0,2),3.47e-14);assert.equal(parse('-2.5',-10,10),-2.5);assert.equal(parse('  .4  ',0,2),.4);assert.equal(parse('2',0,2),2);for(const s of ['', ' ', 'NaN','Infinity','0x1','1.2.3','3','1e999'])assert.equal(parse(s,0,2),null);assert.equal(parse('2.5',1,8,true),null);assert.equal(parse('2e0',1,8,true),2);});
