import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {build} from 'esbuild';
import {realTimePhaseSteps} from '../src/data/real-time-phase-teaching';
import {realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';
import {realTimeReferenceSquare,realTimeReferencePhases,realTimeReferenceAction,realTimeReferenceDamping,
  realTimeReferenceDensities,realTimeReferenceVariances,realTimeReferenceFolds} from '../src/lib/learning-real-time-phase-reference';
import Graphic from '../src/components/learning/real-time-phase-graphic';
import Concept from '../src/components/learning/real-time-phase-concept';
import Cover from '../src/components/learning/real-time-phase-cover';
import {LanguageProvider} from '../src/components/language-provider';

test('bilingual concept progression keeps example links valid and English free of Korean',()=>{
  assert.equal(realTimePhaseSteps.length,13);assert.equal(new Set(realTimePhaseSteps.map(s=>s.id)).size,13);
  const examples=new Set(realTimePhaseExamples.map(x=>x.id));
  for(const step of realTimePhaseSteps){
    for(const field of [step.title,step.body,step.diagram]){assert(field.ko.length>0);assert(field.en.length>0);assert(!/[가-힣]/u.test(field.en));}
    for(const id of step.examples)assert(examples.has(id),id);
  }
});

test('fixed numerical drawing references match independent Fraction/Python calculations',()=>{
  const process=spawnSync('python3',['scripts/learning/verify-real-time-concept-references.py'],{encoding:'utf8'});
  assert.equal(process.status,0,process.stderr);const oracle=JSON.parse(process.stdout);
  assert.deepEqual(realTimeReferenceVariances.map(({hbar,fixedText,scaledText})=>({hbar,fixedText,scaledText})),oracle.variances);
  const fractionValue=(text:string)=>{const [n,d='1']=text.split('/');return Number(n)/Number(d);};
  for(let i=0;i<realTimeReferenceVariances.length;i++){
    assert.equal(realTimeReferenceVariances[i].fixed,fractionValue(oracle.variances[i].fixedText));
    assert.equal(realTimeReferenceVariances[i].scaled,fractionValue(oracle.variances[i].scaledText));
  }
  const compare=(actual:unknown,expected:unknown)=>{
    if(typeof expected==='number'){assert.equal(typeof actual,'number');assert(Math.abs((actual as number)-expected)<2e-12,`${actual} ≠ ${expected}`);}
    else if(Array.isArray(expected)){assert(Array.isArray(actual));assert.equal(actual.length,expected.length);expected.forEach((v,i)=>compare(actual[i],v));}
    else if(expected&&typeof expected==='object'){for(const [k,v] of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v);}
    else assert.equal(actual,expected);
  };
  compare(realTimeReferencePhases,oracle.phases);compare(realTimeReferenceAction,oracle.action);
  compare(realTimeReferenceDamping,oracle.damping);compare(realTimeReferenceDensities,oracle.densities);compare(realTimeReferenceFolds,oracle.folds);
  assert.deepEqual(realTimeReferenceSquare[0],realTimeReferenceSquare.at(-1));
  const steps=realTimeReferenceSquare.slice(1).map((p,i)=>[p[0]-realTimeReferenceSquare[i][0],p[1]-realTimeReferenceSquare[i][1]]);
  assert.deepEqual(steps,[[1,0],[0,1],[-1,0],[0,-1]]);
  assert.equal(oracle.negativeStationaryAction,'1/4');assert.equal(oracle.zeroPivotDeterminant,'-1');
  assert.deepEqual(oracle.packetCovariance,[['5/4','1/4'],['1/4','1/4']]);
});

test('26 diagrams, two compact references and the saved concept render without browser APIs',()=>{
  for(const language of ['ko','en'] as const){
    const rendered=new Set<string>();
    for(let step=0;step<realTimePhaseSteps.length;step++){
      const html=renderToStaticMarkup(createElement(Graphic,{step,language}));rendered.add(html);
      assert(html.includes(`data-real-time-reference="${step}"`));assert(html.includes('<title'));assert(html.includes('<desc'));
      assert(!/NaN|Infinity|undefined/.test(html));if(language==='en')assert(!/[가-힣]/u.test(html));
    }
    assert.equal(rendered.size,13);
    const compact=renderToStaticMarkup(createElement(Graphic,{language,compact:true}));assert(compact.includes('data-reference-compact="true"'));
    const concept=renderToStaticMarkup(createElement(Concept,{language}));assert(concept.includes('data-real-time-concept="0"'));if(language==='en')assert(!/[가-힣]/u.test(concept));
    const invalid=renderToStaticMarkup(createElement(Concept,{language,selectedStep:NaN}));assert(invalid.includes('data-real-time-concept="0"'));
  }
  assert(renderToStaticMarkup(createElement(LanguageProvider,null,createElement(Cover))).includes('data-reference-compact="true"'));
});

test('drawing/controller modules keep numerical producer, validator, Worker and Three.js code out',async()=>{
  const result=await build({entryPoints:['src/components/learning/real-time-phase-concept.tsx'],bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',metafile:true});
  const graph=Object.keys(result.metafile!.inputs);
  assert(graph.includes('src/lib/learning-real-time-phase-reference.ts'));
  assert(!graph.some(p=>/learning-real-time-phase(?:\.ts|-exact|-validation|-evaluator|-core-wire|-controller)|learning-quantum-hilbert-core|three\/|\.worker\.ts/.test(p)),graph.join('\n'));
});
