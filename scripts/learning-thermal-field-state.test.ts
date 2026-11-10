import assert from 'node:assert/strict';
import {test} from 'node:test';
import {emptyThermalState,readThermalState,readThermalDraft,parseStoredThermalState} from '../src/lib/learning-thermal-field-state';
import {parseThermalDraft} from '../src/lib/learning-thermal-field-view';
const limits={steps:6,guides:6,questions:8,options:3};
test('applied input and unfinished draft restore separately with all learning/display choices',()=>{
 const model=structuredClone(emptyThermalState(limits));model.draft.temperature='1e-';model.draft.source.first='unfinished';model.step=4;model.motion=false;model.guide=3;model.explain=true;model.answers[2]=1;model.display={kind:'precision',scale:'signed-log',selected:3,flat:false};
 const restored=parseStoredThermalState(JSON.stringify({version:1,model}),limits);assert(restored);assert.equal(restored.input.temperature,'2');assert.equal(restored.draft.temperature,'1e-');assert.equal(parseThermalDraft(restored.draft),null);assert.deepEqual(restored,model);assert(Object.isFrozen(restored.draft.mass)&&Object.isFrozen(restored.answers));
});
test('malformed or forged state cannot restore receipts,run flags,unbounded drafts or out-of-range choices',()=>{
 const base=emptyThermalState(limits);
 for(const patch of [{step:6},{guide:-1},{motion:'true'},{answers:[0]},{answers:Array(8)},{answers:Array(8).fill(3)},{running:true},{response:{}},{display:{...base.display,selected:4}},{display:{...base.display,kind:'fake'}},{draft:{...base.draft,shape:'x'.repeat(13)}},{draft:{...base.draft,source:{...base.draft.source,value:'x'.repeat(1281)}}}])assert.equal(readThermalState({...base,...patch},limits),null);
 for(const raw of ['{',JSON.stringify({version:2,model:base}),JSON.stringify({version:1,model:base,receipt:{}}),'x'.repeat(24001)])assert.equal(parseStoredThermalState(raw,limits),null);
});
test('descriptor-only draft and state reads reject accessors without calling them',()=>{
 let calls=0;const base=structuredClone(emptyThermalState(limits));Object.defineProperty(base.draft.mass,'first',{get(){calls++;return'0';}});assert.equal(readThermalState(base,limits),null);
 const source={...emptyThermalState(limits).draft};Object.defineProperty(source,'shape',{get(){calls++;return'2';}});assert.equal(readThermalDraft(source),null);assert.equal(calls,0);
});
