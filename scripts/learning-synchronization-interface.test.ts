import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {synchronizationKinds as kinds,synchronizationPresets as presets,synchronizationPresetInput as preset,editSynchronizationField as edit,synchronizationFields} from '../src/lib/learning-synchronization';
import {synchronizationInputKey as key} from '../src/lib/learning-synchronization-protocol';
import {synchronizationSteps as steps} from '../src/lib/learning-synchronization-concepts';
import {synchronizationPlotPath as path} from '../src/components/learning/synchronization-view';
for(const [i,kind] of kinds.entries())test(`reader preparations remain independent reference inputs: ${kind}`,()=>{
 const references=JSON.parse(readFileSync(`scripts/fixtures/${['hopf-amplitude','pair-phase-locking','finite-kuramoto'][i]}-reference.json`,'utf8')).cases;
 assert.equal(steps[kind].length,3);for(const [j,p] of presets[kind].entries()){assert.deepEqual(p.input.parameters,references.find((r:{label:string})=>r.label===p.id).input);assert.equal(key(preset(kind,j)),key(p.input));assert(Object.isFrozen(p.input.parameters));}
 for(const s of steps[kind]){assert(s.presets.length>0);assert(s.presets.every(i=>presets[kind][i]));assert(s.options[s.answer]);assert(s.explanation.ko.length>40&&s.explanation.en.length>60);assert(!/[가-힣]/.test(JSON.stringify(s,(k,v)=>k==='ko'?undefined:v)));}
 const input=preset(kind,0);assert(synchronizationFields(input).every(f=>f.label.ko&&f.label.en));assert.equal(edit(input,'timeNumerator','1e-9999'),null);assert.equal(edit(input,'timeNumerator','1.00000000000000001'),null);assert.equal(edit(input,'timeDenominator','0'),null);assert.equal(edit(input,'unknown','1'),null);assert.equal(edit(input,'timeNumerator','5'),null);assert(edit(input,'timeNumerator','2'));
});
test('time plots retain gaps and use supplied physical times',()=>{
 assert.equal(path([0,1,3,4],[1,null,2,3],x=>x*10,y=>y),'M0 1  M30 2 L40 3');
 assert.equal(path([0,1],[null,null],x=>x,y=>y),' ');
});
test('lab and concept graph does not pull numerical engines onto the main thread',async()=>{
 const b=await build({entryPoints:['src/components/learning/synchronization-lab.tsx','src/components/learning/synchronization-concept.tsx'],outdir:'unused',format:'esm',bundle:true,write:false,metafile:true,platform:'browser',jsx:'automatic'}),files=Object.keys(b.metafile!.inputs);
 assert(!files.some(p=>/learning-(hopf-amplitude|pair-phase-locking|finite-kuramoto|pair-phase-math)\.ts|entropy-production-math|interval-transcendentals/.test(p)),files.join('\n'));
});
