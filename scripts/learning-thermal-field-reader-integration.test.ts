import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {thermalFieldReader} from '../src/data/thermal-field-reader';
import {thermalFieldTeachingId as id} from '../src/lib/learning-thermal-field-teaching';
import {getLearningContent} from '../src/lib/learning-content';
const before=JSON.parse(readFileSync('scripts/fixtures/pre-thermal-field-registry-hashes.json','utf8')) as {unitOrder:string[];guideOrder:string[];units:Record<string,string>;guides:Record<string,string>;domains:Record<string,string>;manuscripts:Record<string,string>;preservedFiles:Record<string,string>};
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const root='.runtime/learning/thermal-field-reader-integration';mkdirSync(root,{recursive:true});
test('new reader appends after601; existing units,guides,scope,1202manuscripts andmenu bytes survive',()=>{
 assert(learningUnits.length>=602);assert.equal(new Set(learningUnits.map(u=>u.id)).size,learningUnits.length);assert.deepEqual(learningUnits.slice(0,601).map(u=>u.id),before.unitOrder);assert.deepEqual(Object.keys(learningGuides).slice(0,601),before.guideOrder);
 for(const unit of learningUnits.slice(0,601)){assert.equal(hash(unit),before.units[unit.id]);assert.equal(hash(learningGuides[unit.id]),before.guides[unit.id]);}
 for(const[p,h]of Object.entries({...before.manuscripts,...before.preservedFiles}))assert.equal(createHash('sha256').update(readFileSync(p)).digest('hex'),h,p);
 for(const domain of curriculumDomains)assert.equal(hash({...domain,unitIds:domain.unitIds.filter(prior=>before.unitOrder.includes(prior))}),before.domains[domain.id]);
 assert.deepEqual(learningUnits[601],thermalFieldReader.unit);assert.deepEqual(learningGuides[id],thermalFieldReader.guide);assert.equal(learningUnits.slice(0,602).reduce((n,u)=>n+learningGuides[u.id].experiments.length,0),1869);
 assert(thermalFieldReader.unit.prerequisiteIds.every(prior=>before.unitOrder.includes(prior)));assert(curriculumDomains.find(d=>d.id==='field-multiscale')!.unitIds.includes(id));
 writeFileSync(root+'/preservation.json',JSON.stringify({previousUnits:601,preservedManuscripts:Object.keys(before.manuscripts).length,newRegisteredUnits:602,newGuides:1869,menuPreserved:true},null,2)+'\n');
});
test('registered bilingual manuscripts render21sections and9equations; reader owns exactly8questions',async()=>{
 const content=await getLearningContent(id);for(const language of ['ko','en'] as const){assert.equal(content.outline[language].length,21);assert(!content[language].includes('katex-error'));assert(content[language].includes('<math'));const raw=readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');assert.equal((raw.match(/^\$\$/gm)??[]).length,18);assert(!raw.includes('preparation draft'));const draft=readFileSync(`docs/online-learning/physics-math-for-ai/drafts/${id}.${language}.md`,'utf8');const snippets=(s:string)=>[...s.matchAll(/```python\n([\s\S]*?)\n```/g)].map(m=>m[1]);assert.deepEqual(snippets(raw),snippets(draft));}
 assert(!/[가-힣]/.test(content.en));assert.equal(thermalFieldReader.unit.questions.length,8);assert.equal(thermalFieldReader.guide.steps.length,6);assert.equal(thermalFieldReader.guide.experiments.length,6);
 const lab=readFileSync('src/components/learning/thermal-field-lab.tsx','utf8');assert(lab.includes('!embedded&&'));assert(readFileSync('src/components/learning/thermal-field-reader-lab.tsx','utf8').includes('embedded'));assert(readFileSync('src/components/learning/interactive-book.tsx','utf8').includes('isThermalFieldTeachingId(unit.id) ? <ThermalFieldReaderLab'));
});
test('full reader bundle excludes thermal calculation,independent math andobservation engines',async()=>{
 const bundle=await build({entryPoints:['src/components/learning/interactive-book.tsx'],bundle:true,write:false,platform:'browser',format:'esm',splitting:true,outdir:root+'/graph',metafile:true,jsx:'automatic',logLevel:'silent'}),inputs=Object.keys(bundle.metafile!.inputs);
 for(const name of ['learning-thermal-field.ts','learning-thermal-field-evaluator.ts','learning-thermal-field-validation.ts','learning-thermal-field-validation-evaluator.ts','learning-thermal-field-observations.ts','learning-quantum-hilbert-core.ts'])assert(!inputs.includes('src/lib/'+name),name);
 writeFileSync(root+'/reader-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');
});
