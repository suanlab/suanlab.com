/** Apply the nine-unit registration to an isolated copy, never to the current workspace. */
import {readFileSync,writeFileSync,realpathSync} from 'node:fs';
import path from 'node:path';
import {informationTeachingIds as ids} from '../src/lib/learning-information-teaching';
const candidate=process.argv[2];if(!candidate||realpathSync(candidate)===realpathSync(process.cwd()))throw Error('An isolated candidate path is required');
const read=(p:string)=>readFileSync(path.join(candidate,p),'utf8'),write=(p:string,s:string)=>writeFileSync(path.join(candidate,p),s);
const edit=(p:string,find:string,replacement:string)=>{const s=read(p);if(!s.includes(find))throw Error(`Missing edit anchor ${p}: ${find}`);write(p,s.replace(find,replacement));};
const imports=(p:string,lines:string)=>{const s=read(p);write(p,s.startsWith("'use client';")?s.replace("'use client';",`'use client';\n${lines}`):`${lines}\n${s}`);};
const book='src/data/interactive-book.ts';if(read(book).includes("import {informationTeaching}"))throw Error('Candidate is already registered');
imports(book,"import {informationTeaching} from './information-teaching';\nimport type {InformationTeachingId} from '@/lib/learning-information-teaching';");
edit(book,'export type LearningUnitId = ',`export type LearningUnitId = ${ids.map(id=>`'${id}'`).join(' | ')} | `);
edit(book,'...qubitTeaching.map(item => item.unit)\n]','...qubitTeaching.map(item => item.unit),\n...informationTeaching.map(item => item.unit)\n]');
const marker="'quantum-instruments-records-feedforward'>\n};";
edit(book,marker,"'quantum-instruments-records-feedforward'>,\n  ...Object.fromEntries(informationTeaching.map(item=>[item.unit.id,item.guide])) as Pick<Record<LearningUnitId,LearningGuide>,InformationTeachingId>\n};");
imports('src/components/learning/interactive-book.tsx',"import {isInformationTeachingId} from '@/lib/learning-information-teaching';\nimport InformationReaderLab from './information-reader-lab';");
edit('src/components/learning/interactive-book.tsx','{isQubitTeachingId(unit.id) ? <QubitReaderLab','{isInformationTeachingId(unit.id) ? <InformationReaderLab key={unit.id} kind={unit.id}/> : isQubitTeachingId(unit.id) ? <QubitReaderLab');
imports('src/components/learning/concept-walkthrough.tsx',"import {isInformationTeachingId} from '@/lib/learning-information-teaching';\nimport InformationConcept from './information-concept';");
edit('src/components/learning/concept-walkthrough.tsx','  if (isQubitTeachingId(kind))','  if (isInformationTeachingId(kind)) return <InformationConcept key={kind} kind={kind} language={language}/>;\n  if (isQubitTeachingId(kind))');
imports('src/components/learning/book-art.tsx',"import {isInformationTeachingId} from '@/lib/learning-information-teaching';\nimport InformationCover from './information-cover';");
edit('src/components/learning/book-art.tsx','  if (isQubitTeachingId(kind))','  if (isInformationTeachingId(kind)) return <InformationCover key={kind} kind={kind}/>;\n  if (isQubitTeachingId(kind))');
for(const family of ['discrimination','entropy','ensemble','relative','bb84','teleportation','error-correction','majorization','capacity']){
 const p=`src/components/learning/information-${family}-lab.tsx`;edit(p,"({language}:{language:'ko'|'en'})","({language,embedded=false}:{language:'ko'|'en';embedded?:boolean})");
 let s=read(p),a=s.indexOf('  <nav className="qubit-actions" aria-label={t(\'개념 단계\',\'Concept steps\')}>'),b=s.indexOf('\n  <h3>',a);if(a<0||b<0)throw Error(`Missing concept boundary ${family}`);s=s.slice(0,a)+'  {!embedded&&<>'+s.slice(a,b)+'</>}'+s.slice(b);
 a=s.indexOf("  <h3>{t('이해 확인','Check your understanding')}</h3>");b=s.lastIndexOf('\n </section>;');if(a<0||b<a)throw Error(`Missing quiz boundary ${family}`);s=s.slice(0,a)+'  {!embedded&&<>'+s.slice(a,b)+'</>}'+s.slice(b);write(p,s);
}
edit('src/components/learning/information-reader-lab.tsx','<Lab language={language}/>','<Lab language={language} embedded/>');
const coverage='docs/online-learning/physics-math-for-ai/curriculum/implementation-coverage.json',c=JSON.parse(read(coverage));if(c.implementedUnitCount!==575)throw Error('Expected the575-unit baseline');c.implementedUnitCount=584;const q=c.domains.find((d:{id:string})=>d.id==='quantum-physics');q.implementedUnitIds.push(...ids);write(coverage,JSON.stringify(c,null,2)+'\n');
// Historical tests retain their exact epoch while allowing later append-only registrations.
const old='scripts/learning-qubit-reader-integration.test.ts';let s=read(old);s=s.replace('assert.equal(learningUnits.length, 575)','assert(learningUnits.length >= 575)').replace('new Set(learningUnits.map(unit => unit.id)).size, 575','new Set(learningUnits.slice(0,575).map(unit => unit.id)).size, 575').replace('learningUnits.reduce((n, unit)','learningUnits.slice(0,575).reduce((n, unit)').replace('learningUnits.slice(571).map','learningUnits.slice(571,575).map').replace('Object.keys(learningGuides).slice(571)','Object.keys(learningGuides).slice(571,575)').replace('domain.unitIds.filter(id => !ids.some(newId => newId === id))','domain.unitIds.filter(id => (before.unitOrder as string[]).includes(id))');write(old,s);
const many='scripts/learning-many-body-reader-integration.test.ts';s=read(many).replace('assert.equal(learningUnits.length,575)','assert(learningUnits.length>=575)').replace('learningUnits.reduce((n,u)','learningUnits.slice(0,575).reduce((n,u)').replace('new Set(learningUnits.map(u=>u.id)).size,575','new Set(learningUnits.slice(0,575).map(u=>u.id)).size,575');write(many,s);
const qubitAuthoring='scripts/learning-qubit-authoring.test.ts';s=read(qubitAuthoring).replace('assert.equal(learningUnits.length,575);assert.deepEqual(learningUnits.slice(-4).map','assert(learningUnits.length>=575);assert.deepEqual(learningUnits.slice(571,575).map');write(qubitAuthoring,s);
console.log(JSON.stringify({candidate,registeredUnits:584,informationUnits:ids.length}));
