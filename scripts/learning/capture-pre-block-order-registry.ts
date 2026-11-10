/** One-time baseline. Never overwrite a historical preservation fixture. */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { learningUnits, learningGuides } from '../../src/data/interactive-book';
import { curriculumDomains } from '../../src/data/learning-curriculum';
const target='scripts/fixtures/pre-block-order-registry-hashes.json';
assert(!existsSync(target),'Baseline already exists');assert.equal(learningUnits.length,602);
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex'),fileHash=(p:string)=>createHash('sha256').update(readFileSync(p)).digest('hex');
const manuscripts=Object.fromEntries(learningUnits.flatMap(u=>['ko','en'].map(language=>{const p=`content/learning/physics-math-for-ai/${u.id}.${language}.md`;return[p,fileHash(p)];})));
const paths=['src/data/navigation.ts','src/app/book/online/page.tsx','src/app/book/interactive/page.tsx','src/components/learning/gaussian-field-matrix-engine.ts','src/components/learning/gaussian-field-matrix-plot.tsx','src/components/learning/scalar-field-plot-engine.ts','src/components/learning/use-saved-state.ts','src/components/language-provider.tsx','src/lib/learning-content.ts'];
writeFileSync(target,JSON.stringify({baselineCommit:'8c88bff',unitOrder:learningUnits.map(u=>u.id),guideOrder:Object.keys(learningGuides),units:Object.fromEntries(learningUnits.map(u=>[u.id,hash(u)])),guides:Object.fromEntries(learningUnits.map(u=>[u.id,hash(learningGuides[u.id])])),domains:Object.fromEntries(curriculumDomains.map(d=>[d.id,hash(d)])),manuscripts,preservedFiles:Object.fromEntries(paths.map(p=>[p,fileHash(p)]))},null,2)+'\n');
console.log('Captured602units and1204manuscripts');
