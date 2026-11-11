import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync,existsSync} from 'node:fs';
import reviewed from './fixtures/reviewed-learning-manuscripts.json';
import {assertPreservedLearningFile,assertPreservedLearningSource,pathIntegralEditorialCorrections} from './learning/assert-manuscript-preservation';

for(const name of ['saddle-fluctuation','real-time-phase','thermal-field','block-order']){
 test(`historical ${name} corpus accepts only unchanged or explicitly reviewed files`,()=>{
  const fixture=JSON.parse(readFileSync(`scripts/fixtures/pre-${name}-registry-hashes.json`,'utf8')) as {manuscripts:Record<string,string>;preservedFiles:Record<string,string>};
  for(const [path,expected] of Object.entries({...fixture.manuscripts,...fixture.preservedFiles}))assertPreservedLearningFile(path,expected);
 });
}
test('every reviewed revision has evidence and rejects an additional content mutation',()=>{
 for(const [path,revision] of Object.entries(reviewed)){
  assert(existsSync(revision.review),path);
  const raw=readFileSync(path,'utf8');
  for(const previous of revision.previousHashes){
   assertPreservedLearningSource(path,raw,previous);
   assert.throws(()=>assertPreservedLearningSource(path,raw+'\nUnreviewed edit.\n',previous),/Changes exceed the reviewed manuscript revision/);
  }
  assert.throws(()=>assertPreservedLearningSource(path,raw,'0'.repeat(64)),/Unreviewed baseline/);
 }
});
test('an unknown preserved file cannot use a reviewed revision as a blanket exception',()=>{
 const [path,revision]=Object.entries(reviewed)[0];
 assert.throws(()=>assertPreservedLearningSource('content/learning/physics-math-for-ai/unreviewed.en.md',readFileSync(path,'utf8'),revision.previousHashes[0]),/Unexpected change to preserved file/);
});
test('reversible editorial corrections still require exact surrounding content',()=>{
 const fixture=JSON.parse(readFileSync('scripts/fixtures/pre-saddle-fluctuation-registry-hashes.json','utf8')) as {manuscripts:Record<string,string>};
 for(const path of Object.keys(pathIntegralEditorialCorrections)){
  const raw=readFileSync(path,'utf8');
  assertPreservedLearningSource(path,raw,fixture.manuscripts[path]);
  assert.throws(()=>assertPreservedLearningSource(path,raw+'\nUnreviewed edit.\n',fixture.manuscripts[path]),/Changes exceed the declared editorial correction/);
 }
});
