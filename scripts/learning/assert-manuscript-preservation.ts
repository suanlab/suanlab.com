import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import reviewedManuscripts from '../fixtures/reviewed-learning-manuscripts.json';
/** Explicit wording corrections made during the path-integral topic review.
 * Historical fixtures are immutable. Reversing ONLY these replacements must
 * recover their exact hash; every other byte still has to match the baseline.
 */
export const pathIntegralEditorialCorrections:Record<string,readonly(readonly[string,string])[]>={
 'content/learning/physics-math-for-ai/euclidean-quantum-kernels-and-time-slicing.en.md':[
  ['The planned exact calculator accepts','The exact calculator accepts'],
  ['Only the input range through32 intervals is planned for full matrices.','Full matrices are computed for the input range through32 intervals.'],
 ],
 'content/learning/physics-math-for-ai/euclidean-quantum-kernels-and-time-slicing.ko.md':[
  ['계획한 정확분수 계산기는','정확분수 계산기는'],
  ['전체 행렬을 계산할 계획 범위는 입력32구간까지다.','전체 행렬은 입력32구간까지 계산한다.'],
 ],
};
const hash=(raw:string)=>createHash('sha256').update(raw).digest('hex');
export function assertPreservedLearningFile(path:string,expected:string){
 assertPreservedLearningSource(path,readFileSync(path,'utf8'),expected);
}
/** A reviewed revision is accepted only for its exact old and new hashes.
 * The historical fixture and protection for all other files stay unchanged.
 */
export function assertPreservedLearningSource(path:string,raw:string,expected:string){
 const actual=hash(raw);if(actual===expected)return;
 const revision=(reviewedManuscripts as Record<string,{previousHashes:string[];reviewedHash:string;review:string}>)[path];
 if(revision){
  assert(revision.previousHashes.includes(expected),'Unreviewed baseline for manuscript: '+path);
  assert.equal(actual,revision.reviewedHash,'Changes exceed the reviewed manuscript revision: '+path);
  return;
 }

 const edits=pathIntegralEditorialCorrections[path];assert(edits,'Unexpected change to preserved file: '+path);let original=raw;
 for(const[before,after]of edits){assert.equal(original.split(after).length-1,1,'Editorial replacement must occur exactly once: '+path);original=original.replace(after,before);}
 assert.equal(hash(original),expected,'Changes exceed the declared editorial correction: '+path);
}
