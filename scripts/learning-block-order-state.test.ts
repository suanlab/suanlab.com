import assert from 'node:assert/strict';
import { test } from 'node:test';
import { blockOrderDefault, blockOrderExamples } from '../src/lib/learning-block-order-input';
import { blockOrderDraft } from '../src/lib/learning-block-order-view';
import { emptyBlockOrderState, readBlockOrderState, parseStoredBlockOrderState } from '../src/lib/learning-block-order-state';
const limits = { steps: 6,guides: 6,questions: 8,options: 3 };
test('saved state preserves unfinished drafts separately from the immutable applied model and omits run state and responses', () => {
  const value = structuredClone(emptyBlockOrderState(limits)); value.draft.disagreementActivity = '1/'; value.draft.selectedUpCounts = 'unfinished';
  value.answers[3] = 1; value.step = 5; value.display.mode = 'conditional'; value.display.flat = false;
  const raw = JSON.stringify({ version: 1,model: value }), read = parseStoredBlockOrderState(raw,limits)!;
  assert(read); assert.deepEqual(read.input,blockOrderDefault); assert.equal(read.draft.disagreementActivity,'1/'); assert.equal(read.draft.selectedUpCounts,'unfinished');
  assert.equal(read.answers[3],1); assert.equal(read.display.mode,'conditional'); assert.equal(read.display.flat,false);
  assert(Object.isFrozen(read) && Object.isFrozen(read.input) && Object.isFrozen(read.answers) && Object.isFrozen(read.display));
  assert(!raw.includes('response') && !raw.includes('running') && !raw.includes('playing'));
  value.input.shape[0] = 8; assert.deepEqual(read.input.shape,[4]);
});
test('saved inputs, choices and pages use bounded whitelists without invoking getters or accepting extra run flags', () => {
  const base = emptyBlockOrderState(limits);
  for (const patch of [{ step: 6 },{ step: -0 },{ guide: 1.5 },{ motion: 0 },{ explain: 'yes' },{ input: {} },{ response: {} },{ playing: true },{ answers: [0] },{ answers: Array(8).fill(3) },{ display: { ...base.display,page: 1 } },{ draft: { ...base.draft,shape: 'x'.repeat(13) } }]) assert.equal(readBlockOrderState({ ...base,...patch },limits),null);
  let calls = 0; assert.equal(readBlockOrderState({ ...base,get input() { calls++; return base.input; } },limits),null); assert.equal(calls,0);
  for (const raw of ['{','null',JSON.stringify({ version: 2,model: base }),JSON.stringify({ version: 1,model: base,extra: true }),'x'.repeat(8001)]) assert.equal(parseStoredBlockOrderState(raw,limits),null);
  const large = structuredClone(base), input = blockOrderExamples.find(e => e.id === 'two-dimensional')!.input;
  large.input = input; large.draft = blockOrderDraft(input); large.display.page = 24;
  assert(readBlockOrderState(large,limits)); large.display.page = 25; assert.equal(readBlockOrderState(large,limits),null);
});
