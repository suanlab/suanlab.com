'use client';
import { useEffect, useState } from 'react';
import { blockOrderSteps, blockOrderExperiments, blockOrderQuestions } from '@/data/block-order-teaching';
import { emptyBlockOrderState, parseStoredBlockOrderState, readBlockOrderState, blockOrderStorageKey, type BlockOrderSavedState } from '@/lib/learning-block-order-state';
export const blockOrderLearningLimits = { steps: blockOrderSteps.length, guides: blockOrderExperiments.length, questions: blockOrderQuestions.length, options: 3 };
export function useSavedBlockOrder() {
  const [model, setModel] = useState(() => emptyBlockOrderState(blockOrderLearningLimits)), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => {
    let next = emptyBlockOrderState(blockOrderLearningLimits);
    try { const raw = localStorage.getItem(blockOrderStorageKey); if (raw !== null) { const read = parseStoredBlockOrderState(raw, blockOrderLearningLimits); if (read) next = read; else setStorage('corrupt'); } } catch { setStorage('unavailable'); }
    setModel(next); setReady(true);
  }, []);
  useEffect(() => { if (ready) try { localStorage.setItem(blockOrderStorageKey, JSON.stringify({ version: 1, model })); } catch { setStorage('unavailable'); } }, [ready, model]);
  const update = (change: (previous: BlockOrderSavedState) => BlockOrderSavedState) => { if (ready) setModel(p => readBlockOrderState(change(p), blockOrderLearningLimits) ?? p); };
  return { model, ready, storage, update };
}
