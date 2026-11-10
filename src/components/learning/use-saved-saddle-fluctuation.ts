'use client';
import { useEffect, useState } from 'react';
import { readSaddleFluctuationInput, type SaddleFluctuationInput } from '@/lib/learning-saddle-fluctuation-input';
import { saddleFluctuationDefaultDisplay, saddleFluctuationDefaultLearning, saddleFluctuationAlignDisplay, readSaddleFluctuationDisplay, readSaddleFluctuationLearning, type SaddleFluctuationDisplay, type SaddleFluctuationLearning } from '@/lib/learning-saddle-fluctuation-view';
export const saddleFluctuationStorageKey = 'suanlab-saddle-fluctuation-v1';
type Saved = { input: Readonly<SaddleFluctuationInput>; display: Readonly<SaddleFluctuationDisplay>; learning: Readonly<SaddleFluctuationLearning> };
const initialDisplay = (input: SaddleFluctuationInput) => saddleFluctuationAlignDisplay(input,saddleFluctuationDefaultDisplay);
export function parseStoredSaddleFluctuation(raw: string): Saved | null {
  try {
    if (raw.length > 4096) return null;
    const d = JSON.parse(raw);
    if (!d || typeof d !== 'object' || Array.isArray(d) || Object.keys(d).length !== 4 || d.version !== 1) return null;
    const input = readSaddleFluctuationInput(d.input); if (!input) return null;
    const display = readSaddleFluctuationDisplay(d.display, input), learning = readSaddleFluctuationLearning(d.learning); return display && learning ? { input, display, learning } : null;
  } catch { return null; }
}
export function useSavedSaddleFluctuation(initial: SaddleFluctuationInput) {
  const [model, setModel] = useState<Saved>({ input: initial, display: initialDisplay(initial), learning: saddleFluctuationDefaultLearning });
  const [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => {
    let next: Saved = { input: initial, display: initialDisplay(initial), learning: saddleFluctuationDefaultLearning };
    try { const raw = localStorage.getItem(saddleFluctuationStorageKey); if (raw !== null) { const parsed = parseStoredSaddleFluctuation(raw); if (parsed) next = parsed; else setStorage('corrupt'); } }
    catch { setStorage('unavailable'); }
    setModel(next); setReady(true);
  }, [initial]);
  useEffect(() => { if (ready) try { localStorage.setItem(saddleFluctuationStorageKey, JSON.stringify({ version: 1, ...model })); } catch { setStorage('unavailable'); } }, [model, ready]);
  return { ...model, ready, storage,
    setInput(value: SaddleFluctuationInput) {
      const input = readSaddleFluctuationInput(value); if (!input) return;
      setModel(previous => ({...previous,input,display:saddleFluctuationAlignDisplay(input,previous.display)}));
    },
    setDisplay(value: SaddleFluctuationDisplay) { setModel(previous => { const display = readSaddleFluctuationDisplay(value, previous.input); return display ? { ...previous, display } : previous; }); },
    setLearning(value: SaddleFluctuationLearning) { const learning = readSaddleFluctuationLearning(value); if (learning) setModel(previous => ({ ...previous, learning })); },
  };
}
