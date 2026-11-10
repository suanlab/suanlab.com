'use client';
import { useEffect, useState } from 'react';
import { readEuclideanKernelInput, type EuclideanKernelInput } from '@/lib/learning-euclidean-kernel-input';
import { euclideanKernelDefaultDisplay, euclideanKernelDefaultLearning, euclideanKernelAlignDisplay, readEuclideanKernelDisplay, readEuclideanKernelLearning, type EuclideanKernelDisplay, type EuclideanKernelLearning } from '@/lib/learning-euclidean-kernel-view';
export const euclideanKernelStorageKey = 'suanlab-euclidean-kernel-v1';
type Saved = { input: Readonly<EuclideanKernelInput>; display: Readonly<EuclideanKernelDisplay>; learning: Readonly<EuclideanKernelLearning> };
const initialDisplay = (input: EuclideanKernelInput) => euclideanKernelAlignDisplay(input,euclideanKernelDefaultDisplay);
export function parseStoredEuclideanKernel(raw: string): Saved | null {
  try {
    if (raw.length > 4096) return null;
    const d = JSON.parse(raw);
    if (!d || typeof d !== 'object' || Array.isArray(d) || Object.keys(d).length !== 4 || d.version !== 1) return null;
    const input = readEuclideanKernelInput(d.input); if (!input) return null;
    const display = readEuclideanKernelDisplay(d.display, input), learning = readEuclideanKernelLearning(d.learning); return display && learning ? { input, display, learning } : null;
  } catch { return null; }
}
export function useSavedEuclideanKernel(initial: EuclideanKernelInput) {
  const [model, setModel] = useState<Saved>({ input: initial, display: initialDisplay(initial), learning: euclideanKernelDefaultLearning });
  const [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => {
    let next: Saved = { input: initial, display: initialDisplay(initial), learning: euclideanKernelDefaultLearning };
    try { const raw = localStorage.getItem(euclideanKernelStorageKey); if (raw !== null) { const parsed = parseStoredEuclideanKernel(raw); if (parsed) next = parsed; else setStorage('corrupt'); } }
    catch { setStorage('unavailable'); }
    setModel(next); setReady(true);
  }, [initial]);
  useEffect(() => { if (ready) try { localStorage.setItem(euclideanKernelStorageKey, JSON.stringify({ version: 1, ...model })); } catch { setStorage('unavailable'); } }, [model, ready]);
  return { ...model, ready, storage,
    setInput(value: EuclideanKernelInput) {
      const input = readEuclideanKernelInput(value); if (!input) return;
      setModel(previous => ({...previous,input,display:euclideanKernelAlignDisplay(input,previous.display)}));
    },
    setDisplay(value: EuclideanKernelDisplay) { setModel(previous => { const display = readEuclideanKernelDisplay(value, previous.input); return display ? { ...previous, display } : previous; }); },
    setLearning(value: EuclideanKernelLearning) { const learning = readEuclideanKernelLearning(value); if (learning) setModel(previous => ({ ...previous, learning })); },
  };
}
