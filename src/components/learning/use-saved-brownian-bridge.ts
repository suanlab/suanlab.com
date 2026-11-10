'use client';
import { useEffect, useState } from 'react';
import { readBrownianBridgeInput, type BrownianBridgeInput } from '@/lib/learning-brownian-bridge-input';
import { brownianBridgeDefaultDisplay, brownianBridgeDefaultLearning, brownianBridgeMatrixSize, readBrownianBridgeDisplay, readBrownianBridgeLearning, type BrownianBridgeDisplay, type BrownianBridgeLearning } from '@/lib/learning-brownian-bridge-view';
export const brownianBridgeStorageKey = 'suanlab-brownian-bridge-v1';
type Saved = { input: Readonly<BrownianBridgeInput>; display: Readonly<BrownianBridgeDisplay>; learning: Readonly<BrownianBridgeLearning> };
const initialDisplay = (input: BrownianBridgeInput) => ({ ...brownianBridgeDefaultDisplay, selectedTime: input.probe, selectedCell: input.probe * (input.intervals + 2) });
export function parseStoredBrownianBridge(raw: string): Saved | null {
  try {
    if (raw.length > 4096) return null;
    const d = JSON.parse(raw);
    if (!d || typeof d !== 'object' || Array.isArray(d) || Object.keys(d).length !== 4 || d.version !== 1) return null;
    const input = readBrownianBridgeInput(d.input); if (!input) return null;
    const display = readBrownianBridgeDisplay(d.display, input), learning = readBrownianBridgeLearning(d.learning); return display && learning ? { input, display, learning } : null;
  } catch { return null; }
}
export function useSavedBrownianBridge(initial: BrownianBridgeInput) {
  const [model, setModel] = useState<Saved>({ input: initial, display: initialDisplay(initial), learning: brownianBridgeDefaultLearning });
  const [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => {
    let next: Saved = { input: initial, display: initialDisplay(initial), learning: brownianBridgeDefaultLearning };
    try { const raw = localStorage.getItem(brownianBridgeStorageKey); if (raw !== null) { const parsed = parseStoredBrownianBridge(raw); if (parsed) next = parsed; else setStorage('corrupt'); } }
    catch { setStorage('unavailable'); }
    setModel(next); setReady(true);
  }, [initial]);
  useEffect(() => { if (ready) try { localStorage.setItem(brownianBridgeStorageKey, JSON.stringify({ version: 1, ...model })); } catch { setStorage('unavailable'); } }, [model, ready]);
  return { ...model, ready, storage,
    setInput(value: BrownianBridgeInput) {
      const input = readBrownianBridgeInput(value); if (!input) return;
      setModel(previous => { const n = brownianBridgeMatrixSize(input, previous.display.kind); return { ...previous, input, display: { ...previous.display, selectedTime: Math.min(input.intervals, previous.display.selectedTime), selectedCell: Math.min(n * n - 1, previous.display.selectedCell) } }; });
    },
    setDisplay(value: BrownianBridgeDisplay) { setModel(previous => { const display = readBrownianBridgeDisplay(value, previous.input); return display ? { ...previous, display } : previous; }); },
    setLearning(value: BrownianBridgeLearning) { const learning = readBrownianBridgeLearning(value); if (learning) setModel(previous => ({ ...previous, learning })); },
  };
}
