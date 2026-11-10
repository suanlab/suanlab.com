'use client';
import { useEffect, useState } from 'react';
import { readRegularPerturbationInput, type RegularPerturbationInput } from '@/lib/learning-regular-perturbation-input';
export const regularPerturbationStorageKey = 'suanlab-regular-perturbation-v1';
export function parseStoredRegularPerturbation(raw: string): RegularPerturbationInput | null {
  try { if (raw.length > 4096) return null; const data = JSON.parse(raw); return data && typeof data === 'object' && !Array.isArray(data) && Object.keys(data).length === 2 && data.version === 1 ? readRegularPerturbationInput(data.value) : null; } catch { return null; }
}
export function useSavedRegularPerturbation(initial: RegularPerturbationInput) {
  const [input, setInput] = useState(initial), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => { let next = initial; try { const raw = localStorage.getItem(regularPerturbationStorageKey); if (raw !== null) { const parsed = parseStoredRegularPerturbation(raw); if (parsed) next = parsed; else setStorage('corrupt'); } } catch { setStorage('unavailable'); } setInput(next); setReady(true); }, [initial]);
  useEffect(() => { if (!ready) return; try { localStorage.setItem(regularPerturbationStorageKey, JSON.stringify({ version: 1, value: input })); } catch { setStorage('unavailable'); } }, [input, ready]);
  return { input, ready, storage, setInput(value: RegularPerturbationInput) { const next = readRegularPerturbationInput(value); if (next) setInput(next); } };
}
