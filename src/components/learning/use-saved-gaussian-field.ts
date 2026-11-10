'use client';
import { useEffect, useState } from 'react';
import { readGaussianFieldInput, type GaussianFieldInput } from '@/lib/learning-gaussian-field-input';
export const gaussianFieldStorageKey = 'suanlab-gaussian-field-v1';
export function parseStoredGaussianField(raw: string): GaussianFieldInput | null {
  try { if (raw.length > 4096) return null; const data = JSON.parse(raw); return data && typeof data === 'object' && !Array.isArray(data) && Object.keys(data).length === 2 && data.version === 1 ? readGaussianFieldInput(data.value) : null; } catch { return null; }
}
export function useSavedGaussianField(initial: GaussianFieldInput) {
  const [input, setInput] = useState(initial), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => { let next = initial; try { const raw = localStorage.getItem(gaussianFieldStorageKey); if (raw !== null) { const parsed = parseStoredGaussianField(raw); if (parsed) next = parsed; else setStorage('corrupt'); } } catch { setStorage('unavailable'); } setInput(next); setReady(true); }, [initial]);
  useEffect(() => { if (!ready) return; try { localStorage.setItem(gaussianFieldStorageKey, JSON.stringify({ version: 1, value: input })); } catch { setStorage('unavailable'); } }, [input, ready]);
  return { input, ready, storage, setInput(value: GaussianFieldInput) { const next = readGaussianFieldInput(value); if (next) setInput(next); } };
}
