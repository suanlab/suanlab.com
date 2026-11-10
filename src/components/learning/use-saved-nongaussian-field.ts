'use client';
import { useEffect, useState } from 'react';
import { readNongaussianFieldInput, type NongaussianFieldInput } from '@/lib/learning-nongaussian-field-input';
export const nongaussianFieldStorageKey = 'suanlab-nongaussian-field-v1';
export function parseStoredNongaussianField(raw: string): NongaussianFieldInput | null {
  try { if (raw.length > 4096) return null; const data = JSON.parse(raw); return data && typeof data === 'object' && !Array.isArray(data) && Object.keys(data).length === 2 && data.version === 1 ? readNongaussianFieldInput(data.value) : null; } catch { return null; }
}
export function useSavedNongaussianField(initial: NongaussianFieldInput) {
  const [input, setInput] = useState(initial), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => { let next = initial; try { const raw = localStorage.getItem(nongaussianFieldStorageKey); if (raw !== null) { const parsed = parseStoredNongaussianField(raw); if (parsed) next = parsed; else setStorage('corrupt'); } } catch { setStorage('unavailable'); } setInput(next); setReady(true); }, [initial]);
  useEffect(() => { if (!ready) return; try { localStorage.setItem(nongaussianFieldStorageKey, JSON.stringify({ version: 1, value: input })); } catch { setStorage('unavailable'); } }, [input, ready]);
  return { input, ready, storage, setInput(value: NongaussianFieldInput) { const next = readNongaussianFieldInput(value); if (next) setInput(next); } };
}
