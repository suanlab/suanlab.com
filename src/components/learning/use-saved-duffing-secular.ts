'use client';
import { useEffect, useState } from 'react';
import { readDuffingSecularInput, type DuffingSecularInput } from '@/lib/learning-duffing-secular-input';
export const duffingSecularStorageKey = 'suanlab-duffing-secular-v1';
export function parseStoredDuffingSecular(raw: string): DuffingSecularInput | null {
  try { if (raw.length > 4096) return null; const data = JSON.parse(raw); return data && typeof data === 'object' && !Array.isArray(data) && Object.keys(data).length === 2 && data.version === 1 ? readDuffingSecularInput(data.value) : null; } catch { return null; }
}
export function useSavedDuffingSecular(initial: DuffingSecularInput) {
  const [input, setInput] = useState(initial), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => { let next = initial; try { const raw = localStorage.getItem(duffingSecularStorageKey); if (raw !== null) { const parsed = parseStoredDuffingSecular(raw); if (parsed) next = parsed; else setStorage('corrupt'); } } catch { setStorage('unavailable'); } setInput(next); setReady(true); }, [initial]);
  useEffect(() => { if (!ready) return; try { localStorage.setItem(duffingSecularStorageKey, JSON.stringify({ version: 1, value: input })); } catch { setStorage('unavailable'); } }, [input, ready]);
  return { input, ready, storage, setInput(value: DuffingSecularInput) { const next = readDuffingSecularInput(value); if (next) setInput(next); } };
}
