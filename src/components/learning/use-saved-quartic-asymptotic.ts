'use client';
import { useEffect, useState } from 'react';
import { readQuarticAsymptoticInput, type QuarticAsymptoticInput } from '@/lib/learning-quartic-asymptotic-input';
export const quarticAsymptoticStorageKey = 'suanlab-quartic-asymptotic-v1';
export function parseStoredQuarticAsymptotic(raw: string): QuarticAsymptoticInput | null {
  try { if (raw.length > 4096) return null; const data = JSON.parse(raw); return data && typeof data === 'object' && !Array.isArray(data) && Object.keys(data).length === 2 && data.version === 1 ? readQuarticAsymptoticInput(data.value) : null; } catch { return null; }
}
export function useSavedQuarticAsymptotic(initial: QuarticAsymptoticInput) {
  const [input, setInput] = useState(initial), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => { let next = initial; try { const raw = localStorage.getItem(quarticAsymptoticStorageKey); if (raw !== null) { const parsed = parseStoredQuarticAsymptotic(raw); if (parsed) next = parsed; else setStorage('corrupt'); } } catch { setStorage('unavailable'); } setInput(next); setReady(true); }, [initial]);
  useEffect(() => { if (!ready) return; try { localStorage.setItem(quarticAsymptoticStorageKey, JSON.stringify({ version: 1, value: input })); } catch { setStorage('unavailable'); } }, [input, ready]);
  return { input, ready, storage, setInput(value: QuarticAsymptoticInput) { const next = readQuarticAsymptoticInput(value); if (next) setInput(next); } };
}
