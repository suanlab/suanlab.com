'use client';
import { useEffect, useState } from 'react';
import { readSpatialSpectrumInput, type SpatialSpectrumInput } from '@/lib/learning-spatial-spectrum-input';
export const spatialSpectrumStorageKey = 'suanlab-spatial-spectrum-v1';
export function parseStoredSpatialSpectrum(raw: string): SpatialSpectrumInput | null {
  try { if (raw.length > 4096) return null; const data = JSON.parse(raw); return data && typeof data === 'object' && !Array.isArray(data) && Object.keys(data).length === 2 && data.version === 1 ? readSpatialSpectrumInput(data.value) : null; } catch { return null; }
}
export function useSavedSpatialSpectrum(initial: SpatialSpectrumInput) {
  const [input, setInput] = useState(initial), [ready, setReady] = useState(false), [storage, setStorage] = useState<'available' | 'corrupt' | 'unavailable'>('available');
  useEffect(() => { let next = initial; try { const raw = localStorage.getItem(spatialSpectrumStorageKey); if (raw !== null) { const parsed = parseStoredSpatialSpectrum(raw); if (parsed) next = parsed; else setStorage('corrupt'); } } catch { setStorage('unavailable'); } setInput(next); setReady(true); }, [initial]);
  useEffect(() => { if (!ready) return; try { localStorage.setItem(spatialSpectrumStorageKey, JSON.stringify({ version: 1, value: input })); } catch { setStorage('unavailable'); } }, [input, ready]);
  return { input, ready, storage, setInput(value: SpatialSpectrumInput) { const next = readSpatialSpectrumInput(value); if (next) setInput(next); } };
}
