'use client';
import {useEffect,useState} from 'react';
import {readGaugeOrbitInput,type GaugeOrbitInput} from '@/lib/learning-gauge-orbits-input';
export const gaugeOrbitStorageKey='suanlab-gauge-orbits-v1';
export function parseStoredGaugeOrbit(raw:string):GaugeOrbitInput|null{try{if(raw.length>65536)return null;const d=JSON.parse(raw);return d&&typeof d==='object'&&!Array.isArray(d)&&Object.keys(d).length===2&&d.version===1?readGaugeOrbitInput(d.value):null;}catch{return null;}}
export function useSavedGaugeOrbit(initial:GaugeOrbitInput){
 const[input,setInput]=useState(initial),[ready,setReady]=useState(false),[storage,setStorage]=useState<'available'|'corrupt'|'unavailable'>('available');
 useEffect(()=>{let next=initial;try{const raw=localStorage.getItem(gaugeOrbitStorageKey);if(raw!==null){const saved=parseStoredGaugeOrbit(raw);if(saved)next=saved;else setStorage('corrupt');}}catch{setStorage('unavailable');}setInput(next);setReady(true);},[initial]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(gaugeOrbitStorageKey,JSON.stringify({version:1,value:input}));}catch{setStorage('unavailable');}},[input,ready]);
 return{input,ready,storage,setInput(value:GaugeOrbitInput){const next=readGaugeOrbitInput(value);if(next)setInput(next);}};
}
