'use client';
import {useEffect,useState} from 'react';
import {readGraphGaugeInput,type GraphGaugeInput} from '@/lib/learning-graph-gauge-input';
export const graphGaugeStorageKey='suanlab-graph-gauge-v1';
export function parseStoredGraphGauge(raw:string):GraphGaugeInput|null{try{if(raw.length>65536)return null;const d=JSON.parse(raw);return d&&typeof d==='object'&&!Array.isArray(d)&&Object.keys(d).length===2&&d.version===1?readGraphGaugeInput(d.value):null;}catch{return null;}}
export function useSavedGraphGauge(initial:GraphGaugeInput){
 const[input,setInput]=useState(initial),[ready,setReady]=useState(false),[storage,setStorage]=useState<'available'|'corrupt'|'unavailable'>('available');
 useEffect(()=>{let next=initial;try{const raw=localStorage.getItem(graphGaugeStorageKey);if(raw!==null){const saved=parseStoredGraphGauge(raw);if(saved)next=saved;else setStorage('corrupt');}}catch{setStorage('unavailable');}setInput(next);setReady(true);},[initial]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(graphGaugeStorageKey,JSON.stringify({version:1,value:input}));}catch{setStorage('unavailable');}},[input,ready]);
 return{input,ready,storage,setInput(value:GraphGaugeInput){const next=readGraphGaugeInput(value);if(next)setInput(next);}};
}
