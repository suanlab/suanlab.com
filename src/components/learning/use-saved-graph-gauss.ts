'use client';
import {useEffect,useState} from 'react';
import {readGraphGaussInput,type GraphGaussInput} from '@/lib/learning-graph-gauss-input';
export const graphGaussStorageKey='suanlab-graph-gauss-v1';
export function parseStoredGraphGauss(raw:string):GraphGaussInput|null{try{if(raw.length>65536)return null;const d=JSON.parse(raw);return d&&typeof d==='object'&&!Array.isArray(d)&&Object.keys(d).length===2&&d.version===1?readGraphGaussInput(d.value):null;}catch{return null;}}
export function useSavedGraphGauss(initial:GraphGaussInput){
 const[input,setInput]=useState(initial),[ready,setReady]=useState(false),[storage,setStorage]=useState<'available'|'corrupt'|'unavailable'>('available');
 useEffect(()=>{let next=initial;try{const raw=localStorage.getItem(graphGaussStorageKey);if(raw!==null){const saved=parseStoredGraphGauss(raw);if(saved)next=saved;else setStorage('corrupt');}}catch{setStorage('unavailable');}setInput(next);setReady(true);},[initial]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(graphGaussStorageKey,JSON.stringify({version:1,value:input}));}catch{setStorage('unavailable');}},[input,ready]);
 return{input,ready,storage,setInput(value:GraphGaussInput){const next=readGraphGaussInput(value);if(next)setInput(next);}};
}
