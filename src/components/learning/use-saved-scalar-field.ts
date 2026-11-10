'use client';
import {useEffect,useState} from 'react';
import {readScalarFieldInput,type ScalarFieldInput} from '@/lib/learning-scalar-field-input';
export const scalarFieldStorageKey='suanlab-scalar-field-v1';
export function parseStoredScalarField(raw:string):ScalarFieldInput|null{try{if(raw.length>16000)return null;const d=JSON.parse(raw);return d&&typeof d==='object'&&!Array.isArray(d)&&Object.keys(d).length===2&&d.version===1?readScalarFieldInput(d.value):null;}catch{return null;}}
export function useSavedScalarField(initial:ScalarFieldInput){
 const[input,setInput]=useState(initial),[ready,setReady]=useState(false),[storage,setStorage]=useState<'available'|'corrupt'|'unavailable'>('available');
 useEffect(()=>{let next=initial;try{const raw=localStorage.getItem(scalarFieldStorageKey);if(raw!==null){const saved=parseStoredScalarField(raw);if(saved)next=saved;else setStorage('corrupt');}}catch{setStorage('unavailable');}setInput(next);setReady(true);},[initial]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(scalarFieldStorageKey,JSON.stringify({version:1,value:input}));}catch{setStorage('unavailable');}},[input,ready]);
 return{input,ready,storage,setInput(value:ScalarFieldInput){const next=readScalarFieldInput(value);if(next)setInput(next);}};
}
