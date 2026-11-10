'use client';
import {useEffect,useState} from 'react';
import {readFieldModesInput,type FieldModesInput} from '@/lib/learning-field-modes-input';
export const fieldModesStorageKey='suanlab-field-modes-v1';
export function parseStoredFieldModes(raw:string):FieldModesInput|null{try{if(raw.length>16000)return null;const d=JSON.parse(raw);return d&&typeof d==='object'&&!Array.isArray(d)&&Object.keys(d).length===2&&d.version===1?readFieldModesInput(d.value):null;}catch{return null;}}
export function useSavedFieldModes(initial:FieldModesInput){
 const[input,setInput]=useState(initial),[ready,setReady]=useState(false),[storage,setStorage]=useState<'available'|'corrupt'|'unavailable'>('available');
 useEffect(()=>{let next=initial;try{const raw=localStorage.getItem(fieldModesStorageKey);if(raw!==null){const saved=parseStoredFieldModes(raw);if(saved)next=saved;else setStorage('corrupt');}}catch{setStorage('unavailable');}setInput(next);setReady(true);},[initial]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(fieldModesStorageKey,JSON.stringify({version:1,value:input}));}catch{setStorage('unavailable');}},[input,ready]);
 return{input,ready,storage,setInput(value:FieldModesInput){const next=readFieldModesInput(value);if(next)setInput(next);}};
}
