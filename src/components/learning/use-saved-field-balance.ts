'use client';
import {useEffect,useState} from 'react';
import {readFieldBalanceInput,type FieldBalanceInput} from '@/lib/learning-field-balance-input';
export const fieldBalanceStorageKey='suanlab-field-balance-v1';
export function parseStoredFieldBalance(raw:string):FieldBalanceInput|null{try{if(raw.length>16000)return null;const d=JSON.parse(raw);return d&&typeof d==='object'&&!Array.isArray(d)&&Object.keys(d).length===2&&d.version===1?readFieldBalanceInput(d.value):null;}catch{return null;}}
export function useSavedFieldBalance(initial:FieldBalanceInput){
 const[input,setInput]=useState(initial),[ready,setReady]=useState(false),[storage,setStorage]=useState<'available'|'corrupt'|'unavailable'>('available');
 useEffect(()=>{let next=initial;try{const raw=localStorage.getItem(fieldBalanceStorageKey);if(raw!==null){const saved=parseStoredFieldBalance(raw);if(saved)next=saved;else setStorage('corrupt');}}catch{setStorage('unavailable');}setInput(next);setReady(true);},[initial]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(fieldBalanceStorageKey,JSON.stringify({version:1,value:input}));}catch{setStorage('unavailable');}},[input,ready]);
 return{input,ready,storage,setInput(value:FieldBalanceInput){const next=readFieldBalanceInput(value);if(next)setInput(next);}};
}
