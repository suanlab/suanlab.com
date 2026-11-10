'use client';
import {useEffect,useState} from 'react';
import {readQuantumInformationInput,parseQuantumInformationStoredInput,type QuantumInformationInput,type QuantumInformationKind} from '@/lib/learning-quantum-information-input';
export function useSavedInformationInput<K extends QuantumInformationKind>(kind:K,initial:Extract<QuantumInformationInput,{kind:K}>){
 const key=`suanlab-information-v1-${kind}`;
 const[input,setInput]=useState(initial),[ready,setReady]=useState(false),[storage,setStorage]=useState<'available'|'corrupt'|'unavailable'>('available');
 useEffect(()=>{let next=initial;try{const raw=localStorage.getItem(key);if(raw!==null){const saved=parseQuantumInformationStoredInput(raw);if(saved?.kind===kind)next=saved as typeof initial;else setStorage('corrupt');}}catch{setStorage('unavailable');}setInput(next);setReady(true);},[initial,key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:input}));}catch{setStorage('unavailable');}},[input,key,ready]);
 return{input,ready,storage,setInput(next:Extract<QuantumInformationInput,{kind:K}>){const owned=readQuantumInformationInput(next);if(owned?.kind===kind)setInput(owned as typeof initial);}};
}
