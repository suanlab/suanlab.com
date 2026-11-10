'use client';
import { useEffect,useRef,useState } from 'react';
import { parseManyBodyStoredInput,readManyBodyInput,snapshotManyBodyInput,type ManyBodyInput } from '@/lib/learning-many-body-input';
/** The supplied initial input is owned once; saved data is checked by kind. */
export function useSavedManyBodyInput(initial:ManyBodyInput){
 const initialRef=useRef(initial);initialRef.current=initial;
 const kind=initial.kind,key=`suanlab-many-body-${kind}`,[state,setState]=useState(()=>({kind,input:snapshotManyBodyInput(initial),ready:false,storage:true}));
 useEffect(()=>{let input=snapshotManyBodyInput(initialRef.current),storage=true;try{const raw=localStorage.getItem(key),saved=raw?parseManyBodyStoredInput(raw):null;if(saved?.kind===kind)input=saved;}catch{storage=false;}setState({kind,input,ready:true,storage});
 },[key,kind]);
 useEffect(()=>{if(!state.ready||state.kind!==kind)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotManyBodyInput(state.input)}));}catch{setState(s=>s.storage?{...s,storage:false}:s);}},[key,kind,state.kind,state.ready,state.input]);
 return{input:state.kind===kind?state.input:initial,ready:state.kind===kind&&state.ready,storage:state.storage,setInput:(next:ManyBodyInput)=>{const owned=readManyBodyInput(next);if(owned?.kind===kind)setState(s=>s.kind===kind?{...s,input:owned}:s);}};
}
