'use client';
import React, {useEffect,useId,useState} from 'react';
import {useLanguage} from '@/components/language-provider';

import {parseNumberDraft} from '@/lib/learning-number-input';
export const parseThermodynamicDraft = parseNumberDraft;
export default function ThermodynamicNumberField({label,value,min,max,step='any',integer=false,onChange}:{label:string;value:number;min:number;max:number;step?:number|'any';integer?:boolean;onChange:(value:number)=>void}) {
 const {language}=useLanguage(),id=useId(),[draft,setDraft]=useState(String(value)),[error,setError]=useState(false);
 useEffect(()=>{setDraft(String(value));setError(false);},[value]);
 const commit=()=>{const next=parseThermodynamicDraft(draft,min,max,integer);setError(next===null);if(next!==null){setDraft(String(next));onChange(next);}};
 return <label className="ai-number-field"><span id={`${id}-label`}>{label} · {min}–{max}</span><input type="number" aria-labelledby={`${id}-label`} value={draft} min={min} max={max} step={step} aria-invalid={error} aria-describedby={error?`${id}-error`:undefined} onChange={e=>{setDraft(e.target.value);setError(false);}} onBlur={commit} onKeyDown={e=>{if(e.key==='Enter'){e.preventDefault();e.currentTarget.blur();}}}/>{error&&<span id={`${id}-error`} role="alert">{language==='ko'?` ${min}부터 ${max}까지의 ${integer?'정수':'유한한 수'}를 입력하세요. 표현할 수 없는 작은 비영 값은 0으로 적용하지 않습니다. 이전 설정을 유지합니다.`:`Enter a ${integer?'whole':'finite'} number from ${min} to ${max}. A nonzero value outside the number range is not applied as zero. The previous setting is retained.`}</span>}</label>;
}
