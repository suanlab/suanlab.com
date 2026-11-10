'use client';
import {useLanguage} from '@/components/language-provider';
import type {OptimizationResult} from '@/lib/learning-optimization';
import MeasureFigure from './measure-figure';
export default function OptimizationFigure({result,label,axis}:{result:OptimizationResult;label:string;axis:string}){
 const {language}=useLanguage(),ko=language==='ko';
 return <><MeasureFigure result={result} label={label} axis={axis} variant="optimization"/>{result.numericNote&&<p className="ai-model-note" role="status">{ko?'수학적으로 유한한 민감도가 숫자 표현 범위를 벗어나 해당 수치를 숨겼습니다. Jacobian이 0인 특이점과는 다른 경우입니다.':'A mathematically finite sensitivity exceeds numeric representation range and its readout is hidden. This differs from a singular zero Jacobian.'}</p>}</>;
}
