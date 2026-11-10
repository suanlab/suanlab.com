'use client';
import {useLanguage} from '@/components/language-provider';
import type {InformationTeachingId} from '@/lib/learning-information-teaching';
import InformationGraphic from './information-graphic';
export default function InformationCover({kind}:{kind:InformationTeachingId}){const{language}=useLanguage();return <InformationGraphic kind={kind} language={language}/>;}
