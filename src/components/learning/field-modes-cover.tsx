'use client';
import {useLanguage} from '@/components/language-provider';
import FieldModesGraphic from './field-modes-graphic';
export default function FieldModesCover(){const{language}=useLanguage();return <FieldModesGraphic language={language} compact/>;}
