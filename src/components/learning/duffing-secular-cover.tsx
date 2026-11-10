'use client';
import {useLanguage} from '@/components/language-provider';
import DuffingSecularGraphic from './duffing-secular-graphic';
export default function DuffingSecularCover(){const{language}=useLanguage();return <DuffingSecularGraphic language={language} compact/>;}
