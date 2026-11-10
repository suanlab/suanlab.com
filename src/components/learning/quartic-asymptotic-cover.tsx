'use client';
import {useLanguage} from '@/components/language-provider';
import QuarticAsymptoticGraphic from './quartic-asymptotic-graphic';
export default function QuarticAsymptoticCover(){const{language}=useLanguage();return <QuarticAsymptoticGraphic language={language} compact/>;}
