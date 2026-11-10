'use client';
import {useLanguage} from '@/components/language-provider';
import FieldBalanceGraphic from './field-balance-graphic';
export default function FieldBalanceCover(){const{language}=useLanguage();return <FieldBalanceGraphic language={language} compact/>;}
