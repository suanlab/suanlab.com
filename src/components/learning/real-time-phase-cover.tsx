'use client';
import React from 'react';
import {useLanguage} from '@/components/language-provider';
import RealTimePhaseGraphic from './real-time-phase-graphic';
export default function RealTimePhaseCover(){const {language}=useLanguage();return <RealTimePhaseGraphic language={language} compact/>;}
