'use client';
import React from 'react';
import Link from 'next/link';
import type { ContentProvenance } from '@/lib/content-provenance';
import { useLanguage } from '@/components/language-provider';

export default function ContentProvenanceNotice({ provenance, paperReview }: { provenance?: ContentProvenance; paperReview: boolean }) {
  const { language } = useLanguage();
  const ko = language === 'ko';
  if (!provenance && !paperReview) return null;
  return <aside aria-label={ko ? '출처와 검토 상태' : 'Source and review status'} className="mb-8 rounded-xl border bg-muted/30 p-5 text-sm leading-relaxed">
    <p className="font-semibold">{paperReview ? (ko ? '논문 리뷰' : 'Paper review') : (ko ? 'AI 보조 작성' : 'AI-assisted writing')}
      {' · '}{provenance?.review.status === 'reviewed' ? (ko ? '검토 완료' : 'Reviewed') : provenance ? (ko ? 'AI 초안 · 검토 대기' : 'AI draft · Awaiting review') : (ko ? '검토 기록 없음' : 'Review not recorded')}</p>
    {provenance && <>
      <p className="mt-2 text-muted-foreground">{ko ? '생성일' : 'Generated'}: {provenance.generatedAt.slice(0, 10)}{provenance.review.status === 'reviewed' && <> · {ko ? '검토자' : 'Reviewer'}: {provenance.review.reviewer} ({provenance.review.reviewedAt?.slice(0, 10)})</>}</p>
      <p className="mt-3 font-medium">{provenance.source.kind === 'topic' ? (ko ? '생성 주제' : 'Requested topic') : provenance.source.kind === 'pdf' ? (ko ? 'PDF에서 추출한 출처 정보' : 'Source metadata extracted from PDF') : (ko ? 'arXiv 원문' : 'arXiv source')}: {provenance.source.title}</p>
      {provenance.source.authors?.length ? <p className="mt-1 text-muted-foreground">{provenance.source.authors.join(', ')}</p> : null}
      {provenance.source.url && <a href={provenance.source.url} target="_blank" rel="noopener noreferrer" className="mt-2 inline-block text-primary underline">{ko ? '원문 확인' : 'Read source'} ↗</a>}
    </>}
    {paperReview && <p className="mt-3 text-muted-foreground">{ko ? '이 글은 논문을 소개하는 학습 자료입니다. 연구실의 논문 목록은 별도로 확인할 수 있습니다.' : 'This article explains a paper for learning. The lab’s publication record is listed separately.'} <Link href="/publication/" className="text-primary underline">Publications</Link></p>}
  </aside>;
}
