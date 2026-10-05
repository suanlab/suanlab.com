import Link from 'next/link';
import { notFound } from 'next/navigation';
import { getLectureContent, getLectureContentSlugs } from '@/lib/lecture-content';
import { site } from '@/config/site';
import 'katex/dist/katex.min.css';
export function generateStaticParams() { return getLectureContentSlugs().map(slug => ({ slug })); }
export async function generateMetadata({ params }: { params: { slug: string } }) {
  const content = await getLectureContent(params.slug);
  return { title: content?.title || 'Lecture notes', alternates: { canonical: `${site.url}/lecture/${params.slug}/notes/` } };
}
export default async function LectureNotes({ params }: { params: { slug: string } }) {
  const content = await getLectureContent(params.slug); if (!content) notFound();
  return <article className="container max-w-4xl py-12">
    <Link href={`/lecture/${params.slug}/`} className="text-sm text-primary">← 강의 소개</Link>
    <h1 className="my-6 text-3xl font-semibold">{content.title}</h1>
    <Link href={`/lecture/${params.slug}/present/`} className="text-primary underline">발표 모드 열기</Link>
    <nav aria-label="목차" className="my-8 rounded-xl border bg-muted/30 p-5"><ol className="list-inside list-decimal space-y-2">{content.slides.map((slide, i) => <li key={i}><a href={`#section-${i + 1}`} className="hover:underline">{slide.title}</a></li>)}</ol></nav>
    {content.slides.map((slide, i) => <section id={`section-${i + 1}`} key={i} className="scroll-mt-24 border-t py-10"><h2 className="mb-6 text-2xl font-semibold">{slide.title}</h2><div className="lecture-prose" dangerouslySetInnerHTML={{ __html: slide.contentHtml || '' }} /><Link href={`/lecture/${params.slug}/present/#${i + 1}`} className="mt-6 inline-block text-sm text-primary underline">이 내용을 슬라이드로 보기</Link></section>)}
  </article>;
}
