import { getLectureContent } from '@/lib/lecture-content';
import { notFound } from 'next/navigation';
import { lectures, getLectureBySlug } from '@/data/lectures';
import { buildLectureSlides } from '@/lib/presentation';
import LectureDeck from '@/components/presentation/LectureDeck';
import { site } from '@/config/site';
export function generateStaticParams() { return lectures.map(({ slug }) => ({ slug })); }
export function generateMetadata({ params }: { params: { slug: string } }) {
  const lecture = getLectureBySlug(params.slug);
  return { title: `${lecture?.titleKo || 'Lecture'} · 발표`, alternates: { canonical: `${site.url}/lecture/${params.slug}/present/` } };
}
export default async function PresentationPage({ params }: { params: { slug: string } }) {
  const lecture = getLectureBySlug(params.slug); if (!lecture) notFound();
  const content = await getLectureContent(lecture.slug);
  return <LectureDeck slides={content?.slides || buildLectureSlides(lecture)} slug={lecture.slug} title={lecture.titleKo} />;
}
