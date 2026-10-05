export interface Slide { title: string; points: string[]; notes?: string; contentHtml?: string; }
export function buildLectureSlides(lecture: { titleKo: string; titleEn: string; descriptionKo: string; topics: string[]; relatedYoutube?: string[] }): Slide[] {
  return [
    { title: lecture.titleKo, points: [lecture.titleEn, lecture.descriptionKo], notes: '강의 개요와 학습 범위를 소개합니다. 이 자료는 전체 강의 교안을 대체하지 않습니다.' },
    ...Array.from({ length: Math.ceil(lecture.topics.length / 4) }, (_, index) => ({
      title: `학습 주제 ${index + 1}`, points: lecture.topics.slice(index * 4, index * 4 + 4),
      notes: '각 주제를 설명하며 선수 지식과 학습 목표를 확인합니다.',
    })),
    { title: '학습을 이어가기', points: ['핵심 개념을 자신의 언어로 설명해 보세요.', '배운 방법을 작은 데이터와 코드로 실험해 보세요.', '강의 상세 페이지에서 관련 영상과 자료를 확인하세요.'] },
  ];
}
export function slideIndex(hash: string, count: number): number {
  const value = Number(hash.replace(/^#/, ''));
  return Number.isInteger(value) && value >= 1 && value <= count ? value - 1 : 0;
}
