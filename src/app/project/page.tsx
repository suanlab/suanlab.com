import type { Metadata } from 'next';
import { Briefcase, Calendar, Users } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { projects, getActiveProjects, getCompletedProjects } from '@/data/projects';
import { researchProjects, projectStats } from '@/data/academic-activities';
import ProjectClient from './ProjectClient';

export const metadata: Metadata = {
  title: 'Projects | SuanLab',
  description:
    '이수안 교수의 연구 프로젝트 - 정부 R&D, 산학협력, AI/빅데이터 연구개발 과제',
  keywords: [
    '연구 프로젝트',
    'Projects',
    'R&D',
    '산학협력',
    'AI 연구',
    '빅데이터',
    '정부과제',
    '이수안',
  ],
  openGraph: {
    title: 'Projects | SuanLab',
    description: '이수안 교수의 연구 프로젝트 - 정부 R&D, 산학협력, AI/빅데이터 연구개발 과제',
    url: 'https://suanlab.com/project',
    type: 'website',
    siteName: 'SuanLab',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Projects | SuanLab',
    description: '이수안 교수의 연구 프로젝트 - 정부 R&D, 산학협력, AI/빅데이터 연구개발 과제',
  },
  alternates: {
    canonical: 'https://suanlab.com/project',
  },
};

function formatBudget(budget: string): string {
  const num = parseInt(budget.replace(/,/g, ''), 10);
  if (num >= 100000000) {
    return `${(num / 100000000).toFixed(1)}억원`;
  }
  if (num >= 10000) {
    return `${(num / 10000).toLocaleString()}만원`;
  }
  return `${num.toLocaleString()}원`;
}

function addFormattedBudget<T extends { budget: string }>(items: T[]) {
  return items.map((item) => ({
    ...item,
    formattedBudget: formatBudget(item.budget),
  }));
}

const uniqueAgencies = [...new Set(researchProjects.map((p) => p.fundingAgency))];
const allProjectsWithBudget = addFormattedBudget(projects);
const activeProjectsWithBudget = addFormattedBudget(getActiveProjects());
const completedProjectsWithBudget = addFormattedBudget(getCompletedProjects());

export default function ProjectPage() {
  return (
    <>
      <PageHeader
        title="Project"
        subtitle="Research projects funded by various organizations"
        breadcrumbs={[{ label: 'Project' }]}
      />

      <section className="py-8 bg-muted/30">
        <div className="container">
          <div className="grid grid-cols-3 gap-4">
            <Card className="text-center border-0 shadow-sm">
              <CardContent className="p-6">
                <Briefcase className="h-8 w-8 mx-auto mb-3 text-primary" />
                <p className="text-3xl font-bold text-primary">{projectStats.totalProjects}</p>
                <p className="text-sm text-muted-foreground mt-1">총 연구과제</p>
              </CardContent>
            </Card>
            <Card className="text-center border-0 shadow-sm">
              <CardContent className="p-6">
                <Calendar className="h-8 w-8 mx-auto mb-3 text-primary" />
                <p className="text-3xl font-bold text-primary">{projectStats.totalYears}년</p>
                <p className="text-sm text-muted-foreground mt-1">총 연구기간</p>
              </CardContent>
            </Card>
            <Card className="text-center border-0 shadow-sm">
              <CardContent className="p-6">
                <Users className="h-8 w-8 mx-auto mb-3 text-primary" />
                <p className="text-3xl font-bold text-primary">{uniqueAgencies.length}</p>
                <p className="text-sm text-muted-foreground mt-1">연구 기관</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
              Research <span className="text-primary">Projects</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              Data science and AI research projects in collaboration with government and industry
            </p>
          </div>

          <ProjectClient
            allProjects={allProjectsWithBudget}
            activeProjects={activeProjectsWithBudget}
            completedProjects={completedProjectsWithBudget}
          />
        </div>
      </section>
    </>
  );
}
