'use client';

import { useState } from 'react';
import { Building, FolderOpen, Calendar, DollarSign, Check, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

type FilterType = 'all' | 'active' | 'completed';

export interface ProjectWithBudget {
  id: number;
  title: string;
  organization: string;
  program: string;
  period: string;
  budget: string;
  completed: boolean;
  items: string[];
  url?: string;
  formattedBudget: string;
}

interface ProjectClientProps {
  allProjects: ProjectWithBudget[];
  activeProjects: ProjectWithBudget[];
  completedProjects: ProjectWithBudget[];
}

function ProjectCard({ project, showActiveStyle }: { project: ProjectWithBudget; showActiveStyle?: boolean }) {
  return (
    <Card className={cn('h-full flex flex-col', showActiveStyle && 'border-green-200 dark:border-green-900')}>
      <CardHeader>
        <div className="flex items-center gap-2 mb-2">
          {project.completed ? (
            <Badge variant="secondary">
              <CheckCircle className="mr-1 h-3 w-3" />
              Completed
            </Badge>
          ) : (
            <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100">
              <FolderOpen className="mr-1 h-3 w-3" />
              Active
            </Badge>
          )}
        </div>
        <CardTitle className="text-lg leading-tight">{project.title}</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <div className="space-y-3 text-sm mb-4">
          <div className="flex items-start gap-2">
            <Building className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
            <div>
              <p className="text-muted-foreground">Organization</p>
              <p className="font-medium">{project.organization}</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <FolderOpen className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
            <div>
              <p className="text-muted-foreground">Program</p>
              <p className="font-medium">{project.program}</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <Calendar className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
            <div>
              <p className="text-muted-foreground">Period</p>
              <p className="font-medium">{project.period}</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <DollarSign className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
            <div>
              <p className="text-muted-foreground">Budget</p>
              <p className="font-medium">{project.formattedBudget}</p>
            </div>
          </div>
        </div>

        <div className="flex-1">
          <ul className="space-y-2">
            {project.items.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                <Check className="h-4 w-4 mt-0.5 text-green-500 shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

export default function ProjectClient({ allProjects, activeProjects, completedProjects }: ProjectClientProps) {
  const [filter, setFilter] = useState<FilterType>('all');

  const filteredProjects = filter === 'all'
    ? allProjects
    : filter === 'active'
      ? activeProjects
      : completedProjects;

  const filters: { key: FilterType; label: string; count: number }[] = [
    { key: 'all', label: 'Total', count: allProjects.length },
    { key: 'active', label: 'Active', count: activeProjects.length },
    { key: 'completed', label: 'Completed', count: completedProjects.length },
  ];

  return (
    <>
      <div className="mt-6 flex justify-center gap-2 flex-wrap">
        {filters.map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={cn(
              'inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium transition-all',
              filter === f.key
                ? f.key === 'active'
                  ? 'bg-green-500 text-white shadow-md'
                  : 'bg-primary text-primary-foreground shadow-md'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground'
            )}
          >
            {f.key === 'active' && (
              <span className={cn(
                'inline-block w-2 h-2 rounded-full',
                filter === f.key ? 'bg-white' : 'bg-green-500',
                filter !== f.key && 'animate-pulse'
              )} />
            )}
            {f.key === 'completed' && <CheckCircle className="h-3.5 w-3.5" />}
            {f.label}
            <span className={cn(
              'ml-1 px-1.5 py-0.5 rounded-full text-xs',
              filter === f.key
                ? 'bg-white/20'
                : 'bg-background'
            )}>
              {f.count}
            </span>
          </button>
        ))}
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredProjects.map((project) => (
          <ProjectCard
            key={project.id}
            project={project}
            showActiveStyle={!project.completed}
          />
        ))}
      </div>

      {filteredProjects.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          No projects found.
        </div>
      )}
    </>
  );
}
