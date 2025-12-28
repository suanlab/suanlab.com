const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '../../WWW/project/index.html'), 'utf-8');

const projects = [];

// Find all POST ITEM sections
const postItemRegex = /<!-- POST ITEM -->([\s\S]*?)<!-- \/POST ITEM -->/g;
let match;
let id = 1;

while ((match = postItemRegex.exec(html)) !== null) {
  const section = match[1];

  // Extract title
  const titleMatch = section.match(/<h2 class="size-20">([^<]+)<\/h2>/);
  if (!titleMatch) continue;
  const title = titleMatch[1].trim();

  // Extract organization
  const orgMatch = section.match(/<i class="fa fa-sitemap"><\/i>\s*<span class="font-lato">([^<]+)<\/span>/);
  const organization = orgMatch ? orgMatch[1].trim() : '';

  // Extract program
  const programMatch = section.match(/<i class="fa fa-folder-open-o"><\/i>\s*<span class="font-lato">([^<]+)<\/span>/);
  const program = programMatch ? programMatch[1].trim() : '';

  // Extract period
  const periodMatch = section.match(/<i class="fa fa-clock-o"><\/i>\s*<span class="font-lato">([^<]+)<\/span>/);
  const period = periodMatch ? periodMatch[1].trim() : '';

  // Extract budget
  const budgetMatch = section.match(/<i class="fa fa-krw"><\/i>\s*<span class="font-lato">([^<]+)<\/span>/);
  const budget = budgetMatch ? budgetMatch[1].trim() : '';

  // Check if completed (has ribbon)
  const isCompleted = section.includes('<div class="ribbon-inner">Completed</div>');

  // Extract description items
  const items = [];
  const listMatch = section.match(/<ul class="list-inline">\s*([\s\S]*?)<\/ul>/);
  if (listMatch) {
    const listContent = listMatch[1];
    const itemRegex = /<li>([^<]+)<\/li>/g;
    let itemMatch;
    while ((itemMatch = itemRegex.exec(listContent)) !== null) {
      items.push(itemMatch[1].trim());
    }
  }

  // Extract URL
  const urlMatch = section.match(/<a href="([^"]+)" target="_blank_?"/);
  const url = urlMatch ? urlMatch[1] : undefined;

  const project = {
    id: id++,
    title,
    organization,
    program,
    period,
    budget,
    completed: isCompleted,
    items,
  };

  if (url) project.url = url;

  projects.push(project);
}

// Generate TypeScript
const output = `export interface Project {
  id: number;
  title: string;
  organization: string;
  program: string;
  period: string;
  budget: string;
  completed: boolean;
  items: string[];
  url?: string;
}

export const projects: Project[] = ${JSON.stringify(projects, null, 2)};

export const getActiveProjects = (): Project[] => {
  return projects.filter((p) => !p.completed);
};

export const getCompletedProjects = (): Project[] => {
  return projects.filter((p) => p.completed);
};
`;

fs.writeFileSync(path.join(__dirname, '../src/data/projects/index.ts'), output);
console.log('Extracted ' + projects.length + ' projects');
console.log('Active: ' + projects.filter(p => !p.completed).length);
console.log('Completed: ' + projects.filter(p => p.completed).length);
