const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '../../WWW/course/index.html'), 'utf-8');

const courses = [];

// Find all POST ITEM sections
const postItemRegex = /<!-- POST ITEM -->([\s\S]*?)<!-- \/POST ITEM -->/g;
let match;
let id = 1;

while ((match = postItemRegex.exec(html)) !== null) {
  const section = match[1];

  // Extract title from h2
  const titleMatch = section.match(/<h2>([^<]+)<\/h2>/);
  if (!titleMatch) continue;
  const title = titleMatch[1].trim();

  // Extract subtitle from span after h2
  const subtitleMatch = section.match(/<\/h2>\s*<span>([^<]+)<\/span>/);
  const subtitle = subtitleMatch ? subtitleMatch[1].trim() : '';

  // Extract date
  const dateMatch = section.match(/<i class="fa fa-clock-o"><\/i>\s*<span class="font-lato">([^<]+)<\/span>/);
  const date = dateMatch ? dateMatch[1].trim() : '';

  // Extract location
  const locationMatch = section.match(/<i class="fa fa-map-marker"><\/i>\s*<span class="font-lato">([^<]+)<\/span>/);
  const location = locationMatch ? locationMatch[1].trim() : '';

  // Extract materials (tasks with links)
  const materials = [];
  const taskRegex = /<i class="fa fa-tasks"><\/i>\s*<span class="font-lato">\s*([\s\S]*?)<\/span>\s*<\/li>/g;
  let taskMatch;

  while ((taskMatch = taskRegex.exec(section)) !== null) {
    const taskContent = taskMatch[1].trim();

    // Extract task title (text before any link)
    let taskTitle = taskContent.replace(/<[^>]+>/g, '').replace(/<!--[\s\S]*?-->/g, '').trim();

    // Check for colab link
    const colabMatch = taskContent.match(/<a href="(https:\/\/colab\.research\.google\.com[^"]+)"/);
    // Check for pdf link
    const pdfMatch = taskContent.match(/<a href="([^"]+\.pdf)"/);
    // Check for drive link
    const driveMatch = taskContent.match(/<a href="(https:\/\/drive\.google\.com[^"]+)"/);

    if (colabMatch) {
      materials.push({ title: taskTitle, url: colabMatch[1], type: 'colab' });
    } else if (pdfMatch) {
      let pdfUrl = pdfMatch[1];
      if (pdfUrl.startsWith('../')) {
        pdfUrl = pdfUrl.replace('../', '/');
      }
      materials.push({ title: taskTitle, url: pdfUrl, type: 'pdf' });
    } else if (driveMatch) {
      materials.push({ title: taskTitle, url: driveMatch[1], type: 'drive' });
    } else if (taskTitle) {
      // Task without active link
      materials.push({ title: taskTitle, url: '', type: 'none' });
    }
  }

  // Extract images if present
  const images = [];
  const imageRegex = /src="\.\.\/assets\/images\/course\/([^"]+)"/g;
  let imageMatch;
  while ((imageMatch = imageRegex.exec(section)) !== null) {
    images.push('/assets/images/course/' + imageMatch[1]);
  }

  const course = {
    id: id++,
    title,
    subtitle,
    date,
    location,
  };

  if (materials.length > 0) {
    // Only include materials with actual links
    const activeMaterials = materials.filter(m => m.url !== '');
    if (activeMaterials.length > 0) {
      course.materials = activeMaterials;
    }
  }

  if (images.length > 0) {
    course.images = images;
  }

  courses.push(course);
}

// Generate TypeScript
const output = `export interface CourseMaterial {
  title: string;
  url: string;
  type: 'colab' | 'pdf' | 'drive';
}

export interface Course {
  id: number;
  title: string;
  subtitle: string;
  date: string;
  location: string;
  materials?: CourseMaterial[];
  images?: string[];
}

export const courses: Course[] = ${JSON.stringify(courses, null, 2)};

export const getCoursesByYear = (year: string): Course[] => {
  return courses.filter((c) => c.date.includes(year));
};
`;

// Create data directory if needed
const dataDir = path.join(__dirname, '../src/data/courses');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

fs.writeFileSync(path.join(dataDir, 'index.ts'), output);
console.log('Extracted ' + courses.length + ' courses');
