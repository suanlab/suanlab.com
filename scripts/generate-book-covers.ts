import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

const BOOKS_IMAGES_DIR = 'public/assets/images/books';

interface BookInfo {
  slug: string;
  title: string;
  prompt: string;
}

const books: BookInfo[] = [
  {
    slug: 'python-programming-basics',
    title: '파이썬 프로그래밍 한번에 끝내기',
    prompt: 'Python programming book cover. Clean modern design with Python snake logo stylized, code snippets floating, blue and yellow color scheme, minimalist geometric patterns, professional tech book aesthetic, no text'
  },
  {
    slug: 'data-science-python',
    title: '데이터 처리 및 시각화 한번에 끝내기',
    prompt: 'Data science and visualization book cover. Beautiful data charts, graphs, scatter plots, bar charts in 3D isometric style, colorful gradient (orange, purple, teal), flowing data streams, modern clean design, no text'
  },
  {
    slug: 'machine-learning-fundamentals',
    title: '머신러닝 한번에 끝내기',
    prompt: 'Machine learning book cover. Neural network nodes connecting, gradient descent visualization, decision tree branches, abstract geometric brain patterns, blue and purple gradient, futuristic tech aesthetic, no text'
  },
  {
    slug: 'deep-learning-pytorch',
    title: '딥러닝 한번에 끝내기 with PyTorch',
    prompt: 'Deep learning with PyTorch book cover. Deep neural network layers visualization, PyTorch flame logo stylized, tensor operations, GPU computing visual, red-orange fire theme with dark background, dramatic lighting, no text'
  },
  {
    slug: 'natural-language-processing',
    title: '자연어 처리부터 언어 모델까지 한번에 끝내기',
    prompt: 'NLP and language models book cover. Word embeddings visualization, transformer attention patterns, text flowing and transforming, chat bubbles, GPT-style interface elements, green and cyan color scheme, modern AI aesthetic, no text'
  },
  {
    slug: 'computer-vision',
    title: '이미지 처리부터 컴퓨터 비전까지 한번에 끝내기',
    prompt: 'Computer vision book cover. Eye with digital circuits, image recognition bounding boxes, convolutional filter visualization, camera lens with AI overlay, object detection grid, blue and cyan tech aesthetic, no text'
  },
  {
    slug: 'audio-speech-processing',
    title: '오디오 신호 처리부터 음성 인식 생성까지 한번에 끝내기',
    prompt: 'Audio and speech processing book cover. Sound waveforms visualization, spectrogram colorful display, microphone with digital waves, music notes transforming to data, purple and pink gradient, modern audio tech design, no text'
  },
  {
    slug: 'agentic-ai',
    title: 'Agentic AI 한번에 끝내기',
    prompt: 'Agentic AI book cover. AI agent robot with glowing eyes, multiple connected AI nodes working together, tool icons floating around, autonomous workflow visualization, electric blue and gold color scheme, futuristic and powerful design, no text'
  }
];

async function generateBookCover(book: BookInfo): Promise<Buffer> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not set');
  }

  const fullPrompt = `Generate a professional book cover image (portrait orientation, 3:4 aspect ratio). ${book.prompt}. High quality, suitable for online book display, clean professional look.`;

  console.log(`Generating cover for: ${book.title}`);

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: fullPrompt }]
        }],
        generationConfig: {
          responseModalities: ['IMAGE', 'TEXT']
        }
      })
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  const parts = data.candidates?.[0]?.content?.parts || [];

  for (const part of parts) {
    if (part.inlineData?.data) {
      return Buffer.from(part.inlineData.data, 'base64');
    }
  }

  throw new Error('No image data in response');
}

async function saveBookCover(buffer: Buffer, slug: string): Promise<string> {
  const fullDir = path.join(process.cwd(), BOOKS_IMAGES_DIR);
  await fs.mkdir(fullDir, { recursive: true });

  const filename = `${slug}.jpg`;
  const filepath = path.join(fullDir, filename);
  await fs.writeFile(filepath, buffer);

  return `/assets/images/books/${filename}`;
}

async function updateBookFrontmatter(slug: string, imagePath: string): Promise<void> {
  const bookPath = path.join(process.cwd(), 'content/books', `${slug}.md`);

  try {
    let content = await fs.readFile(bookPath, 'utf-8');

    // Update or add image field in frontmatter
    if (content.includes('image:')) {
      content = content.replace(/image:\s*"[^"]*"/, `image: "${imagePath}"`);
    } else {
      // Add image after date line
      content = content.replace(/(date:\s*"[^"]*")/, `$1\nimage: "${imagePath}"`);
    }

    await fs.writeFile(bookPath, content);
    console.log(`Updated frontmatter for ${slug}`);
  } catch (error) {
    console.error(`Failed to update frontmatter for ${slug}:`, error);
  }
}

async function main() {
  console.log('Starting book cover generation with Nano Banana Pro...\n');

  for (const book of books) {
    try {
      console.log(`\n--- Processing: ${book.title} ---`);

      const imageBuffer = await generateBookCover(book);
      const imagePath = await saveBookCover(imageBuffer, book.slug);
      await updateBookFrontmatter(book.slug, imagePath);

      console.log(`✓ Saved: ${imagePath}`);

      // Small delay between API calls
      await new Promise(resolve => setTimeout(resolve, 2000));
    } catch (error) {
      console.error(`✗ Failed for ${book.title}:`, error instanceof Error ? error.message : error);
    }
  }

  console.log('\n=== Book cover generation complete ===');
}

main().catch(console.error);
