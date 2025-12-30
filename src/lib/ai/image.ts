import OpenAI from 'openai';
import fs from 'fs/promises';
import path from 'path';

const IMAGES_DIR = 'public/assets/images/blog';

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OPENAI_API_KEY is not set');
    }
    client = new OpenAI({ apiKey });
  }
  return client;
}

export interface ImageGenerationOptions {
  topic: string;
  style?: 'technical' | 'abstract' | 'realistic';
  size?: '1024x1024' | '1792x1024' | '1024x1792';
}

/**
 * Generate a blog thumbnail image using DALL-E 3
 */
export async function generateBlogImage(
  options: ImageGenerationOptions
): Promise<string> {
  const { topic, style = 'technical', size = '1792x1024' } = options;

  const stylePrompts: Record<string, string> = {
    technical: 'Clean technical illustration with simple geometric shapes, circuit-like patterns, modern flat design, subtle gradients, professional color palette (blues, teals, purples), minimal and elegant, suitable for a tech blog header',
    abstract: 'Minimalist abstract art, clean geometric shapes, modern design, soft gradients, professional and sophisticated, subtle color palette',
    realistic: 'Photorealistic professional image, high quality, detailed, modern lighting, clean composition',
  };

  const prompt = `Create a wide banner image for a blog post about "${topic}". ${stylePrompts[style]}. No text or letters in the image.`;

  console.log('Generating thumbnail image with DALL-E 3...');

  const openai = getClient();
  const response = await openai.images.generate({
    model: 'dall-e-3',
    prompt,
    n: 1,
    size,
    quality: 'standard',
  });

  const imageUrl = response.data?.[0]?.url;
  if (!imageUrl) {
    throw new Error('Failed to generate image');
  }

  return imageUrl;
}

/**
 * Download image from URL and save to local filesystem
 */
export async function saveImageFromUrl(
  imageUrl: string,
  filename: string
): Promise<string> {
  const response = await fetch(imageUrl);
  if (!response.ok) {
    throw new Error(`Failed to download image: ${response.statusText}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());

  // Ensure directory exists
  const fullDir = path.join(process.cwd(), IMAGES_DIR);
  await fs.mkdir(fullDir, { recursive: true });

  // Save with .png extension
  const finalFilename = filename.endsWith('.png') ? filename : `${filename}.png`;
  const filepath = path.join(fullDir, finalFilename);

  await fs.writeFile(filepath, buffer);

  // Return the public path for use in frontmatter
  return `/assets/images/blog/${finalFilename}`;
}

/**
 * Generate and save blog thumbnail
 */
export async function generateAndSaveThumbnail(
  topic: string,
  slug: string,
  style: 'technical' | 'abstract' | 'realistic' = 'technical'
): Promise<string> {
  try {
    const imageUrl = await generateBlogImage({ topic, style });
    const savedPath = await saveImageFromUrl(imageUrl, slug);
    console.log(`Thumbnail saved: ${savedPath}`);
    return savedPath;
  } catch (error) {
    console.warn('Failed to generate image:', error instanceof Error ? error.message : error);
    console.warn('Using default thumbnail');
    return '/assets/images/blog/default.jpg';
  }
}
