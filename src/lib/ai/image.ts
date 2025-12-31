import fs from 'fs/promises';
import path from 'path';

const IMAGES_DIR = 'public/assets/images/blog';

export interface ImageGenerationOptions {
  topic: string;
  style?: 'technical' | 'abstract' | 'realistic';
}

/**
 * Generate a blog thumbnail image using Nano Banana Pro (Gemini 3 Pro Image)
 */
export async function generateBlogImage(
  options: ImageGenerationOptions
): Promise<Buffer> {
  const { topic, style = 'technical' } = options;

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not set');
  }

  const stylePrompts: Record<string, string> = {
    technical: 'Clean technical illustration with simple geometric shapes, circuit-like patterns, modern flat design, subtle gradients, professional color palette (blues, teals, purples), minimal and elegant, suitable for a tech blog header',
    abstract: 'Minimalist abstract art, clean geometric shapes, modern design, soft gradients, professional and sophisticated, subtle color palette',
    realistic: 'Photorealistic professional image, high quality, detailed, modern lighting, clean composition',
  };

  const prompt = `Generate an image: Create a wide banner image (16:9 aspect ratio) for a blog post about "${topic}". ${stylePrompts[style]}. No text or letters in the image.`;

  console.log('Generating thumbnail image with Nano Banana Pro...');

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: prompt }]
        }],
        generationConfig: {
          responseModalities: ['IMAGE', 'TEXT']
        }
      })
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Nano Banana Pro API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();

  // Extract image from response
  const parts = data.candidates?.[0]?.content?.parts || [];
  for (const part of parts) {
    if (part.inlineData?.data) {
      const imageBuffer = Buffer.from(part.inlineData.data, 'base64');
      return imageBuffer;
    }
  }

  throw new Error('No image data in response');
}

/**
 * Save image buffer to local filesystem
 */
export async function saveImageBuffer(
  buffer: Buffer,
  filename: string,
  mimeType: string = 'image/jpeg'
): Promise<string> {
  // Ensure directory exists
  const fullDir = path.join(process.cwd(), IMAGES_DIR);
  await fs.mkdir(fullDir, { recursive: true });

  // Determine extension based on mime type
  const ext = mimeType.includes('png') ? '.png' : '.jpg';
  const finalFilename = filename.endsWith(ext) ? filename : `${filename}${ext}`;
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
    const imageBuffer = await generateBlogImage({ topic, style });
    const savedPath = await saveImageBuffer(imageBuffer, slug);
    console.log(`Thumbnail saved: ${savedPath}`);
    return savedPath;
  } catch (error) {
    console.warn('Failed to generate image:', error instanceof Error ? error.message : error);
    console.warn('Using default thumbnail');
    return '/assets/images/blog/default.jpg';
  }
}
