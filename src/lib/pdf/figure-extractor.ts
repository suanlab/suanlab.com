import { execSync } from 'child_process';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';

const BLOG_IMAGE_DIR = path.join(process.cwd(), 'public/assets/images/blog');

interface FigureExtractionOptions {
  pdfBuffer: Buffer;
  slug: string;
}

interface ExtractionResult {
  success: boolean;
  imagePath?: string;
  error?: string;
}

/**
 * Extract first page of PDF (with paper title) as blog thumbnail
 * Uses high DPI for quality and crops to OG image standard size
 */
export async function extractFigureFromPdf(
  options: FigureExtractionOptions
): Promise<ExtractionResult> {
  const { pdfBuffer, slug } = options;
  const tempDir = os.tmpdir();
  const tempPdfPath = path.join(tempDir, `${slug}-temp.pdf`);
  const tempImageBase = path.join(tempDir, `${slug}-page`);

  try {
    // Save PDF to temp file
    await fs.writeFile(tempPdfPath, pdfBuffer);

    // Convert first page at high resolution (300 DPI for better quality)
    const outputPath = `${tempImageBase}-1`;
    execSync(
      `pdftoppm -png -f 1 -l 1 -r 300 "${tempPdfPath}" "${outputPath}"`,
      { encoding: 'utf-8', timeout: 30000 }
    );

    // Find the generated file
    const imageFile = await findGeneratedImage(outputPath);
    if (!imageFile) {
      return { success: false, error: 'Could not extract first page from PDF' };
    }

    const buffer = await fs.readFile(imageFile);
    const outputFilePath = path.join(BLOG_IMAGE_DIR, `${slug}.jpg`);
    await fs.mkdir(BLOG_IMAGE_DIR, { recursive: true });

    const targetWidth = 1200;
    const targetHeight = 630; // OG image standard

    // Get image metadata
    const metadata = await sharp(buffer).metadata();
    const { width = 1200, height = 1600 } = metadata;

    // Use top 60% of the first page (title, authors, abstract area)
    const cropHeight = Math.floor(height * 0.6);

    await sharp(buffer)
      .extract({
        left: 0,
        top: 0,
        width: width,
        height: cropHeight
      })
      .resize(targetWidth, targetHeight, {
        fit: 'cover',
        position: 'north'  // Keep top portion (title area)
      })
      .jpeg({ quality: 90 })
      .toFile(outputFilePath);

    // Cleanup temp files
    await cleanupTempFiles(tempPdfPath, imageFile);

    return { success: true, imagePath: `/assets/images/blog/${slug}.jpg` };

  } catch (error) {
    // Cleanup on error
    await fs.unlink(tempPdfPath).catch(() => {});
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error during extraction'
    };
  }
}

/**
 * Find generated image file (pdftoppm adds various suffixes)
 */
async function findGeneratedImage(basePath: string): Promise<string | null> {
  const possibleNames = [
    `${basePath}-1.png`,
    `${basePath}-01.png`,
    `${basePath}.png`
  ];

  for (const name of possibleNames) {
    try {
      await fs.access(name);
      return name;
    } catch {
      continue;
    }
  }
  return null;
}

/**
 * Alias for extractFigureFromPdf (for backwards compatibility)
 */
export async function extractFirstFigure(
  pdfBuffer: Buffer,
  slug: string
): Promise<ExtractionResult> {
  return extractFigureFromPdf({ pdfBuffer, slug });
}

/**
 * Cleanup temporary files
 */
async function cleanupTempFiles(pdfPath: string, imagePath: string): Promise<void> {
  await Promise.all([
    fs.unlink(pdfPath).catch(() => {}),
    fs.unlink(imagePath).catch(() => {})
  ]);
}
