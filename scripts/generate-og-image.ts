import fs from 'fs/promises';
import path from 'path';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

async function generateOGImage(): Promise<void> {
  if (!GEMINI_API_KEY) {
    throw new Error('GEMINI_API_KEY is not set');
  }

  const prompt = `Generate an image: Create a professional Open Graph banner image (1200x630 pixels, 1.91:1 aspect ratio) for "SuanLab - Data Science & AI Research Lab".

Design requirements:
- Modern, clean tech aesthetic with gradient background (deep blue to purple/teal)
- Abstract data visualization elements (neural network nodes, data flow lines, geometric patterns)
- Professional and academic feel suitable for a research lab
- No text or letters in the image
- Subtle grid or matrix pattern in background
- Glowing accent elements suggesting AI/ML concepts
- High contrast, vibrant but professional colors`;

  console.log('Generating OG image with Gemini...');

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${GEMINI_API_KEY}`,
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
    throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();

  // Extract image from response
  const parts = data.candidates?.[0]?.content?.parts || [];
  for (const part of parts) {
    if (part.inlineData?.data) {
      const imageBuffer = Buffer.from(part.inlineData.data, 'base64');

      // Save to public/assets/images/
      const outputPath = path.join(process.cwd(), 'public/assets/images/og-image.jpg');
      await fs.writeFile(outputPath, imageBuffer);

      console.log(`OG image saved to: ${outputPath}`);
      console.log(`File size: ${(imageBuffer.length / 1024).toFixed(2)} KB`);
      return;
    }
  }

  throw new Error('No image data in response');
}

generateOGImage()
  .then(() => {
    console.log('Done!');
    process.exit(0);
  })
  .catch((err) => {
    console.error('Error:', err);
    process.exit(1);
  });
