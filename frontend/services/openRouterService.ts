import { AnalysisResult } from "../types";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const SITE_URL = window.location.origin;
const SITE_NAME = "VisionAnalytica";

// Using a reliable vision-capable model on OpenRouter
// Google Gemini 2.0 Flash is a good default on OpenRouter
const MODEL = "google/gemma-3-4b-it:free";

const fileToDataURL = async (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

export const analyzeImageWithOpenRouter = async (file: File): Promise<AnalysisResult> => {
  if (!OPENROUTER_API_KEY) {
    throw new Error("OpenRouter API Key is missing.");
  }

  try {
    const base64Image = await fileToDataURL(file);

    const prompt = `
      You are an expert image analysis system.
      Analyze the provided image and extract ONLY the following fields in strict JSON format:
      - object: The main object or scene description (string).
      - people_count: The number of people visible (integer).
      - helmet: Whether people are wearing helmets. Return 'yes', 'no', or 'unknown'.

      Return ONLY valid JSON. Do not include markdown formatting or explanations.
    `;

    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              {
                type: "image_url",
                image_url: {
                  url: base64Image
                }
              }
            ]
          }
        ]
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`OpenRouter API Error: ${response.status} ${errorData.error?.message || response.statusText}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    if (!content) {
      throw new Error("Empty response from OpenRouter.");
    }

    // Extract JSON block using regex to handle potential markdown code blocks or extra text
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    const jsonString = jsonMatch ? jsonMatch[0] : content.replace(/```json\n?|```/g, "").trim();
    
    const result = JSON.parse(jsonString) as AnalysisResult;
    return { ...result, provider: 'OpenRouter' };

  } catch (error) {
    console.error("OpenRouter analysis failed:", error);
    throw error;
  }
};