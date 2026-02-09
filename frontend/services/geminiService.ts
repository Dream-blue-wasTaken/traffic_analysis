import { GoogleGenAI, Type } from "@google/genai";
import { AnalysisResult } from "../types";
import { analyzeImageWithOpenRouter } from "./openRouterService";

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
const MODEL_NAME = "gemini-3-flash-preview";

/**
 * Converts a File object to a Base64 string (raw data, no prefix).
 */
const fileToGenerativePart = async (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      const base64Data = base64String.split(',')[1];
      resolve(base64Data);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

/**
 * Internal function to attempt analysis with Gemini.
 */
const analyzeWithGemini = async (file: File): Promise<AnalysisResult> => {
  const base64Image = await fileToGenerativePart(file);

  const prompt = `
    You are an expert image analysis system.
    Analyze the provided image and extract ONLY the following fields:
    - object: The main object or scene description (string).
    - people_count: The number of people visible (integer).
    - helmet: Whether people are wearing helmets. Return 'yes', 'no', or 'unknown'.
  `;

  const response = await ai.models.generateContent({
    model: MODEL_NAME,
    contents: {
      parts: [
        {
          inlineData: {
            mimeType: file.type,
            data: base64Image,
          },
        },
        {
          text: prompt,
        },
      ],
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          object: {
            type: Type.STRING,
            description: "The main object or subject of the image.",
          },
          people_count: {
            type: Type.INTEGER,
            description: "Count of people in the image.",
          },
          helmet: {
            type: Type.STRING,
            description: "Indicates if helmets are worn. Values: yes, no, unknown.",
            enum: ["yes", "no", "unknown"],
          },
        },
        required: ["object", "people_count", "helmet"],
      },
    },
  });

  const text = response.text;
  if (!text) {
    throw new Error("No response received from Gemini.");
  }

  const result = JSON.parse(text) as AnalysisResult;
  return { ...result, provider: 'Gemini' };
};

/**
 * Main analysis function with Fallback Strategy:
 * 1. Try OpenRouter (Main)
 * 2. If OpenRouter fails or missing key, try Gemini (Fallback)
 */
export const analyzeImage = async (file: File): Promise<AnalysisResult> => {
  let openRouterError = null;

  // 1. Try OpenRouter (Main)
  if (process.env.OPENROUTER_API_KEY) {
    try {
      console.log("Attempting analysis with OpenRouter...");
      return await analyzeImageWithOpenRouter(file);
    } catch (error) {
      console.warn("OpenRouter analysis failed. Attempting fallback.", error);
      openRouterError = error;
    }
  } else {
    console.warn("OpenRouter API Key missing. Skipping to fallback.");
  }

  // 2. Fallback to Gemini
  if (process.env.API_KEY) {
    try {
      console.log("Attempting analysis with Gemini fallback...");
      return await analyzeWithGemini(file);
    } catch (geminiError) {
      console.error("Gemini fallback also failed.", geminiError);
      throw new Error(`Analysis failed. OpenRouter Error: ${openRouterError?.message || 'N/A'}. Gemini Error: ${(geminiError as Error).message}`);
    }
  }

  // If we reach here, neither worked
  throw new Error(openRouterError ? `OpenRouter failed: ${(openRouterError as Error).message}` : "No API keys configured for OpenRouter or Gemini.");
};