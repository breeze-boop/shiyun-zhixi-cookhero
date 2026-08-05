export type Source = {
  title: string;
  dish_name: string;
  category: string;
  difficulty: string;
  source: string;
  score: number | null;
  data_source: string;
};

export type ChatResponse = {
  answer: string;
  thought: string;
  action: string;
  observation: Record<string, unknown>;
  rewritten_query: string;
  metadata_expression: string | null;
  sources: Source[];
  trace: string[];
  context_preview: string;
};

export type PersonalDocumentResponse = {
  doc_id: string;
  indexed: boolean;
};

export type ToolCatalogItem = {
  name: string;
  description: string;
  provider: "local" | "mcp" | "subagent";
  input_schema: Record<string, unknown>;
};

export type VisionAnalyzeResponse = {
  dish_name: string;
  ingredients: string[];
  nutrition: Record<string, string | number>;
  advice: string[];
  confidence: number;
};

export type MealImageAnalysis = Record<string, unknown>;

export type DietPlanResponse = {
  plan_id: string;
  user_id: string;
  goal: string;
  days: number;
  content: string;
};

export type MealCheckinResponse = {
  checkin_id: string;
  user_id: string;
  meal_time: string;
  description: string;
  image_analysis: MealImageAnalysis;
};

export type NutritionAnalysisResponse = {
  report_id: string;
  user_id: string;
  date: string;
  content: string;
  metrics: Record<string, string | number>;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `API request failed: ${response.status}`);
  }
  return response.json();
}

export async function askCookHero(
  message: string,
  user_id: string,
  enabled_tools?: string[]
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, user_id, sources: ["recipes", "personal"], enabled_tools }),
  });
  return parseResponse<ChatResponse>(response);
}

export async function listTools(): Promise<ToolCatalogItem[]> {
  const response = await fetch(`${API_BASE}/tools`);
  return parseResponse<ToolCatalogItem[]>(response);
}

export async function uploadPersonalDocument(payload: {
  user_id: string;
  title: string;
  content: string;
  category?: string;
  difficulty?: string;
}): Promise<PersonalDocumentResponse> {
  const response = await fetch(`${API_BASE}/knowledge/personal`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse<PersonalDocumentResponse>(response);
}

export async function analyzeFoodImage(payload: {
  image_url?: string;
  image_base64?: string;
  user_goal?: string;
}): Promise<VisionAnalyzeResponse> {
  const response = await fetch(`${API_BASE}/vision/food`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse<VisionAnalyzeResponse>(response);
}

export async function createDietPlan(payload: {
  user_id: string;
  goal: string;
  days: number;
  context_query?: string;
}): Promise<DietPlanResponse> {
  const response = await fetch(`${API_BASE}/diet/plans`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse<DietPlanResponse>(response);
}

export async function createMealCheckin(payload: {
  user_id: string;
  meal_time: string;
  description: string;
  image_url?: string;
  image_base64?: string;
  user_goal?: string;
  image_analysis?: MealImageAnalysis;
}): Promise<MealCheckinResponse> {
  const response = await fetch(`${API_BASE}/checkins`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse<MealCheckinResponse>(response);
}

export async function analyzeNutrition(payload: {
  user_id: string;
  date: string;
}): Promise<NutritionAnalysisResponse> {
  const response = await fetch(`${API_BASE}/nutrition/analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse<NutritionAnalysisResponse>(response);
}
