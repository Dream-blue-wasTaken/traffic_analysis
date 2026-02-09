export interface AnalysisResult {
  object: string;
  people_count: number;
  helmet: 'yes' | 'no' | 'unknown';
  provider?: string;
}

export enum AnalysisStatus {
  IDLE = 'IDLE',
  ANALYZING = 'ANALYZING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR',
}

export interface AnalysisState {
  status: AnalysisStatus;
  result: AnalysisResult | null;
  error: string | null;
  imagePreview: string | null;
}