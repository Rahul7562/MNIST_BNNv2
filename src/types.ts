export interface PredictionResult {
  digit: number;
  confidence: number;
  scores: number[];
  xnorOps: number;
  popcountOps: number;
  estimatedLatencyUs: number;
}

export interface PresetDigit {
  label: string;
  digit: number;
  data: number[];
}
