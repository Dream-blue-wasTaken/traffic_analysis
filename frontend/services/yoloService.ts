export interface Detection {
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  label: string;
  class_id: number;
  person_id?: number;
}

export interface PersonDetection {
  id: number;
  box: [number, number, number, number];
  confidence: number;
  helmet_status: 'helmet' | 'no_helmet' | 'unknown';
  on_motorcycle: boolean;
  motorcycle_id: number | null;
}

export interface MotorcycleDetection {
  id: number;
  box: [number, number, number, number];
  confidence: number;
  rider_count: number;
  rider_ids: number[];
}

export interface Violation {
  type: 'no_helmet' | 'triple_riding';
  severity: string;
  description: string;
  person_box?: [number, number, number, number];
  motorcycle_box: [number, number, number, number];
  person_id?: number;
  motorcycle_id: number;
  rider_count?: number;
  person_boxes?: [number, number, number, number][];
  person_ids?: number[];
}

export interface DetectionResponse {
  detections: Detection[];
  persons: PersonDetection[];
  motorcycles: MotorcycleDetection[];
  violations: Violation[];
  image_size: {
    width: number;
    height: number;
  };
}

export const detectHelmets = async (file: File): Promise<DetectionResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Detection failed');
  }

  return response.json();
};
