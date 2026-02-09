export interface Detection {
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  label: string;
  class_id: number;
}

export interface DetectionResponse {
  detections: Detection[];
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
