import React from 'react';
import { Detection } from '../services/yoloService';

interface DetectionOverlayProps {
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
  containerWidth: number;
  containerHeight: number;
}

const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  detections,
  imageWidth,
  imageHeight,
  containerWidth,
  containerHeight,
}) => {
  const scaleX = containerWidth / imageWidth;
  const scaleY = containerHeight / imageHeight;

  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    >
      {detections.map((det, index) => {
        const [x1, y1, x2, y2] = det.box;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;
        const left = x1 * scaleX;
        const top = y1 * scaleY;

        const isNoHelmet = det.label.toLowerCase().includes('without');
        const color = isNoHelmet ? '#ef4444' : '#10b981'; // Red for 'Without Helmet', Green otherwise

        return (
          <div
            key={index}
            style={{
              position: 'absolute',
              border: `2px solid ${color}`,
              left: `${left}px`,
              top: `${top}px`,
              width: `${width}px`,
              height: `${height}px`,
              boxSizing: 'border-box',
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: '-24px',
                left: '-2px',
                backgroundColor: color,
                color: 'white',
                padding: '2px 6px',
                fontSize: '12px',
                fontWeight: 'bold',
                whiteSpace: 'nowrap',
                borderRadius: '2px 2px 0 0',
              }}
            >
              {det.label} ({Math.round(det.confidence * 100)}%)
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default DetectionOverlay;
