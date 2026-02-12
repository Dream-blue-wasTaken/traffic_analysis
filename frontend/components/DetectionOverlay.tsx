import React from 'react';
import { Detection, PersonDetection, MotorcycleDetection, Violation } from '../services/yoloService';

interface DetectionOverlayProps {
  detections: Detection[];
  persons?: PersonDetection[];
  motorcycles?: MotorcycleDetection[];
  violations?: Violation[];
  imageWidth: number;
  imageHeight: number;
  containerWidth: number;
  containerHeight: number;
}

const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  detections,
  persons = [],
  motorcycles = [],
  violations = [],
  imageWidth,
  imageHeight,
  containerWidth,
  containerHeight,
}) => {
  const scaleX = containerWidth / imageWidth;
  const scaleY = containerHeight / imageHeight;

  const renderBox = (
    box: [number, number, number, number],
    label: string,
    color: string,
    key: string,
    dashed: boolean = false,
    thick: boolean = false,
  ) => {
    const [x1, y1, x2, y2] = box;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;
    const left = x1 * scaleX;
    const top = y1 * scaleY;

    return (
      <div
        key={key}
        style={{
          position: 'absolute',
          border: `${thick ? 3 : 2}px ${dashed ? 'dashed' : 'solid'} ${color}`,
          left: `${left}px`,
          top: `${top}px`,
          width: `${width}px`,
          height: `${height}px`,
          boxSizing: 'border-box',
          borderRadius: '2px',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: '-22px',
            left: '-2px',
            backgroundColor: color,
            color: 'white',
            padding: '1px 6px',
            fontSize: '11px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            borderRadius: '3px 3px 0 0',
            letterSpacing: '0.3px',
          }}
        >
          {label}
        </div>
      </div>
    );
  };

  // Set of motorcycle IDs that have triple-riding violations
  const tripleRidingMotoIds = new Set(
    violations.filter(v => v.type === 'triple_riding').map(v => v.motorcycle_id)
  );

  // Set of person IDs that have no-helmet violations
  const noHelmetPersonIds = new Set(
    violations.filter(v => v.type === 'no_helmet').map(v => v.person_id)
  );

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
      {/* Render motorcycle boxes */}
      {motorcycles.map((moto) => {
        const isTripleRiding = tripleRidingMotoIds.has(moto.id);
        const color = isTripleRiding ? '#f97316' : '#3b82f6'; // orange for violation, blue otherwise
        const label = isTripleRiding
          ? `🏍️ ${moto.rider_count} Riders ⚠️`
          : `🏍️ Motorcycle (${moto.rider_count} rider${moto.rider_count !== 1 ? 's' : ''})`;

        return renderBox(
          moto.box as [number, number, number, number],
          label,
          color,
          `moto-${moto.id}`,
          false,
          isTripleRiding,
        );
      })}

      {/* Render person boxes */}
      {persons.map((person) => {
        const isNoHelmet = noHelmetPersonIds.has(person.id);
        const isOnBike = person.on_motorcycle;

        let color = '#6b7280'; // gray for pedestrians
        let label = `Person`;

        if (isOnBike) {
          if (person.helmet_status === 'helmet') {
            color = '#10b981'; // green
            label = `✅ Helmet`;
          } else if (person.helmet_status === 'no_helmet') {
            color = '#ef4444'; // red
            label = `❌ No Helmet`;
          } else {
            color = '#eab308'; // yellow for unknown
            label = `⚠️ Rider`;
          }
        } else {
          // Pedestrian — still show helmet status if detected
          if (person.helmet_status === 'helmet') {
            color = '#10b981';
            label = `Person ✅`;
          } else if (person.helmet_status === 'no_helmet') {
            color = '#f97316';
            label = `Person (no helmet)`;
          }
        }

        return renderBox(
          person.box as [number, number, number, number],
          `${label} ${Math.round(person.confidence * 100)}%`,
          color,
          `person-${person.id}`,
          !isOnBike, // dashed for pedestrians
          isNoHelmet, // thick for violations
        );
      })}

      {/* Fallback: render raw helmet detections if no persons data */}
      {persons.length === 0 && detections.map((det, index) => {
        const isNoHelmet = det.label.toLowerCase().includes('without') || det.label.toLowerCase().includes('no');
        const color = isNoHelmet ? '#ef4444' : '#10b981';

        return renderBox(
          det.box,
          `${det.label} ${Math.round(det.confidence * 100)}%`,
          color,
          `det-${index}`,
        );
      })}
    </div>
  );
};

export default DetectionOverlay;
