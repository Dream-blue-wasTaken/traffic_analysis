import React, { useRef, useState } from 'react';

interface UploadZoneProps {
  onImageSelected: (file: File) => void;
  disabled?: boolean;
}

const UploadZone: React.FC<UploadZoneProps> = ({ onImageSelected, disabled }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndPassFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndPassFile(e.target.files[0]);
    }
  };

  const validateAndPassFile = (file: File) => {
    if (file.type.startsWith('image/')) {
      onImageSelected(file);
    } else {
      alert("Please upload a valid image file.");
    }
  };

  const handleClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        relative group cursor-pointer flex flex-col items-center justify-center 
        w-full h-64 rounded-2xl border-2 border-dashed transition-all duration-300
        ${disabled ? 'opacity-50 cursor-not-allowed bg-white/5 border-white/10' : ''}
        ${isDragging 
          ? 'border-indigo-500 bg-indigo-500/10 scale-[1.01] shadow-2xl shadow-indigo-500/10' 
          : 'border-white/10 bg-white/5 hover:border-white/20 hover:bg-white/[0.08] shadow-sm'
        }
      `}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="image/*"
        disabled={disabled}
      />
      
      <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center px-4">
        <div className={`mb-4 p-4 rounded-full transition-colors duration-300 ${isDragging ? 'bg-indigo-500/20 text-indigo-400' : 'bg-white/5 text-slate-500 group-hover:bg-white/10 group-hover:text-indigo-400'}`}>
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
          </svg>
        </div>
        <p className={`mb-2 text-sm font-medium ${isDragging ? 'text-indigo-400' : 'text-slate-300'}`}>
          <span className="font-semibold text-white">Click to upload</span> or drag and drop
        </p>
        <p className="text-xs text-slate-500">
          SVG, PNG, JPG or WEBP
        </p>
      </div>
    </div>
  );
};

export default UploadZone;