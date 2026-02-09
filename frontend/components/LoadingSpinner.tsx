import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="relative w-16 h-16">
        <div className="absolute top-0 left-0 w-full h-full border-4 border-indigo-200 rounded-full opacity-25"></div>
        <div className="absolute top-0 left-0 w-full h-full border-4 border-indigo-600 rounded-full border-t-transparent animate-spin"></div>
      </div>
      <h3 className="mt-4 text-lg font-medium text-white">Analyzing Image...</h3>
      <p className="text-slate-400 text-sm mt-1">Extracting objects and safety data</p>
    </div>
  );
};

export default LoadingSpinner;