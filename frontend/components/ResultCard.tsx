import React from 'react';
import { AnalysisResult } from '../types';

interface ResultCardProps {
  result: AnalysisResult;
}

const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  const getHelmetColor = (status: string) => {
    switch (status) {
      case 'yes': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'no': return 'bg-red-500/10 text-red-400 border-red-500/20';
      default: return 'bg-slate-500/10 text-slate-400 border-slate-500/20';
    }
  };

  const getHelmetIcon = (status: string) => {
    switch (status) {
      case 'yes': 
        return (
          <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
        );
      case 'no':
        return (
          <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
        );
      default:
        return (
          <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
        );
    }
  };

  return (
    <div className="bg-slate-900 overflow-hidden shadow-2xl rounded-2xl border border-white/5">
      <div className="px-6 py-5 border-b border-white/5 bg-white/5 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Analysis Results</h3>
        <div className="flex items-center space-x-2">
          {result.provider && (
            <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${
              result.provider === 'Gemini' 
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' 
                : 'bg-purple-500/10 text-purple-400 border-purple-500/20'
            }`}>
              {result.provider}
            </span>
          )}
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            JSON Output
          </span>
        </div>
      </div>
      <div className="px-6 py-6 space-y-6">
        
        {/* Main Object */}
        <div>
          <label className="block text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Detected Object</label>
          <div className="text-xl font-bold text-white capitalize flex items-center">
            <svg className="w-5 h-5 mr-2 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
            </svg>
            {result.object}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* People Count */}
          <div>
            <label className="block text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">People Count</label>
            <div className="flex items-baseline">
              <span className="text-3xl font-extrabold text-white">{result.people_count}</span>
              <span className="ml-2 text-sm text-slate-400">detected</span>
            </div>
          </div>

          {/* Helmet Status */}
          <div>
            <label className="block text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Safety Helmet</label>
            <div className={`inline-flex items-center px-3 py-1.5 rounded-lg border ${getHelmetColor(result.helmet)}`}>
              {getHelmetIcon(result.helmet)}
              <span className="font-bold text-sm uppercase">{result.helmet}</span>
            </div>
          </div>
        </div>

        {/* JSON Preview Code Block */}
        <div className="mt-4">
          <label className="block text-xs font-medium text-slate-500 mb-2">RAW JSON API RESPONSE</label>
          <div className="bg-black/40 rounded-lg p-4 overflow-x-auto border border-white/5">
            <pre className="text-xs text-green-400/90 font-mono">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        </div>

      </div>
    </div>
  );
};

export default ResultCard;