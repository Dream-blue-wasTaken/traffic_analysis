import React, { useState } from 'react';
import UploadZone from './components/UploadZone';
import ResultCard from './components/ResultCard';
import LoadingSpinner from './components/LoadingSpinner';
import { analyzeImage } from './services/geminiService';
import { AnalysisState, AnalysisStatus } from './types';

const App: React.FC = () => {
  const [state, setState] = useState<AnalysisState>({
    status: AnalysisStatus.IDLE,
    result: null,
    error: null,
    imagePreview: null,
  });

  const handleImageSelected = async (file: File) => {
    // Create local preview
    const objectUrl = URL.createObjectURL(file);
    
    setState({
      status: AnalysisStatus.ANALYZING,
      result: null,
      error: null,
      imagePreview: objectUrl,
    });

    try {
      // Call Analysis Service (OpenRouter -> Gemini)
      const result = await analyzeImage(file);
      
      setState(prev => ({
        ...prev,
        status: AnalysisStatus.SUCCESS,
        result: result,
      }));
    } catch (err: any) {
      setState(prev => ({
        ...prev,
        status: AnalysisStatus.ERROR,
        error: err.message || "Failed to analyze image. Please try again.",
      }));
    }
  };

  const resetAnalysis = () => {
    if (state.imagePreview) {
      URL.revokeObjectURL(state.imagePreview);
    }
    setState({
      status: AnalysisStatus.IDLE,
      result: null,
      error: null,
      imagePreview: null,
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      
      {/* Header */}
      <header className="bg-slate-900/50 backdrop-blur-md border-b border-white/5 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
              </svg>
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white">Vision<span className="text-indigo-500">Analytica</span></h1>
          </div>
          <div className="flex items-center space-x-4">
             <span className="text-xs font-medium px-2 py-1 bg-green-500/10 text-green-400 border border-green-500/20 rounded-md">
               AI Powered
             </span>
             <a href="#" className="text-sm font-medium text-slate-400 hover:text-indigo-400 transition-colors">Documentation</a>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column: Upload & Preview */}
          <div className="lg:col-span-7 space-y-6">
            
            {/* Intro Text */}
            <div className="mb-6">
              <h2 className="text-3xl font-bold text-white mb-3">AI Vision Analysis</h2>
              <p className="text-slate-400 text-lg leading-relaxed">
                Upload an image to detect objects, count people, and check for safety compliance instantly using Gemini and advanced vision models.
              </p>
            </div>

            {/* Upload Area */}
            {state.status === AnalysisStatus.IDLE && (
               <UploadZone onImageSelected={handleImageSelected} />
            )}

            {/* Image Preview */}
            {state.imagePreview && (
              <div className="relative group rounded-2xl overflow-hidden shadow-2xl ring-1 ring-white/10 bg-slate-900">
                <img 
                  src={state.imagePreview} 
                  alt="Preview" 
                  className="w-full h-auto max-h-[500px] object-contain opacity-90 group-hover:opacity-100 transition-opacity" 
                />
                
                {state.status !== AnalysisStatus.ANALYZING && (
                  <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-slate-950 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex justify-center">
                    <button 
                      onClick={resetAnalysis}
                      className="bg-white text-slate-950 px-4 py-2 rounded-lg font-medium shadow-lg hover:bg-slate-100 transition-colors text-sm"
                    >
                      Analyze Another Image
                    </button>
                  </div>
                )}
              </div>
            )}

             {/* Error Message */}
             {state.status === AnalysisStatus.ERROR && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex items-start space-x-3">
                <svg className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <div className="flex-1">
                  <h3 className="text-sm font-medium text-red-300">Analysis Failed</h3>
                  <p className="text-sm text-red-400/80 mt-1">{state.error}</p>
                  <button 
                    onClick={() => setState(prev => ({ ...prev, status: AnalysisStatus.IDLE, error: null }))}
                    className="text-sm text-red-400 font-medium hover:text-red-300 hover:underline mt-2"
                  >
                    Try again
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-5">
            <div className="sticky top-24">
              {state.status === AnalysisStatus.IDLE && (
                 <div className="bg-slate-900/50 rounded-2xl border border-dashed border-white/10 p-8 text-center h-full flex flex-col items-center justify-center min-h-[300px]">
                    <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mb-4">
                      <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                      </svg>
                    </div>
                    <p className="text-slate-300 font-medium">Results will appear here</p>
                    <p className="text-slate-500 text-sm mt-1">Upload an image to start analysis</p>
                 </div>
              )}

              {state.status === AnalysisStatus.ANALYZING && (
                <div className="bg-slate-900 rounded-2xl shadow-2xl shadow-black/50 border border-white/5 p-8 min-h-[300px] flex items-center justify-center">
                  <LoadingSpinner />
                </div>
              )}

              {state.status === AnalysisStatus.SUCCESS && state.result && (
                <ResultCard result={state.result} />
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
};

export default App;