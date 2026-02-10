import React, { useState, useRef, useEffect } from 'react';
import UploadZone from './components/UploadZone';
import LoadingSpinner from './components/LoadingSpinner';
import DetectionOverlay from './components/DetectionOverlay';
import { detectHelmets, DetectionResponse, Detection } from './services/yoloService';

const App: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const imageRef = useRef<HTMLImageElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateSize = () => {
      if (imageRef.current) {
        setContainerSize({
          width: imageRef.current.clientWidth,
          height: imageRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', updateSize);
    updateSize();
    return () => window.removeEventListener('resize', updateSize);
  }, [previewUrl, results]);

  const handleImageSelected = async (file: File) => {
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
    setIsAnalyzing(true);

    try {
      const data = await detectHelmets(file);
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred during analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const reset = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Background decoration */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-indigo-500/10 blur-[120px] rounded-full"></div>
        <div className="absolute -bottom-[10%] -right-[10%] w-[40%] h-[40%] bg-blue-500/10 blur-[120px] rounded-full"></div>
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-4 py-12">
        <header className="mb-16 text-center">
          <div className="inline-flex items-center px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-sm font-medium mb-4">
            <span className="relative flex h-2 w-2 mr-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
            Powered by YOLOv11
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-white mb-6 bg-clip-text text-transparent bg-gradient-to-b from-white to-white/60">
            VisionAnalytica
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Real-time industrial safety analysis. Upload an image to detect safety helmets and personal protective equipment.
          </p>
        </header>

        <main className="space-y-8">
          {!previewUrl ? (
            <div className="max-w-2xl mx-auto">
              <UploadZone onImageSelected={handleImageSelected} />
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-1 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
              <div className="bg-slate-900 rounded-3xl p-4 border border-white/5 shadow-2xl relative overflow-hidden group">
                <div className="relative aspect-auto max-h-[70vh] flex items-center justify-center bg-black/20 rounded-2xl overflow-hidden p-1">
                  <div className="relative inline-block">
                    <img
                      ref={imageRef}
                      src={previewUrl}
                      alt="Preview"
                      className="max-w-full max-h-[68vh] block rounded-xl shadow-lg"
                      onLoad={() => {
                        if (imageRef.current) {
                          setContainerSize({
                            width: imageRef.current.clientWidth,
                            height: imageRef.current.clientHeight,
                          });
                        }
                      }}
                    />
                    
                    {results && containerSize.width > 0 && (
                      <DetectionOverlay
                        detections={results.detections}
                        imageWidth={results.image_size.width}
                        imageHeight={results.image_size.height}
                        containerWidth={containerSize.width}
                        containerHeight={containerSize.height}
                      />
                    )}

                    {isAnalyzing && (
                      <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-[2px] rounded-xl flex flex-col items-center justify-center">
                        <LoadingSpinner />
                        <p className="mt-4 text-indigo-300 font-medium animate-pulse">Analyzing...</p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-6 flex items-center justify-between px-2">
                  <div className="flex gap-4">
                    <button
                      onClick={reset}
                      className="px-6 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-white font-medium transition-all border border-white/10"
                    >
                      New Analysis
                    </button>
                  </div>
                  
                  {results && (
                    <div className="text-right">
                      <p className="text-sm text-slate-500 uppercase tracking-widest font-bold">Detection Summary</p>
                      <p className="text-white font-medium">
                        {results.detections.length} Probable Objects Detected
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {error && (
                <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-center animate-in shake duration-500">
                  {error}
                </div>
              )}
              
              {results && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {results.detections.map((det, i) => (
                    <div key={i} className="bg-slate-900/50 border border-white/5 p-4 rounded-2xl flex items-center justify-between">
                      <div>
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-tighter">Object</span>
                        <p className="text-white font-semibold capitalize">{det.label}</p>
                      </div>
                      <div className="text-right">
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-tighter">Confidence</span>
                        <p className="text-indigo-400 font-mono font-bold">{Math.round(det.confidence * 100)}%</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </main>

        <footer className="mt-24 text-center text-slate-600 text-sm">
          <p>© 2024 VisionAnalytica AI. All Rights Reserved.</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
