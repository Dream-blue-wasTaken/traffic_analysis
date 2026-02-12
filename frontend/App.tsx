import React, { useState, useRef, useEffect } from 'react';
import UploadZone from './components/UploadZone';
import LoadingSpinner from './components/LoadingSpinner';
import DetectionOverlay from './components/DetectionOverlay';
import { detectHelmets, DetectionResponse, Violation } from './services/yoloService';

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

  const noHelmetCount = results?.violations.filter(v => v.type === 'no_helmet').length || 0;
  const tripleRidingCount = results?.violations.filter(v => v.type === 'triple_riding').length || 0;
  const totalViolations = noHelmetCount + tripleRidingCount;

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
            Real-time traffic violation detection. Upload an image to detect helmet violations and triple-riding on motorcycles.
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
                        persons={results.persons}
                        motorcycles={results.motorcycles}
                        violations={results.violations}
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
                        {results.persons?.length || 0} Persons · {results.motorcycles?.length || 0} Motorcycles
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

              {/* ── Violations Panel ── */}
              {results && results.violations && results.violations.length > 0 && (
                <div className="bg-slate-900/80 border border-red-500/20 rounded-3xl p-6 shadow-2xl">
                  <div className="flex items-center gap-3 mb-5">
                    <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center text-xl">
                      🚨
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">
                        {totalViolations} Violation{totalViolations !== 1 ? 's' : ''} Detected
                      </h2>
                      <p className="text-sm text-slate-400">Traffic safety violations found in this image</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {noHelmetCount > 0 && (
                      <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-4 flex items-start gap-3">
                        <div className="text-2xl mt-0.5">🪖</div>
                        <div>
                          <p className="text-red-400 font-bold text-lg">{noHelmetCount}× No Helmet</p>
                          <p className="text-slate-400 text-sm mt-1">
                            {noHelmetCount} rider{noHelmetCount !== 1 ? 's' : ''} detected without helmet on motorcycle
                          </p>
                        </div>
                      </div>
                    )}

                    {tripleRidingCount > 0 && (
                      <div className="bg-orange-500/10 border border-orange-500/20 rounded-2xl p-4 flex items-start gap-3">
                        <div className="text-2xl mt-0.5">🏍️</div>
                        <div>
                          <p className="text-orange-400 font-bold text-lg">{tripleRidingCount}× Triple Riding</p>
                          <p className="text-slate-400 text-sm mt-1">
                            More than 2 persons detected on {tripleRidingCount} motorcycle{tripleRidingCount !== 1 ? 's' : ''}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Individual violation details */}
                  <div className="mt-4 space-y-2">
                    {results.violations.map((v: Violation, i: number) => (
                      <div
                        key={i}
                        className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${v.type === 'no_helmet'
                            ? 'bg-red-500/5 border-red-500/10'
                            : 'bg-orange-500/5 border-orange-500/10'
                          }`}
                      >
                        <span className="text-lg">{v.type === 'no_helmet' ? '❌' : '⚠️'}</span>
                        <span className={`font-medium ${v.type === 'no_helmet' ? 'text-red-300' : 'text-orange-300'
                          }`}>
                          {v.description}
                        </span>
                        <span className={`ml-auto text-xs px-2 py-0.5 rounded-full font-bold uppercase tracking-wider ${v.severity === 'high'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-yellow-500/20 text-yellow-400'
                          }`}>
                          {v.severity}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── No Violations ── */}
              {results && results.violations && results.violations.length === 0 && results.persons && results.persons.length > 0 && (
                <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-3xl p-6 text-center">
                  <div className="text-4xl mb-2">✅</div>
                  <h2 className="text-xl font-bold text-emerald-400">No Violations Detected</h2>
                  <p className="text-slate-400 text-sm mt-1">All detected persons appear to be following traffic safety rules</p>
                </div>
              )}

              {/* ── Detection Details Grid ── */}
              {results && results.persons && results.persons.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {results.persons.map((person) => (
                    <div key={`p-${person.id}`} className="bg-slate-900/50 border border-white/5 p-4 rounded-2xl flex items-center justify-between">
                      <div>
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-tighter">
                          {person.on_motorcycle ? 'Rider' : 'Person'}
                        </span>
                        <p className={`font-semibold capitalize ${person.helmet_status === 'helmet'
                            ? 'text-emerald-400'
                            : person.helmet_status === 'no_helmet'
                              ? 'text-red-400'
                              : 'text-slate-300'
                          }`}>
                          {person.helmet_status === 'helmet' ? '✅ Helmet' :
                            person.helmet_status === 'no_helmet' ? '❌ No Helmet' :
                              '⚠️ Unknown'}
                        </p>
                      </div>
                      <div className="text-right">
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-tighter">Confidence</span>
                        <p className="text-indigo-400 font-mono font-bold">{Math.round(person.confidence * 100)}%</p>
                      </div>
                    </div>
                  ))}
                  {results.motorcycles && results.motorcycles.map((moto) => (
                    <div key={`m-${moto.id}`} className="bg-slate-900/50 border border-white/5 p-4 rounded-2xl flex items-center justify-between">
                      <div>
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-tighter">Motorcycle</span>
                        <p className={`font-semibold ${moto.rider_count > 2 ? 'text-orange-400' : 'text-blue-400'
                          }`}>
                          🏍️ {moto.rider_count} Rider{moto.rider_count !== 1 ? 's' : ''}
                        </p>
                      </div>
                      <div className="text-right">
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-tighter">Confidence</span>
                        <p className="text-indigo-400 font-mono font-bold">{Math.round(moto.confidence * 100)}%</p>
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
