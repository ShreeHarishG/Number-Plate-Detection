"use client";

import { useState, useRef, useEffect } from "react";

type Plate = {
  confidence: number;
  bbox: number[];
  text: string;
  ocr_confidence: number;
};

type Vehicle = {
  vehicle_type: string;
  vehicle_confidence: number;
  vehicle_bbox: number[];
  plate: Plate | null;
};

type DetectionResult = {
  success: boolean;
  vehicle_count: number;
  vehicles: Vehicle[];
  annotated_image: string;
};

type HistoryItem = {
  id: number;
  fileName: string;
  originalImage: string;
  result: DetectionResult;
};

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string>("");
  const [progress, setProgress] = useState<number>(0);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [backendStatus, setBackendStatus] = useState<"healthy" | "offline" | "checking">("checking");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Check backend health
    fetch("http://localhost:8000/health")
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "healthy") {
          setBackendStatus("healthy");
        } else {
          setBackendStatus("offline");
        }
      })
      .catch(() => setBackendStatus("offline"));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setSelectedImage(URL.createObjectURL(file));
      setResult(null); // Clear previous result
    }
  };

  const processImage = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setResult(null);
    setProgress(10);
    setLoadingStep("Uploading image...");

    // Simulate loading steps for polish
    const steps = [
      "Detecting vehicles...",
      "Detecting license plates...",
      "Reading plate text...",
    ];
    let stepIndex = 0;
    const stepInterval = setInterval(() => {
      if (stepIndex < steps.length) {
        setLoadingStep(steps[stepIndex]);
        setProgress(10 + ((stepIndex + 1) / steps.length) * 80); // Progress up to 90%
        stepIndex++;
      }
    }, 1200); // Change step every 1.2s

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/detect", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Detection failed");
      }

      const data: DetectionResult = await response.json();
      
      clearInterval(stepInterval);
      setProgress(100);
      setLoadingStep("Complete");
      
      setTimeout(() => {
        setResult(data);
        setIsProcessing(false);
        setLoadingStep("");
        setProgress(0);

        // Add to history
        if (data.success) {
          setHistory((prev) => [
            {
              id: Date.now(),
              fileName: selectedFile.name,
              originalImage: URL.createObjectURL(selectedFile),
              result: data,
            },
            ...prev,
          ]);
        }
      }, 500);

    } catch (error) {
      console.error(error);
      clearInterval(stepInterval);
      setIsProcessing(false);
      setLoadingStep("");
      setProgress(0);
      alert("Error connecting to backend");
    }
  };

  const validPlatesCount = result?.vehicles.filter(v => v.plate && v.plate.text).length || 0;

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#fafafa] font-sans pb-10 flex flex-col relative">
      {/* Header */}
      <header className="bg-[#171717] border-b border-[#262626] py-6 px-8 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Vehicle & License Plate Detection</h1>
            <p className="text-[#a3a3a3] mt-1 text-sm">AI-powered vehicle detection and number plate OCR</p>
          </div>
          <div className="flex items-center space-x-2 bg-[#0a0a0a] border border-[#262626] px-4 py-2 rounded-full text-sm font-medium">
            {backendStatus === "healthy" ? (
              <>
                <span className="w-3 h-3 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span>
                <span className="text-green-400">AI Engine Online</span>
              </>
            ) : backendStatus === "offline" ? (
              <>
                <span className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]"></span>
                <span className="text-red-400">Backend Offline</span>
              </>
            ) : (
              <>
                <span className="w-3 h-3 rounded-full bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.6)]"></span>
                <span className="text-yellow-400">Checking...</span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto mt-8 px-4 sm:px-6 lg:px-8 flex-grow w-full">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Input Section */}
          <div className="flex flex-col h-full">
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#f59e0b] mb-3 flex items-center">
              <span className="w-2 h-2 rounded-full bg-[#f59e0b] mr-2"></span>
              Input Image
            </h2>
            <div className="bg-[#171717] rounded-xl border border-[#262626] p-6 flex flex-col items-center justify-center min-h-[450px] shadow-lg relative overflow-hidden flex-grow transition-all hover:border-[#f59e0b]/30">
              
              {!selectedImage ? (
                <div className="text-center w-full">
                  <div className="mx-auto h-20 w-20 bg-[#262626] rounded-full flex items-center justify-center text-[#f59e0b] mb-6 shadow-inner">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" className="w-10 h-10">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <p className="text-[#fafafa] font-medium text-lg">Select an image to process</p>
                  <p className="text-sm text-[#a3a3a3] mt-2 mb-8">Supports JPG, PNG</p>
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-gradient-to-r from-[#f59e0b] to-[#ea580c] text-white px-8 py-3 rounded-lg font-bold hover:shadow-[0_0_15px_rgba(245,158,11,0.4)] transition-all transform hover:-translate-y-0.5"
                  >
                    Upload Image
                  </button>
                </div>
              ) : (
                <div className="w-full h-full flex flex-col">
                  <div className="relative flex-grow bg-[#0a0a0a] rounded-lg overflow-hidden flex items-center justify-center border border-[#262626]">
                    <img src={selectedImage} alt="Input" className="max-w-full max-h-[350px] object-contain drop-shadow-2xl" />
                  </div>
                  <div className="mt-5 flex justify-between items-center bg-[#0a0a0a] p-4 rounded-lg border border-[#262626]">
                    <span className="font-mono text-sm text-[#a3a3a3] truncate max-w-[200px]">{selectedFile?.name}</span>
                    <div className="space-x-4">
                       <button 
                        onClick={() => {
                          setSelectedImage(null);
                          setSelectedFile(null);
                          setResult(null);
                        }}
                        className="text-sm text-[#a3a3a3] hover:text-red-500 font-medium transition-colors"
                      >
                        Clear
                      </button>
                      <button 
                        onClick={processImage}
                        disabled={isProcessing}
                        className={`px-6 py-2.5 rounded-lg font-bold text-white transition-all shadow-md ${isProcessing ? 'bg-[#262626] text-[#a3a3a3] cursor-not-allowed' : 'bg-gradient-to-r from-[#f59e0b] to-[#ea580c] hover:shadow-[0_0_15px_rgba(245,158,11,0.4)]'}`}
                      >
                        {isProcessing ? 'Processing...' : 'Run Detection'}
                      </button>
                    </div>
                  </div>
                </div>
              )}
              
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                className="hidden" 
                accept="image/jpeg, image/png, image/jpg"
              />
            </div>
          </div>

          {/* Output Section */}
          <div className="flex flex-col h-full">
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#f59e0b] mb-3 flex items-center">
              <span className="w-2 h-2 rounded-full bg-[#f59e0b] mr-2"></span>
              Detection Result
            </h2>
            <div className="bg-[#171717] rounded-xl border border-[#262626] p-6 flex flex-col min-h-[450px] shadow-lg flex-grow transition-all hover:border-[#f59e0b]/30">
              
              {!result ? (
                <div className="flex-grow flex flex-col items-center justify-center text-[#a3a3a3]">
                   <svg className="w-12 h-12 mb-4 text-[#262626]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                     <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                   </svg>
                   <p className="font-medium">Waiting for input...</p>
                </div>
              ) : (
                <div className="flex flex-col h-full space-y-5">
                  {/* Annotated Image */}
                  <div className="bg-[#0a0a0a] rounded-lg p-2 flex items-center justify-center border border-[#262626] shadow-inner relative group overflow-hidden">
                    <img 
                      src={result.annotated_image} 
                      alt="Result" 
                      className="max-w-full max-h-[300px] object-contain rounded" 
                    />
                    {/* Subtle scanning line effect */}
                    <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#f59e0b]/10 to-transparent h-[150%] animate-[scan_3s_ease-in-out_infinite] opacity-50 pointer-events-none"></div>
                  </div>
                  
                  {/* Summary Stats */}
                  <div className="flex justify-between items-center bg-[#0a0a0a] border border-[#262626] rounded-lg p-3">
                    <div className="flex items-center space-x-2">
                      <span className="text-[#a3a3a3] text-sm uppercase tracking-wide">Vehicles</span>
                      <span className="font-bold text-xl text-[#f59e0b] bg-[#f59e0b]/10 px-3 py-0.5 rounded">{result.vehicle_count}</span>
                    </div>
                    <div className="w-px h-6 bg-[#262626]"></div>
                    <div className="flex items-center space-x-2">
                      <span className="text-[#a3a3a3] text-sm uppercase tracking-wide">Plates</span>
                      <span className="font-bold text-xl text-[#f59e0b] bg-[#f59e0b]/10 px-3 py-0.5 rounded">{validPlatesCount}</span>
                    </div>
                  </div>

                  {/* Plate Cards */}
                  <div className="space-y-3 overflow-y-auto max-h-[160px] pr-2 custom-scrollbar">
                    {result.vehicles.map((v, i) => {
                      if (!v.plate || !v.plate.text) return null;
                      return (
                        <div key={i} className="bg-[#0a0a0a] p-4 rounded-lg border border-[#f59e0b]/30 flex justify-between items-center relative overflow-hidden group hover:border-[#f59e0b] transition-colors">
                          <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-[#f59e0b] to-[#ea580c]"></div>
                          <div className="pl-3">
                            <div className="font-mono font-bold text-xl tracking-widest text-white">{v.plate.text}</div>
                            <div className="text-xs text-[#a3a3a3] uppercase mt-1 flex items-center">
                              <span className="bg-[#262626] px-2 py-0.5 rounded text-white mr-2">{v.vehicle_type}</span>
                              OCR Conf: {(v.plate.ocr_confidence * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div className="bg-[#f59e0b]/10 text-[#f59e0b] px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider flex items-center border border-[#f59e0b]/20">
                            <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                            Match
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {/* Action Button */}
                  <div className="pt-4 mt-auto border-t border-[#262626]">
                    <button 
                      onClick={() => {
                        setResult(null);
                        setSelectedImage(null);
                        setSelectedFile(null);
                        fileInputRef.current?.click();
                      }}
                      className="w-full bg-gradient-to-r from-[#f59e0b] to-[#ea580c] text-white py-3 rounded-lg font-bold hover:shadow-[0_0_15px_rgba(245,158,11,0.4)] transition-all transform hover:-translate-y-0.5 flex justify-center items-center"
                    >
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      Process Another Image
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* History Section */}
        {history.length > 0 && (
          <div className="mt-12 mb-8">
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#f59e0b] mb-4 flex items-center">
              <span className="w-2 h-2 rounded-full bg-[#f59e0b] mr-2"></span>
              Session History
            </h2>
            <div className="bg-[#171717] rounded-xl border border-[#262626] overflow-hidden shadow-lg">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-[#0a0a0a] border-b border-[#262626]">
                    <th className="py-4 px-6 font-semibold text-xs uppercase tracking-wider text-[#a3a3a3]">Image Source</th>
                    <th className="py-4 px-6 font-semibold text-xs uppercase tracking-wider text-[#a3a3a3]">Detected Plates</th>
                    <th className="py-4 px-6 font-semibold text-xs uppercase tracking-wider text-[#a3a3a3]">Vehicle Count</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#262626]">
                  {history.map((item) => (
                    <tr key={item.id} className="hover:bg-[#262626]/40 transition-colors">
                      <td className="py-4 px-6 flex items-center space-x-4">
                        <div className="w-14 h-14 rounded-md overflow-hidden bg-[#0a0a0a] border border-[#262626] flex items-center justify-center p-1">
                          <img src={item.originalImage} className="max-w-full max-h-full object-contain rounded-sm" alt="thumbnail" />
                        </div>
                        <span className="font-mono text-sm text-[#fafafa] truncate max-w-[200px]">{item.fileName}</span>
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex flex-wrap gap-2">
                          {item.result.vehicles.filter(v => v.plate?.text).length === 0 ? (
                            <span className="text-sm text-[#a3a3a3] italic">No plates found</span>
                          ) : (
                            item.result.vehicles.map((v, idx) => v.plate?.text && (
                              <span key={idx} className="bg-[#f59e0b] text-[#0a0a0a] px-3 py-1 rounded text-xs font-mono font-bold tracking-wider shadow-sm">
                                {v.plate.text}
                              </span>
                            ))
                          )}
                        </div>
                      </td>
                      <td className="py-4 px-6 text-sm text-[#a3a3a3]">
                        <span className="font-bold text-[#fafafa] mr-1">{item.result.vehicle_count}</span>
                        {item.result.vehicle_count === 1 ? 'vehicle' : 'vehicles'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      {/* Full-screen Loading Overlay */}
      {isProcessing && (
        <div className="fixed inset-0 bg-[#0a0a0a]/90 backdrop-blur-md flex flex-col items-center justify-center z-50 transition-all">
          <div className="max-w-md w-full px-6 flex flex-col items-center">
            
            <div className="relative mb-8">
              <div className="animate-spin rounded-full h-20 w-20 border-t-2 border-b-2 border-[#f59e0b]"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="h-3 w-3 bg-[#f59e0b] rounded-full animate-ping"></div>
              </div>
            </div>

            <h3 className="text-[#f59e0b] font-bold text-2xl mb-2 text-center tracking-wide">{loadingStep}</h3>
            
            {/* Progress Bar Container */}
            <div className="w-full h-4 bg-[#262626] rounded-full overflow-hidden mt-6 mb-2 shadow-inner border border-[#3f3f46]">
              <div 
                className="h-full bg-gradient-to-r from-[#f59e0b] to-[#ea580c] transition-all duration-300 ease-out relative"
                style={{ width: `${progress}%` }}
              >
                {/* Shine effect on progress bar */}
                <div className="absolute top-0 left-0 right-0 bottom-0 bg-white/20 animate-[scan_2s_ease-in-out_infinite]"></div>
              </div>
            </div>
            
            <div className="flex justify-between w-full text-[#a3a3a3] text-sm font-mono mt-2">
              <span>Processing...</span>
              <span>{Math.round(progress)}%</span>
            </div>

          </div>
        </div>
      )}
      
      {/* Add global styles for animations */}
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes scan {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100%); }
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #0a0a0a;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #262626;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #3f3f46;
        }
      `}} />
    </div>
  );
}
