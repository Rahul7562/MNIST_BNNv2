import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Cpu, 
  RotateCcw, 
  Sparkles, 
  Layers, 
  Activity, 
  CheckCircle2, 
  Info, 
  Sliders, 
  Zap, 
  Terminal, 
  Key, 
  ExternalLink,
  Pencil,
  Trash2,
  BarChart3
} from 'lucide-react';
import { GoogleGenAI } from '@google/genai';
import { runBnnInference, generatePresetDigits } from './utils/bnnInference';
import { PredictionResult, PresetDigit } from './types';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [pixelData, setPixelData] = useState<number[]>(() => new Array(784).fill(0));
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [activeTab, setActiveTab] = useState<'inference' | 'fpga' | 'quickstart'>('inference');
  
  const [geminiApiKey, setGeminiApiKey] = useState(() => {
    return (typeof process !== 'undefined' && process.env?.GEMINI_API_KEY) || 
           ((import.meta as any).env?.VITE_GEMINI_API_KEY || '');
  });
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [geminiAnalysis, setGeminiAnalysis] = useState<string | null>(null);

  const presets = useRef<PresetDigit[]>(generatePresetDigits());

  useEffect(() => {
    const result = runBnnInference(pixelData);
    setPrediction(result);
  }, [pixelData]);

  const updatePixelsFromCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    const blockSize = Math.floor(canvas.width / 28);
    const newPixels = new Array(784).fill(0);

    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        let sum = 0;
        let count = 0;
        for (let br = 0; br < blockSize; br++) {
          for (let bc = 0; bc < blockSize; bc++) {
            const px = (r * blockSize + br) * canvas.width + (c * blockSize + bc);
            const rVal = data[px * 4];
            sum += rVal;
            count++;
          }
        }
        newPixels[r * 28 + c] = Math.min(1, sum / (count * 255));
      }
    }
    setPixelData(newPixels);
    setGeminiAnalysis(null);
  }, []);

  const handleClear = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPixelData(new Array(784).fill(0));
    setGeminiAnalysis(null);
  };

  const handleLoadPreset = (digit: PresetDigit) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const blockSize = Math.floor(canvas.width / 28);
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const val = digit.data[r * 28 + c];
        if (val > 0.05) {
          const colorVal = Math.round(val * 255);
          ctx.fillStyle = `rgb(${colorVal}, ${colorVal}, ${colorVal})`;
          ctx.fillRect(c * blockSize, r * blockSize, blockSize, blockSize);
        }
      }
    }
    setPixelData(digit.data);
    setGeminiAnalysis(null);
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      updatePixelsFromCanvas();
    }
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing && e.type !== 'mousedown' && e.type !== 'touchstart') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX = 0;
    let clientY = 0;

    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(x, y, 16, 0, Math.PI * 2);
    ctx.fill();
    updatePixelsFromCanvas();
  };

  useEffect(() => {
    handleClear();
    if (presets.current[7]) {
      handleLoadPreset(presets.current[7]);
    }
  }, []);

  const handleGeminiAnalyze = async () => {
    if (!prediction) return;
    const apiKey = geminiApiKey || ((import.meta as any).env?.VITE_GEMINI_API_KEY || '');
    if (!apiKey) {
      setShowApiKeyModal(true);
      return;
    }

    setIsAnalyzing(true);
    setGeminiAnalysis(null);

    try {
      const ai = new GoogleGenAI({ apiKey });
      const prompt = `You are an expert AI & neuromorphic hardware engineer evaluating a Binarized Neural Network (BNN) classifying handwritten digits from the MNIST dataset.
The current BNN architecture is 784 -> 256 -> 256 -> 10, running 1-bit XNOR and popcount logic mapped to an FPGA (Zedboard XC7Z020).
Current BNN Classification:
- Predicted Digit: ${prediction.digit}
- Confidence: ${(prediction.confidence * 100).toFixed(1)}%
- Probability distribution across 0-9: ${prediction.scores.map((s, d) => `${d}: ${(s * 100).toFixed(1)}%`).join(', ')}

Please provide:
1. Visual & topological analysis of the digit (loops, stroke balance, ascenders).
2. Assessment of why a 1-bit quantized neural network predicted this digit.
3. Any potential classification ambiguity (e.g. looks slightly like another digit).
Keep your analysis structured, clear, and insightful.`;

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: [prompt],
      });

      setGeminiAnalysis(response.text || 'Analysis completed.');
    } catch (err: any) {
      setGeminiAnalysis(`Gemini API Error: ${err.message || String(err)}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 py-4 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-20">
        <div className="flex items-center space-x-3">
          <div className="bg-indigo-600 p-2.5 rounded-xl shadow-lg shadow-indigo-500/20">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight text-white">MNIST BNN</h1>
              <span className="text-xs bg-indigo-500/20 text-indigo-300 font-semibold px-2 py-0.5 rounded-full border border-indigo-500/30">
                Google AI Studio Ready
              </span>
            </div>
            <p className="text-xs text-slate-400">
              784 → 256 → 256 → 10 Binarized Neural Network • FPGA Zedboard HW/SW Co-Design
            </p>
          </div>
        </div>

        <div className="flex items-center bg-slate-800/80 p-1 rounded-xl border border-slate-700/60 text-sm">
          <button
            onClick={() => setActiveTab('inference')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'inference'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Pencil className="w-4 h-4" />
            <span>Classifier</span>
          </button>
          <button
            onClick={() => setActiveTab('fpga')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'fpga'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Cpu className="w-4 h-4" />
            <span>FPGA RTL</span>
          </button>
          <button
            onClick={() => setActiveTab('quickstart')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg font-medium transition ${
              activeTab === 'quickstart'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Terminal className="w-4 h-4" />
            <span>CLI &amp; Vivado</span>
          </button>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowApiKeyModal(true)}
            className="flex items-center space-x-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-2 rounded-lg border border-slate-700 transition"
          >
            <Key className="w-3.5 h-3.5 text-indigo-400" />
            <span>{geminiApiKey ? 'Gemini Key Configured' : 'Configure Gemini Key'}</span>
          </button>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6">
        {activeTab === 'inference' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5 flex flex-col space-y-4">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                      Digit Canvas (28 × 28)
                    </h2>
                  </div>
                  <button
                    onClick={handleClear}
                    className="flex items-center space-x-1 text-xs text-rose-400 hover:text-rose-300 transition bg-rose-500/10 hover:bg-rose-500/20 px-2.5 py-1.5 rounded-lg border border-rose-500/20"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear</span>
                  </button>
                </div>

                <div className="relative mx-auto bg-black rounded-xl overflow-hidden border-2 border-slate-700 shadow-inner flex justify-center items-center cursor-crosshair">
                  <canvas
                    ref={canvasRef}
                    width={280}
                    height={280}
                    onMouseDown={startDrawing}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onMouseMove={draw}
                    onTouchStart={startDrawing}
                    onTouchEnd={stopDrawing}
                    onTouchMove={draw}
                    className="touch-none block"
                  />
                </div>

                <p className="text-xs text-slate-400 text-center mt-3">
                  Draw any single digit (0–9) above, or select a preset below.
                </p>

                <div className="mt-4 pt-4 border-t border-slate-800">
                  <span className="text-xs font-semibold text-slate-400 mb-2 block">
                    MNIST Presets:
                  </span>
                  <div className="grid grid-cols-5 gap-2">
                    {presets.current.map((preset) => (
                      <button
                        key={preset.digit}
                        onClick={() => handleLoadPreset(preset)}
                        className="py-2 bg-slate-800 hover:bg-indigo-600 text-slate-200 hover:text-white rounded-lg text-sm font-semibold transition border border-slate-700 hover:border-indigo-500"
                      >
                        {preset.digit}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl text-xs space-y-2.5">
                <div className="flex items-center space-x-2 text-indigo-400 font-semibold">
                  <Zap className="w-4 h-4" />
                  <span>FPGA Hardware Simulation Metrics</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-slate-300">
                  <div className="bg-slate-800/60 p-2.5 rounded-xl border border-slate-800">
                    <span className="text-slate-400 block text-[10px] uppercase">Total XNOR Ops</span>
                    <span className="text-sm font-mono font-bold text-white">268,800</span>
                  </div>
                  <div className="bg-slate-800/60 p-2.5 rounded-xl border border-slate-800">
                    <span className="text-slate-400 block text-[10px] uppercase">DSP Multipliers</span>
                    <span className="text-sm font-mono font-bold text-emerald-400">0 (100% Bitwise)</span>
                  </div>
                  <div className="bg-slate-800/60 p-2.5 rounded-xl border border-slate-800">
                    <span className="text-slate-400 block text-[10px] uppercase">Popcount Tree Ops</span>
                    <span className="text-sm font-mono font-bold text-white">522 ops</span>
                  </div>
                  <div className="bg-slate-800/60 p-2.5 rounded-xl border border-slate-800">
                    <span className="text-slate-400 block text-[10px] uppercase">HW/SW Agreement</span>
                    <span className="text-sm font-mono font-bold text-indigo-400">100.0% Exact</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="lg:col-span-7 flex flex-col space-y-5">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row items-center justify-between gap-6">
                <div className="flex items-center space-x-5">
                  <div className="w-24 h-24 rounded-2xl bg-indigo-600/20 border-2 border-indigo-500/50 flex flex-col items-center justify-center shadow-lg shadow-indigo-500/10">
                    <span className="text-5xl font-black text-indigo-400 font-mono">
                      {prediction ? prediction.digit : '-'}
                    </span>
                    <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                      Predicted
                    </span>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">
                      Digit {prediction ? prediction.digit : '?'}
                    </h3>
                    <p className="text-sm text-slate-400">
                      Confidence:{' '}
                      <span className="text-emerald-400 font-bold font-mono">
                        {prediction ? (prediction.confidence * 100).toFixed(1) : 0}%
                      </span>
                    </p>
                    <span className="inline-block mt-1 text-xs text-indigo-300 bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-800">
                      Binarized Layer Sign Activations
                    </span>
                  </div>
                </div>

                <button
                  onClick={handleGeminiAnalyze}
                  disabled={isAnalyzing}
                  className="flex items-center space-x-2 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-medium px-4 py-2.5 rounded-xl shadow-lg shadow-purple-500/20 transition disabled:opacity-50"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>{isAnalyzing ? 'Analyzing with Gemini...' : 'Explain with Gemini'}</span>
                </button>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
                    <BarChart3 className="w-4 h-4 text-indigo-400" />
                    <span>Class Score Distribution (0–9)</span>
                  </h3>
                </div>

                <div className="space-y-2">
                  {prediction &&
                    prediction.scores.map((prob, d) => {
                      const isTop = d === prediction.digit;
                      const percent = (prob * 100).toFixed(1);
                      return (
                        <div key={d} className="flex items-center space-x-3 text-xs">
                          <span
                            className={`w-5 font-mono font-bold text-right ${
                              isTop ? 'text-indigo-400 text-sm' : 'text-slate-400'
                            }`}
                          >
                            {d}
                          </span>
                          <div className="flex-1 bg-slate-800/80 h-3.5 rounded-full overflow-hidden border border-slate-700/50">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${
                                isTop
                                  ? 'bg-gradient-to-r from-indigo-500 to-emerald-400'
                                  : 'bg-slate-600'
                              }`}
                              style={{ width: `${Math.max(2, prob * 100)}%` }}
                            />
                          </div>
                          <span
                            className={`w-12 text-right font-mono ${
                              isTop ? 'font-bold text-emerald-400' : 'text-slate-400'
                            }`}
                          >
                            {percent}%
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>

              {geminiAnalysis && (
                <div className="bg-slate-900 border border-purple-500/30 rounded-2xl p-5 shadow-xl bg-purple-950/10">
                  <div className="flex items-center space-x-2 mb-2 text-purple-400 font-semibold text-sm">
                    <Sparkles className="w-4 h-4" />
                    <span>Gemini AI Studio Analysis</span>
                  </div>
                  <div className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap font-sans bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                    {geminiAnalysis}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'fpga' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-emerald-500/20 text-emerald-400 rounded-xl">
                <Cpu className="w-6 h-6" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">Zedboard FPGA Hardware Architecture</h2>
                <p className="text-xs text-slate-400">
                  Full SystemVerilog implementation in <code className="text-indigo-300">hw/rtl/</code>
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800">
                <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">
                  1. Input Layer
                </h4>
                <p className="text-sm font-bold text-white">784 Grayscale Inputs</p>
                <p className="text-xs text-slate-400 mt-1">
                  Per-pixel standardization applied. Normalized floating inputs mapped to binary sign {-1, +1}.
                </p>
              </div>

              <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800">
                <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">
                  2. Hidden Layers (BNN)
                </h4>
                <p className="text-sm font-bold text-white">256 → 256 Neurons</p>
                <p className="text-xs text-slate-400 mt-1">
                  Replaces MAC operations with bitwise XNOR gates and balanced adder popcount tree. Zero DSP logic slices!
                </p>
              </div>

              <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800">
                <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">
                  3. Output Classifier
                </h4>
                <p className="text-sm font-bold text-white">10 Output Classes</p>
                <p className="text-xs text-slate-400 mt-1">
                  Parallel comparator tree outputs argmax digit ID (0-9) with bit-exact agreement to Python PyTorch inference.
                </p>
              </div>
            </div>

            <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-800">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                SystemVerilog RTL Core Files in Repository:
              </h4>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs font-mono text-slate-300">
                <li className="p-2 bg-slate-900 rounded border border-slate-800 flex items-center justify-between">
                  <span>hw/rtl/bnn_pkg.sv</span>
                  <span className="text-slate-500 text-[10px]">Config &amp; Types</span>
                </li>
                <li className="p-2 bg-slate-900 rounded border border-slate-800 flex items-center justify-between">
                  <span>hw/rtl/popcount.sv</span>
                  <span className="text-slate-500 text-[10px]">Combinational Adder Tree</span>
                </li>
                <li className="p-2 bg-slate-900 rounded border border-slate-800 flex items-center justify-between">
                  <span>hw/rtl/bnn_layer.sv</span>
                  <span className="text-slate-500 text-[10px]">Fully Connected BNN Layer</span>
                </li>
                <li className="p-2 bg-slate-900 rounded border border-slate-800 flex items-center justify-between">
                  <span>hw/rtl/bnn_top.sv</span>
                  <span className="text-slate-500 text-[10px]">Top-level Zedboard Module</span>
                </li>
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'quickstart' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-indigo-500/20 text-indigo-400 rounded-xl">
                <Terminal className="w-6 h-6" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">CLI Quickstart &amp; Hardware Flow</h2>
                <p className="text-xs text-slate-400">
                  Reproducible PyTorch training, Icarus Verilog simulation, and Vivado synthesis
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  1. Python Training &amp; Artifact Export:
                </h4>
                <pre className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs text-emerald-400 overflow-x-auto">
{`pip install -r requirements.txt
python scripts/train.py              # train & export mem_files/
python scripts/verify_phase1.py 10000 # verify 10k test set SW vs HW popcount`}
                </pre>
              </div>

              <div>
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  2. Icarus Verilog RTL Verification:
                </h4>
                <pre className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs text-indigo-400 overflow-x-auto">
{`bash hw/sim/run_sim.sh               # Simulates HW and verifies 100% agreement with SW`}
                </pre>
              </div>

              <div>
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  3. Xilinx Vivado Zedboard Bitstream:
                </h4>
                <pre className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs text-purple-400 overflow-x-auto">
{`bash hw/vivado/run_vivado.sh         # Generates bnn_top.bit for Xilinx xc7z020`}
                </pre>
              </div>
            </div>
          </div>
        )}
      </main>

      {showApiKeyModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl">
            <h3 className="text-base font-bold text-white mb-2 flex items-center space-x-2">
              <Key className="w-5 h-5 text-indigo-400" />
              <span>Configure Gemini API Key</span>
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Enter your Gemini API key to enable instant AI-powered handwriting analysis.
              You can get a free key at <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer" className="text-indigo-400 underline">aistudio.google.com/apikey</a>.
            </p>
            <input
              type="password"
              placeholder="AIzaSy..."
              value={geminiApiKey}
              onChange={(e) => setGeminiApiKey(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 mb-4 font-mono"
            />
            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setShowApiKeyModal(false)}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold transition"
              >
                Save &amp; Close
              </button>
            </div>
          </div>
        </div>
      )}

      <footer className="border-t border-slate-800/80 bg-slate-950/80 px-6 py-3 text-center text-xs text-slate-500">
        MNIST BNN v2 • Google AI Studio Ready • 100% Bit-Exact FPGA Co-Design
      </footer>
    </div>
  );
}
