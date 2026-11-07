import React, { useEffect, useState } from 'react';
import { 
  Loader2, 
  CheckCircle2, 
  AlertCircle, 
  Layers, 
  Search, 
  Sparkles, 
  FileText, 
  ArrowRight,
  Video
} from 'lucide-react';
import { inspectionService } from '../services/api';

export default function ProcessingPage({ inspectionId, setActiveTab, setSelectedInspectionId }) {
  const [progress, setProgress] = useState({
    status: 'pending',
    progress_percent: 0,
    current_stage: 'Initializing inspection pipeline...',
    total_frames: 0,
    extracted_frames: 0,
    selected_frames: 0,
    processed_frames: 0,
    detection_count: 0,
    error_message: null,
  });

  useEffect(() => {
    if (!inspectionId) return;

    let intervalId = null;

    const pollStatus = async () => {
      try {
        const res = await inspectionService.getStatus(inspectionId);
        const data = res.data;
        setProgress(data);

        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(intervalId);
        }
      } catch (err) {
        console.error('Failed to poll status:', err);
      }
    };

    pollStatus();
    intervalId = setInterval(pollStatus, 1500);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [inspectionId]);

  const stages = [
    { id: 'extracting', label: 'Frame Extraction', icon: Video, activeStages: ['extracting_frames'] },
    { id: 'selecting', label: 'Intelligent Quality Filter', icon: Layers, activeStages: ['selecting_frames'] },
    { id: 'enhancing', label: 'Underwater Image Enhancement', icon: Sparkles, activeStages: ['enhancing'] },
    { id: 'detecting', label: 'YOLO26 Crack Detection & Severity', icon: Search, activeStages: ['detecting', 'assessing_severity'] },
    { id: 'reporting', label: 'Report & Video Generation', icon: FileText, activeStages: ['generating_video', 'generating_report'] },
  ];

  const isCompleted = progress.status === 'completed';
  const isFailed = progress.status === 'failed';

  return (
    <div className="max-w-3xl mx-auto space-y-8 py-8">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white">
          {isCompleted ? 'Inspection Analysis Complete' : isFailed ? 'Inspection Failed' : 'Autonomous Video Inspection In Progress'}
        </h1>
        <p className="text-sm text-slate-400">
          Inspection ID: <span className="font-mono text-cyan-400">{inspectionId}</span>
        </p>
      </div>

      {/* Main Status Box */}
      <div className="bg-navy-900/60 border border-slate-800 rounded-2xl p-8 shadow-2xl space-y-8">
        {/* Progress Bar & Percentage */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="font-semibold text-slate-200 flex items-center gap-2">
              {!isCompleted && !isFailed && <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />}
              {progress.current_stage || 'Processing...'}
            </span>
            <span className="font-mono text-cyan-400 font-bold text-base">
              {Math.round(progress.progress_percent)}%
            </span>
          </div>

          <div className="w-full h-3 bg-slate-800/80 rounded-full overflow-hidden p-0.5 border border-slate-700/60">
            <div 
              className={`h-full rounded-full transition-all duration-500 ${
                isFailed 
                  ? 'bg-rose-500' 
                  : isCompleted 
                  ? 'bg-emerald-500' 
                  : 'bg-gradient-to-r from-cyan-500 to-blue-600 shadow-md shadow-cyan-500/50'
              }`}
              style={{ width: `${Math.max(5, Math.min(100, progress.progress_percent))}%` }}
            />
          </div>
        </div>

        {/* Live Counters */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
            <div className="text-xs text-slate-400">Extracted</div>
            <div className="text-xl font-bold text-white mt-1">{progress.extracted_frames}</div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
            <div className="text-xs text-slate-400">Selected</div>
            <div className="text-xl font-bold text-cyan-400 mt-1">{progress.selected_frames}</div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
            <div className="text-xs text-slate-400">Analyzed</div>
            <div className="text-xl font-bold text-blue-400 mt-1">{progress.processed_frames}</div>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
            <div className="text-xs text-slate-400">Cracks Found</div>
            <div className="text-xl font-bold text-amber-400 mt-1">{progress.detection_count}</div>
          </div>
        </div>

        {/* Pipeline Stage Indicators */}
        <div className="space-y-3 pt-2">
          <h2 className="text-xs uppercase font-bold tracking-wider text-slate-400">Pipeline Stages</h2>
          <div className="space-y-2">
            {stages.map((st, i) => {
              const Icon = st.icon;
              const isCurrent = st.activeStages.includes(progress.status);
              return (
                <div 
                  key={st.id}
                  className={`flex items-center justify-between p-3 rounded-xl border text-sm transition-all ${
                    isCurrent 
                      ? 'bg-cyan-950/40 border-cyan-500 text-cyan-300' 
                      : 'bg-slate-900/40 border-slate-800/80 text-slate-400'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{st.label}</span>
                  </div>
                  <div>
                    {isCurrent && <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Error message */}
        {isFailed && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold">Pipeline Execution Halted</div>
              <div className="text-xs mt-1 text-rose-200">{progress.error_message || 'An unexpected error occurred during processing.'}</div>
            </div>
          </div>
        )}

        {/* Completion CTA */}
        {isCompleted && (
          <div className="pt-4 text-center">
            <button
              onClick={() => {
                setSelectedInspectionId(inspectionId);
                setActiveTab('results');
              }}
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-sm shadow-xl shadow-emerald-600/30 transition-all hover:scale-[1.02]"
            >
              <CheckCircle2 className="w-5 h-5" />
              View Complete Inspection Report & Results
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
