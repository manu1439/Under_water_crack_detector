import React, { useEffect, useState } from 'react';
import { 
  Download, 
  FileText, 
  Video, 
  Image as ImageIcon, 
  AlertTriangle, 
  ShieldAlert, 
  CheckCircle2, 
  Eye, 
  Maximize2,
  Layers,
  Sparkles,
  ArrowLeft
} from 'lucide-react';
import { inspectionService } from '../services/api';

export default function ResultsPage({ inspectionId, setActiveTab }) {
  const [data, setData] = useState(null);
  const [framesInfo, setFramesInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Active view states
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [activeTabMode, setActiveTabMode] = useState('annotated'); // 'annotated', 'comparison', 'video'
  const [highlightedDetectionId, setHighlightedDetectionId] = useState(null);

  useEffect(() => {
    if (!inspectionId) return;
    loadResults();
  }, [inspectionId]);

  const loadResults = async () => {
    try {
      setLoading(true);
      setError(null);

      const [res, framesRes] = await Promise.all([
        inspectionService.getResults(inspectionId),
        inspectionService.getFrames(inspectionId).catch(() => ({ data: null })),
      ]);

      setData(res.data);
      setFramesInfo(framesRes.data);
    } catch (err) {
      console.error('Failed to load inspection results:', err);
      setError(err.response?.data?.detail || 'Failed to fetch inspection results.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-10 h-10 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 text-sm">Loading inspection defect analytics...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-xl mx-auto py-16 text-center space-y-4">
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm">
          {error || 'No inspection selected.'}
        </div>
        <button
          onClick={() => setActiveTab('dashboard')}
          className="px-4 py-2 rounded-lg bg-slate-800 text-slate-200 text-xs font-semibold hover:bg-slate-700"
        >
          Return to Dashboard
        </button>
      </div>
    );
  }

  const { summary, config, frame_results = [] } = data;
  const currentFrame = frame_results[selectedFrameIndex] || null;

  // Flatten all detections with frame context for table
  const allDetectionsList = [];
  frame_results.forEach((fr, fIdx) => {
    (fr.detections || []).forEach((d, dIdx) => {
      allDetectionsList.push({
        ...d,
        frame_number: fr.frame_number,
        timestamp: fr.timestamp,
        frameIndex: fIdx,
        defectNumber: dIdx + 1,
        totalInFrame: fr.detections.length,
      });
    });
  });

  const getSeverityBadge = (level) => {
    switch (level) {
      case 'HIGH':
        return 'bg-rose-950 text-rose-400 border-rose-800';
      case 'MEDIUM':
        return 'bg-orange-950 text-orange-400 border-orange-800';
      case 'LOW':
        return 'bg-emerald-950 text-emerald-400 border-emerald-800';
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };

  const handleRowClick = (det) => {
    setHighlightedDetectionId(det.detection.detection_id);
    setSelectedFrameIndex(det.frameIndex);
    setActiveTabMode('annotated');
    setTimeout(() => {
      const card = document.getElementById(`defect-card-${det.detection.detection_id}`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }, 150);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Top Header & Export Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <button
            onClick={() => setActiveTab('dashboard')}
            className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-cyan-400 mb-2 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Back to Dashboard
          </button>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-extrabold text-white tracking-tight">
              Inspection Report: {summary.filename}
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800">
              ANALYSIS COMPLETE
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            ID: {summary.inspection_id} | Created: {summary.created_at}
          </p>
        </div>

        {/* Action Buttons: PDF & JSON download */}
        <div className="flex items-center gap-3">
          <a
            href={inspectionService.getReportDownloadUrl(summary.inspection_id, 'json')}
            download
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition-all"
          >
            <Download className="w-4 h-4 text-slate-400" />
            JSON Report
          </a>
          <a
            href={inspectionService.getReportDownloadUrl(summary.inspection_id, 'pdf')}
            download
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs shadow-lg shadow-cyan-600/30 transition-all hover:scale-[1.02]"
          >
            <FileText className="w-4 h-4" />
            Download PDF Report
          </a>
        </div>
      </div>

      {/* Metric Cards Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className="p-4 rounded-xl bg-navy-900/50 border border-slate-800">
          <div className="text-[11px] text-slate-400 uppercase font-semibold">Analyzed Frames</div>
          <div className="text-xl font-bold text-white mt-1">{summary.processed_frames}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">of {summary.total_frames_extracted} extracted</div>
        </div>

        <div className="p-4 rounded-xl bg-navy-900/50 border border-slate-800">
          <div className="text-[11px] text-slate-400 uppercase font-semibold">Total Cracks</div>
          <div className="text-xl font-bold text-cyan-400 mt-1">{summary.total_detections}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">YOLO26 candidates</div>
        </div>

        <div className="p-4 rounded-xl bg-navy-900/50 border border-slate-800">
          <div className="text-[11px] text-rose-400 uppercase font-semibold">High Severity</div>
          <div className="text-xl font-bold text-rose-400 mt-1">{summary.severity_distribution?.HIGH || 0}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Critical defect</div>
        </div>

        <div className="p-4 rounded-xl bg-navy-900/50 border border-slate-800">
          <div className="text-[11px] text-orange-400 uppercase font-semibold">Medium Severity</div>
          <div className="text-xl font-bold text-orange-400 mt-1">{summary.severity_distribution?.MEDIUM || 0}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Moderate extent</div>
        </div>

        <div className="p-4 rounded-xl bg-navy-900/50 border border-slate-800">
          <div className="text-[11px] text-emerald-400 uppercase font-semibold">Low Severity</div>
          <div className="text-xl font-bold text-emerald-400 mt-1">{summary.severity_distribution?.LOW || 0}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Hairline / minor</div>
        </div>

        <div className="p-4 rounded-xl bg-navy-900/50 border border-slate-800">
          <div className="text-[11px] text-slate-400 uppercase font-semibold">Mean Confidence</div>
          <div className="text-xl font-bold text-white mt-1">{(summary.avg_confidence * 100).toFixed(1)}%</div>
          <div className="text-[10px] text-slate-500 mt-0.5">YOLO class cert.</div>
        </div>
      </div>

      {/* Main Inspection View Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Visual Canvas (Annotated / Comparison / Video) */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-navy-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
            {/* View Mode Toolbar */}
            <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveTabMode('annotated')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    activeTabMode === 'annotated'
                      ? 'bg-cyan-600 text-white shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Annotated Detection Keyframe
                </button>
                <button
                  onClick={() => setActiveTabMode('comparison')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    activeTabMode === 'comparison'
                      ? 'bg-cyan-600 text-white shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Enhancement Comparison (Original vs CLAHE)
                </button>
                {data.annotated_video_path && (
                  <button
                    onClick={() => setActiveTabMode('video')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeTabMode === 'video'
                        ? 'bg-cyan-600 text-white shadow'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Annotated Video Player
                  </button>
                )}
              </div>

              {currentFrame && (
                <div className="text-xs text-slate-400 font-mono hidden sm:block">
                  Frame #{currentFrame.frame_number} @ {currentFrame.timestamp.toFixed(2)}s
                </div>
              )}
            </div>

            {/* Display Viewport */}
            <div className="p-4 bg-black/70 flex items-center justify-center min-h-[380px]">
              {activeTabMode === 'annotated' && currentFrame && (
                <div className="w-full flex flex-col items-center">
                  <img
                    src={`/api/inspection/${summary.inspection_id}/static/annotated/${currentFrame.annotated_path ? currentFrame.annotated_path.split(/[\\/]/).pop() : (currentFrame.enhanced_path || currentFrame.original_path).split(/[\\/]/).pop()}`}
                    alt="Annotated Frame"
                    className="max-h-[460px] w-auto object-contain rounded-lg border border-slate-800 shadow-md"
                    onError={(e) => {
                      // Fallback to static output direct path
                      e.target.src = `/static/outputs/${summary.inspection_id}/selected/${currentFrame.original_path.split(/[\\/]/).pop()}`;
                    }}
                  />
                  {currentFrame.detections?.length > 0 ? (
                    <div className="mt-3 flex flex-wrap gap-2 justify-center">
                      {currentFrame.detections.map((d) => (
                        <span
                          key={d.detection.detection_id}
                          className={`text-xs px-2.5 py-1 rounded-md border font-mono font-medium ${getSeverityBadge(
                            d.severity.severity
                          )}`}
                        >
                          {d.detection.class_name} ({(d.detection.confidence * 100).toFixed(0)}%) — {d.severity.severity} [{d.severity.severity_score.toFixed(0)}]
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div className="mt-2 text-xs text-slate-500">No cracks detected in this frame.</div>
                  )}
                </div>
              )}

              {activeTabMode === 'comparison' && currentFrame && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full">
                  <div className="space-y-1.5 text-center">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Original Raw Frame</span>
                    <div className="border border-slate-800 rounded-lg overflow-hidden bg-black/40">
                      <img
                        src={`/api/inspection/${summary.inspection_id}/static/${currentFrame.original_path && currentFrame.original_path.includes('selected') ? 'selected' : 'extracted'}/${currentFrame.original_path ? currentFrame.original_path.split(/[\\/]/).pop() : ''}`}
                        alt="Original"
                        className="max-h-72 w-full object-contain"
                      />
                    </div>
                  </div>
                  <div className="space-y-1.5 text-center">
                    <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                      Enhanced ({summary.enhancement_method.toUpperCase()})
                    </span>
                    <div className="border border-cyan-800/40 rounded-lg overflow-hidden bg-black/40">
                      <img
                        src={`/api/inspection/${summary.inspection_id}/static/enhanced/${currentFrame.enhanced_path ? currentFrame.enhanced_path.split(/[\\/]/).pop() : currentFrame.original_path.split(/[\\/]/).pop()}`}
                        alt="Enhanced"
                        className="max-h-72 w-full object-contain"
                      />
                    </div>
                  </div>
                </div>
              )}

              {activeTabMode === 'video' && data.annotated_video_path && (
                <div className="w-full flex justify-center bg-black/60 rounded-lg p-2">
                  <video
                    key={summary.inspection_id}
                    src={inspectionService.getVideoUrl(summary.inspection_id)}
                    controls
                    preload="auto"
                    playsInline
                    className="max-h-[460px] w-full rounded-lg border border-slate-800 shadow-xl"
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              )}
            </div>

            {/* Frame Selector Carousel / Thumbnails */}
            {frame_results.length > 1 && (
              <div className="p-3 bg-slate-950 border-t border-slate-800 flex items-center gap-2 overflow-x-auto">
                <span className="text-xs font-bold text-slate-400 shrink-0 px-2 uppercase">Frames:</span>
                {frame_results.map((fr, idx) => (
                  <button
                    key={fr.frame_number}
                    onClick={() => {
                      setSelectedFrameIndex(idx);
                      setHighlightedDetectionId(null);
                    }}
                    className={`shrink-0 px-3 py-1.5 rounded-lg text-xs font-mono font-medium border transition-all ${
                      selectedFrameIndex === idx
                        ? 'bg-cyan-600 border-cyan-400 text-white'
                        : fr.crack_count > 0
                        ? 'bg-rose-950/40 border-rose-800/80 text-rose-300 hover:bg-slate-800'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    #{fr.frame_number} {fr.crack_count > 0 && `(${fr.crack_count})`}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Col: Frame Details & Severity Explanations */}
        <div className="space-y-4">
          <div className="bg-navy-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-xl">
            <h2 className="text-base font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              Frame Defect Inspection Details
            </h2>

            {currentFrame && currentFrame.detections?.length > 0 ? (
              <div className="space-y-4 max-h-[520px] overflow-y-auto pr-1">
                {currentFrame.detections.map((dr, i) => {
                  const isHighlighted = highlightedDetectionId === dr.detection.detection_id;
                  const sev = dr.severity;
                  return (
                    <div
                      id={`defect-card-${dr.detection.detection_id}`}
                      key={dr.detection.detection_id}
                      className={`p-4 rounded-xl border transition-all ${
                        isHighlighted
                          ? 'bg-cyan-950/60 border-cyan-400 shadow-lg ring-2 ring-cyan-500 scale-[1.01]'
                          : 'bg-slate-900/60 border-slate-800'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs font-semibold text-slate-200">
                          Crack #{i + 1}
                        </span>
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded border ${getSeverityBadge(
                            sev.severity
                          )}`}
                        >
                          {sev.severity} (Score: {sev.severity_score.toFixed(0)}/100)
                        </span>
                      </div>

                      <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-400 font-mono">
                        <div>Conf: {(dr.detection.confidence * 100).toFixed(1)}%</div>
                        <div>Box: {Math.round(dr.detection.bbox.x2 - dr.detection.bbox.x1)}x{Math.round(dr.detection.bbox.y2 - dr.detection.bbox.y1)} px</div>
                      </div>

                      {/* Explanation Reasons */}
                      <div className="mt-3 pt-2 border-t border-slate-800/80">
                        <span className="text-[11px] font-semibold text-slate-300 block mb-1">
                          Heuristic Reasons:
                        </span>
                        <ul className="list-disc list-inside space-y-0.5 text-[11px] text-slate-400">
                          {sev.reasons?.map((reason, rIdx) => (
                            <li key={rIdx}>{reason}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="py-12 text-center text-slate-500 text-xs">
                No crack detected on this selected frame.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Frame-Level Detections Full Table */}
      <div className="bg-navy-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-xl space-y-0">
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/40">
          <div>
            <h2 className="text-base font-bold text-white">Consolidated Frame-Level Crack Detections</h2>
            <p className="text-xs text-slate-400 mt-0.5">Click any row to jump and highlight that defect in the viewer.</p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800 px-2.5 py-1 rounded-lg">
            {allDetectionsList.length} Total Candidates
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900 text-slate-400 uppercase font-semibold border-b border-slate-800">
              <tr>
                <th className="px-5 py-3">Frame</th>
                <th className="px-5 py-3">Timestamp</th>
                <th className="px-5 py-3">YOLO Conf</th>
                <th className="px-5 py-3">Severity</th>
                <th className="px-5 py-3">Severity Score</th>
                <th className="px-5 py-3">BBox (px)</th>
                <th className="px-5 py-3">Primary Assessment Reason</th>
                <th className="px-5 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {allDetectionsList.length === 0 ? (
                <tr>
                  <td colSpan="8" className="px-6 py-8 text-center text-slate-500">
                    No cracks were detected in this inspection video.
                  </td>
                </tr>
              ) : (
                allDetectionsList.map((det) => {
                  const bbox = det.detection.bbox;
                  const w = Math.round(bbox.x2 - bbox.x1);
                  const h = Math.round(bbox.y2 - bbox.y1);
                  const isHighlighted = highlightedDetectionId === det.detection.detection_id;

                  return (
                    <tr
                      key={det.detection.detection_id}
                      onClick={() => handleRowClick(det)}
                      className={`cursor-pointer transition-colors ${
                        isHighlighted
                          ? 'bg-cyan-950/60 border-l-4 border-l-cyan-400 text-white'
                          : 'hover:bg-slate-800/40'
                      }`}
                    >
                      <td className="px-5 py-3 font-mono font-medium">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-white">#{det.frame_number}</span>
                          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-cyan-300 font-sans border border-slate-700">
                            Crack #{det.defectNumber} of {det.totalInFrame}
                          </span>
                        </div>
                      </td>
                      <td className="px-5 py-3 font-mono">{det.timestamp.toFixed(2)}s</td>
                      <td className="px-5 py-3 font-mono font-bold text-slate-200">
                        {(det.detection.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="px-5 py-3">
                        <span
                          className={`inline-block px-2 py-0.5 rounded font-semibold border text-[10px] ${getSeverityBadge(
                            det.severity.severity
                          )}`}
                        >
                          {det.severity.severity}
                        </span>
                      </td>
                      <td className="px-5 py-3 font-mono">{det.severity.severity_score.toFixed(0)}/100</td>
                      <td className="px-5 py-3 font-mono text-slate-400">
                        {w} x {h}
                      </td>
                      <td className="px-5 py-3 text-slate-400 truncate max-w-xs">
                        {det.severity.reasons?.[0] || 'Visual extent'}
                      </td>
                      <td className="px-5 py-3 text-right">
                        <button className="px-2.5 py-1 rounded bg-slate-800 hover:bg-cyan-600 hover:text-white text-[11px] font-medium text-slate-300">
                          View Frame
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
