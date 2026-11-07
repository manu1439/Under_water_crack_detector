import React, { useState } from 'react';
import { 
  UploadCloud, 
  FileVideo, 
  Sliders, 
  Sparkles, 
  Play, 
  CheckCircle2, 
  AlertCircle,
  FileImage,
  Info
} from 'lucide-react';
import { inspectionService } from '../services/api';

export default function NewInspectionPage({ setActiveTab, setSelectedInspectionId }) {
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);

  // Inspection Config options
  const [config, setConfig] = useState({
    frame_interval: 2,
    process_all_frames: true,
    enhancement_method: 'clahe',
    clahe_clip_limit: 3.0,
    clahe_tile_grid: 8,
    gamma_value: 1.5,
    confidence_threshold: 0.25,
    iou_threshold: 0.45,
    inference_img_size: 640,
  });

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    setFile(selected);
    setError(null);

    // Create object preview URL
    const url = URL.createObjectURL(selected);
    setFilePreview({
      url,
      type: selected.type.startsWith('video') ? 'video' : 'image',
      name: selected.name,
      size: (selected.size / (1024 * 1024)).toFixed(2),
    });
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      const url = URL.createObjectURL(selected);
      setFilePreview({
        url,
        type: selected.type.startsWith('video') ? 'video' : 'image',
        name: selected.name,
        size: (selected.size / (1024 * 1024)).toFixed(2),
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please choose or drop an underwater video or image first.');
      return;
    }

    try {
      setIsUploading(true);
      setError(null);

      // 1. Upload file
      const uploadRes = await inspectionService.uploadFile(file, (percent) => {
        setUploadProgress(percent);
      });

      const inspectionId = uploadRes.data.inspection_id;

      // 2. Trigger pipeline
      await inspectionService.startProcessing(inspectionId, config);

      setSelectedInspectionId(inspectionId);
      setActiveTab('processing');
    } catch (err) {
      console.error('Failed to submit inspection:', err);
      let message = 'Failed to upload and start processing.';
      if (err.response?.data?.detail) {
        if (typeof err.response.data.detail === 'string') {
          message = err.response.data.detail;
        } else if (Array.isArray(err.response.data.detail)) {
          message = err.response.data.detail.map(d => d.msg || JSON.stringify(d)).join(', ');
        } else {
          message = JSON.stringify(err.response.data.detail);
        }
      } else if (err.message) {
        message = err.message;
      }
      setError(message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-16">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white">Initiate Underwater Infrastructure Inspection</h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload underwater videos (pier, dam, caisson) or high-res images to detect cracks and assess severity.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Upload Zone & Preview Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* File Upload Dropzone */}
          <div 
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="border-2 border-dashed border-slate-700 hover:border-cyan-500/80 rounded-2xl p-8 flex flex-col items-center justify-center text-center bg-navy-900/40 hover:bg-navy-900/70 transition-all cursor-pointer relative"
          >
            <input 
              type="file" 
              accept="video/*,image/*" 
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="p-4 rounded-full bg-cyan-950 border border-cyan-800 text-cyan-400 mb-4">
              <UploadCloud className="w-8 h-8" />
            </div>
            <h2 className="text-base font-semibold text-white">Click or drag & drop inspection media</h2>
            <p className="text-xs text-slate-400 mt-1 max-w-xs">
              Supports MP4, AVI, MOV, MKV underwater footage and JPG, PNG crack photographs (Max 500MB).
            </p>
            {file && (
              <div className="mt-4 px-3 py-1.5 rounded-lg bg-cyan-900/40 border border-cyan-700 text-cyan-300 text-xs font-mono">
                Selected: {file.name} ({filePreview?.size} MB)
              </div>
            )}
          </div>

          {/* Media Preview Box */}
          <div className="bg-navy-900/40 border border-slate-800 rounded-2xl p-5 flex flex-col justify-center items-center overflow-hidden min-h-[250px]">
            {filePreview ? (
              <div className="w-full space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-medium text-slate-200">{filePreview.name}</span>
                  <span>{filePreview.size} MB</span>
                </div>
                <div className="rounded-xl overflow-hidden bg-black/60 max-h-64 flex items-center justify-center border border-slate-800">
                  {filePreview.type === 'video' ? (
                    <video src={filePreview.url} controls className="max-h-60 w-full object-contain" />
                  ) : (
                    <img src={filePreview.url} alt="Preview" className="max-h-60 w-full object-contain" />
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center text-slate-500">
                <FileVideo className="w-12 h-12 mx-auto stroke-[1.2] mb-2" />
                <p className="text-sm">Video or image preview will appear here</p>
              </div>
            )}
          </div>
        </div>

        {/* Processing Configuration Card */}
        <div className="bg-navy-900/50 border border-slate-800 rounded-2xl p-6 space-y-6 shadow-xl">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <Sliders className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white">Inspection & Vision Pipeline Parameters</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Frame Sampling Interval */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
                <span>Frame Extraction Interval</span>
                <span className="font-mono text-cyan-400 text-xs">
                  {config.frame_interval === 1 ? 'Every single frame (100% coverage)' : `Every ${config.frame_interval} frames`}
                </span>
              </label>
              <input
                type="range"
                min="1"
                max="60"
                step="1"
                value={config.frame_interval}
                onChange={(e) => setConfig({ ...config, frame_interval: parseInt(e.target.value) })}
                className="w-full accent-cyan-500"
              />
              <p className="text-[11px] text-slate-400">
                Set to 1 or 2 to analyze every frame in the video.
              </p>
            </div>

            {/* Quality Filtering Mode */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300">Frame Selection Mode</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setConfig({ ...config, process_all_frames: true })}
                  className={`flex-1 py-2 px-3 rounded-lg text-xs font-medium border transition-all ${
                    config.process_all_frames 
                      ? 'bg-cyan-950 border-cyan-500 text-cyan-300' 
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  Analyze All Frames (Recommended)
                </button>
                <button
                  type="button"
                  onClick={() => setConfig({ ...config, process_all_frames: false })}
                  className={`flex-1 py-2 px-3 rounded-lg text-xs font-medium border transition-all ${
                    !config.process_all_frames 
                      ? 'bg-cyan-950 border-cyan-500 text-cyan-300' 
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  Filter Blurry/Dark
                </button>
              </div>
              <p className="text-[11px] text-slate-400">"Analyze All Frames" guarantees no frames are skipped.</p>
            </div>

            {/* Enhancement Method */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300">Underwater Enhancement</label>
              <select
                value={config.enhancement_method}
                onChange={(e) => setConfig({ ...config, enhancement_method: e.target.value })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value="clahe">CLAHE (Adaptive Lightness - Default)</option>
                <option value="gamma">Gamma Brightness Correction</option>
                <option value="retinex">Retinex (Illumination Equalization)</option>
                <option value="original">Original (No Enhancement)</option>
              </select>
              <p className="text-[11px] text-slate-400">CLAHE restores visibility in turbid green/blue waters.</p>
            </div>

            {/* CLAHE Clip Limit (shown when CLAHE selected) */}
            {config.enhancement_method === 'clahe' && (
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
                  <span>CLAHE Clip Limit</span>
                  <span className="font-mono text-cyan-400 text-xs">{config.clahe_clip_limit}</span>
                </label>
                <input
                  type="range"
                  min="1.0"
                  max="6.0"
                  step="0.5"
                  value={config.clahe_clip_limit}
                  onChange={(e) => setConfig({ ...config, clahe_clip_limit: parseFloat(e.target.value) })}
                  className="w-full accent-cyan-500"
                />
                <p className="text-[11px] text-slate-400">Higher clip limit increases local contrast.</p>
              </div>
            )}

            {/* Gamma (shown when gamma selected) */}
            {config.enhancement_method === 'gamma' && (
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
                  <span>Gamma Factor</span>
                  <span className="font-mono text-cyan-400 text-xs">{config.gamma_value}</span>
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="3.0"
                  step="0.1"
                  value={config.gamma_value}
                  onChange={(e) => setConfig({ ...config, gamma_value: parseFloat(e.target.value) })}
                  className="w-full accent-cyan-500"
                />
                <p className="text-[11px] text-slate-400">&gt; 1.0 brightens shadows in deep water.</p>
              </div>
            )}

            {/* Detection Confidence Threshold */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
                <span>Detection Confidence</span>
                <span className="font-mono text-cyan-400 text-xs">{(config.confidence_threshold * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={config.confidence_threshold}
                onChange={(e) => setConfig({ ...config, confidence_threshold: parseFloat(e.target.value) })}
                className="w-full accent-cyan-500"
              />
              <p className="text-[11px] text-slate-400">Minimum score to retain a candidate crack detection.</p>
            </div>

            {/* Inference Resolution */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300">Inference Image Size</label>
              <select
                value={config.inference_img_size}
                onChange={(e) => setConfig({ ...config, inference_img_size: parseInt(e.target.value) })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value={640}>640 x 640 px (Fast - Standard)</option>
                <option value={1024}>1024 x 1024 px (High resolution - Slower)</option>
              </select>
              <p className="text-[11px] text-slate-400">YOLO26 detector input resolution.</p>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex items-center justify-end gap-4">
          <button
            type="button"
            onClick={() => setActiveTab('dashboard')}
            className="px-5 py-2.5 rounded-xl border border-slate-700 text-slate-300 hover:bg-slate-800 text-sm font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!file || isUploading}
            className={`inline-flex items-center gap-2 px-7 py-3 rounded-xl font-bold text-sm text-white shadow-xl transition-all ${
              !file || isUploading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 shadow-cyan-600/30 hover:scale-[1.02]'
            }`}
          >
            <Play className="w-4 h-4 fill-current" />
            {isUploading ? `Uploading (${uploadProgress}%)…` : 'Start AI Inspection'}
          </button>
        </div>
      </form>
    </div>
  );
}
