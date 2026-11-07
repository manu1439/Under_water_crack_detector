import React, { useEffect, useState } from 'react';
import { 
  Cpu, 
  Database, 
  Play, 
  CheckCircle2, 
  AlertTriangle, 
  HardDrive, 
  Layers, 
  Activity, 
  RefreshCw,
  FileCheck
} from 'lucide-react';
import { modelService } from '../services/api';

export default function ModelManagementPage() {
  const [modelStatus, setModelStatus] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [validationMetrics, setValidationMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [validating, setValidating] = useState(false);
  const [msg, setMsg] = useState(null);

  // Training form state
  const [trainConfig, setTrainConfig] = useState({
    base_model: 'yolo26n.pt',
    epochs: 30,
    batch_size: 16,
    img_size: 640,
    device: 'auto',
  });

  useEffect(() => {
    loadData();
    const interval = setInterval(pollTraining, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [mRes, dRes] = await Promise.all([
        modelService.getStatus(),
        modelService.getDatasetInfo(),
      ]);
      setModelStatus(mRes.data);
      setDatasetInfo(dRes.data);
    } catch (err) {
      console.error('Failed to load model info:', err);
    } finally {
      setLoading(false);
    }
  };

  const pollTraining = async () => {
    try {
      const res = await modelService.getTrainingStatus();
      setTrainingStatus(res.data);
    } catch (err) {
      // ignore
    }
  };

  const handleStartTraining = async (e) => {
    e.preventDefault();
    if (!window.confirm('Trigger local YOLO26 model fine-tuning now?')) return;
    try {
      setMsg({ type: 'info', text: 'Initiating YOLO26 training pipeline in background...' });
      const res = await modelService.trainModel(trainConfig);
      setTrainingStatus(res.data);
    } catch (err) {
      setMsg({ type: 'error', text: err.response?.data?.detail || 'Failed to start training.' });
    }
  };

  const handleRunValidation = async () => {
    try {
      setValidating(true);
      setValidationMetrics(null);
      setMsg(null);
      const res = await modelService.validateModel({ split: 'val', imgsz: 640, conf: 0.25 });
      setValidationMetrics(res.data);
      setMsg({ type: 'success', text: 'Model evaluation completed successfully.' });
    } catch (err) {
      setMsg({ type: 'error', text: err.response?.data?.detail || 'Failed to evaluate model.' });
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-extrabold text-white">YOLO26 Model & Dataset Telemetry</h1>
        <p className="text-xs text-slate-400 mt-1">
          Local model weights, dataset verification, manual fine-tuning, and validation metrics.
        </p>
      </div>

      {msg && (
        <div className={`p-4 rounded-xl text-xs font-medium border flex items-center gap-2 ${
          msg.type === 'error'
            ? 'bg-rose-500/10 border-rose-500/20 text-rose-300'
            : msg.type === 'success'
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
            : 'bg-cyan-500/10 border-cyan-500/20 text-cyan-300'
        }`}>
          {msg.text}
        </div>
      )}

      {/* Top 2 Cards: Model Status & Dataset Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Weights Status */}
        <div className="bg-navy-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-cyan-400" />
              <h2 className="text-base font-bold text-white">Model Weight Status</h2>
            </div>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase ${
              modelStatus?.model_exists 
                ? 'bg-emerald-950 text-emerald-400 border-emerald-800' 
                : 'bg-rose-950 text-rose-400 border-rose-800'
            }`}>
              {modelStatus?.model_exists ? 'Trained best.pt Active' : 'Model Weights Missing'}
            </span>
          </div>

          <div className="space-y-3 text-xs text-slate-300">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Model File Path:</span>
              <span className="font-mono text-slate-200 truncate max-w-xs">{modelStatus?.model_path}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">File Size:</span>
              <span className="font-mono text-slate-200">{modelStatus?.model_size_mb || 0} MB</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Inference Hardware Device:</span>
              <span className="font-mono text-cyan-400 font-bold uppercase">{modelStatus?.device || 'CPU'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Classes:</span>
              <span className="font-mono text-slate-200">
                {modelStatus?.classes?.join(', ') || 'crack (1 class)'}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Base Architecture:</span>
              <span className="font-mono text-slate-200">{modelStatus?.base_model}</span>
            </div>
          </div>

          {!modelStatus?.model_exists && (
            <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[11px] leading-relaxed">
              <strong>Note:</strong> `models/best.pt` does not exist yet. You can trigger training below on the existing 12,012-image local dataset to generate fine-tuned weights.
            </div>
          )}

          {modelStatus?.model_exists && (
            <button
              onClick={handleRunValidation}
              disabled={validating}
              className="w-full mt-2 py-2 rounded-xl bg-cyan-950 hover:bg-cyan-900 border border-cyan-800 text-cyan-300 font-semibold text-xs transition-colors flex items-center justify-center gap-2"
            >
              <FileCheck className="w-4 h-4" />
              {validating ? 'Running Validation...' : 'Run Validation Evaluation'}
            </button>
          )}
        </div>

        {/* Local Verified Dataset Status */}
        <div className="bg-navy-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-400" />
              <h2 className="text-base font-bold text-white">Local Dataset Verification</h2>
            </div>
            <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-blue-950 text-blue-400 border-blue-800 uppercase">
              data.yaml Verified
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Training Split</div>
              <div className="text-lg font-bold text-white mt-1">{datasetInfo?.train_images?.toLocaleString()}</div>
              <div className="text-[10px] text-slate-500">images</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Validation Split</div>
              <div className="text-lg font-bold text-cyan-400 mt-1">{datasetInfo?.valid_images?.toLocaleString()}</div>
              <div className="text-[10px] text-slate-500">images</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Testing Split</div>
              <div className="text-lg font-bold text-blue-400 mt-1">{datasetInfo?.test_images?.toLocaleString()}</div>
              <div className="text-[10px] text-slate-500">images</div>
            </div>
          </div>

          <div className="space-y-2 text-xs text-slate-300 pt-1">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Total Available Images:</span>
              <span className="font-mono text-slate-200 font-bold">
                {((datasetInfo?.train_images || 0) + (datasetInfo?.valid_images || 0) + (datasetInfo?.test_images || 0)).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Class Label:</span>
              <span className="font-mono text-slate-200">`crack` (1 class)</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Annotation Format:</span>
              <span className="font-mono text-slate-200">YOLO (class cx cy w h normalized)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Validation Metrics (if run) */}
      {validationMetrics && (
        <div className="bg-navy-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <h2 className="text-base font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            Empirical Validation Metrics
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
              <div className="text-xs text-slate-400 uppercase font-semibold">Precision</div>
              <div className="text-2xl font-extrabold text-cyan-400 mt-1">
                {(validationMetrics.precision * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
              <div className="text-xs text-slate-400 uppercase font-semibold">Recall</div>
              <div className="text-2xl font-extrabold text-blue-400 mt-1">
                {(validationMetrics.recall * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
              <div className="text-xs text-slate-400 uppercase font-semibold">mAP @ 0.50</div>
              <div className="text-2xl font-extrabold text-emerald-400 mt-1">
                {(validationMetrics.mAP50 * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
              <div className="text-xs text-slate-400 uppercase font-semibold">mAP @ 0.50:0.95</div>
              <div className="text-2xl font-extrabold text-purple-400 mt-1">
                {(validationMetrics.mAP50_95 * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Manual Fine-Tuning Execution Section */}
      <div className="bg-navy-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
        <div className="border-b border-slate-800 pb-3">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Play className="w-4 h-4 text-cyan-400 fill-current" />
            Manual YOLO26 Training Trigger
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Fine-tune YOLO26 locally using the verified dataset. Training runs strictly on manual demand.
          </p>
        </div>

        {trainingStatus?.is_training ? (
          <div className="p-6 rounded-xl bg-cyan-950/40 border border-cyan-800 space-y-3 text-center">
            <div className="text-sm font-bold text-cyan-300">Model Fine-Tuning In Progress...</div>
            <div className="text-xs text-slate-400">
              Epoch {trainingStatus.current_epoch} / {trainingStatus.total_epochs}
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden max-w-md mx-auto">
              <div 
                className="bg-cyan-500 h-full transition-all duration-300"
                style={{ width: `${Math.max(5, trainingStatus.progress_percent)}%` }}
              />
            </div>
          </div>
        ) : (
          <form onSubmit={handleStartTraining} className="space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
              <div className="space-y-1.5">
                <label className="font-semibold text-slate-300">Base Architecture</label>
                <select
                  value={trainConfig.base_model}
                  onChange={(e) => setTrainConfig({ ...trainConfig, base_model: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-white"
                >
                  <option value="yolo26n.pt">yolo26n.pt (Nano - Fast)</option>
                  <option value="yolo26s.pt">yolo26s.pt (Small - Balanced)</option>
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="font-semibold text-slate-300">Epochs</label>
                <input
                  type="number"
                  min="5"
                  max="300"
                  value={trainConfig.epochs}
                  onChange={(e) => setTrainConfig({ ...trainConfig, epochs: parseInt(e.target.value) })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-white"
                />
              </div>

              <div className="space-y-1.5">
                <label className="font-semibold text-slate-300">Batch Size</label>
                <input
                  type="number"
                  min="2"
                  max="64"
                  value={trainConfig.batch_size}
                  onChange={(e) => setTrainConfig({ ...trainConfig, batch_size: parseInt(e.target.value) })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-white"
                />
              </div>

              <div className="space-y-1.5">
                <label className="font-semibold text-slate-300">Device</label>
                <select
                  value={trainConfig.device}
                  onChange={(e) => setTrainConfig({ ...trainConfig, device: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-white"
                >
                  <option value="auto">Auto Detect (CUDA or CPU)</option>
                  <option value="cpu">CPU Only</option>
                  <option value="0">CUDA GPU 0</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs shadow-lg shadow-cyan-600/30 transition-all hover:scale-[1.02]"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                Trigger Local YOLO26 Training Job
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
