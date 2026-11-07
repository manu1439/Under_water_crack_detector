import React, { useEffect, useState } from 'react';
import { 
  AlertTriangle, 
  CheckCircle2, 
  FileVideo, 
  Layers, 
  TrendingUp, 
  ArrowRight,
  ShieldAlert,
  Clock,
  Play
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  PieChart,
  Pie
} from 'recharts';
import { inspectionService } from '../services/api';

export default function DashboardPage({ setActiveTab, setSelectedInspectionId }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const res = await inspectionService.getHistory(20);
      setHistory(res.data || []);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  // Compute aggregated statistics
  const totalInspections = history.length;
  const totalFramesProcessed = history.reduce((acc, h) => acc + (h.processed_frames || 0), 0);
  const totalDetections = history.reduce((acc, h) => acc + (h.total_detections || 0), 0);
  
  // Aggregate severities
  const aggregatedSeverity = history.reduce(
    (acc, h) => {
      const dist = h.severity_distribution || {};
      acc.HIGH += dist.HIGH || 0;
      acc.MEDIUM += dist.MEDIUM || 0;
      acc.LOW += dist.LOW || 0;
      acc.UNKNOWN += dist.UNKNOWN || 0;
      return acc;
    },
    { HIGH: 0, MEDIUM: 0, LOW: 0, UNKNOWN: 0 }
  );

  const chartData = [
    { name: 'HIGH Severity', count: aggregatedSeverity.HIGH, color: '#ef4444' },
    { name: 'MEDIUM Severity', count: aggregatedSeverity.MEDIUM, color: '#f97316' },
    { name: 'LOW Severity', count: aggregatedSeverity.LOW, color: '#22c55e' },
    { name: 'UNKNOWN', count: aggregatedSeverity.UNKNOWN, color: '#94a3b8' },
  ];

  const handleViewInspection = (id) => {
    setSelectedInspectionId(id);
    setActiveTab('results');
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Top Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-navy-900 via-navy-800 to-slate-900 border border-slate-800 p-8 shadow-2xl">
        <div className="relative z-10 max-w-2xl">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-cyan-950/80 border border-cyan-800/80 text-cyan-400 text-xs font-semibold mb-4">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            Operational Mission Control
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            Autonomous Underwater Crack & Infrastructure Inspection
          </h1>
          <p className="mt-3 text-sm text-slate-300 leading-relaxed">
            High-fidelity frame selection, adaptive underwater CLAHE enhancement, fine-tuned YOLO26 crack detection, and explainable defect severity assessment.
          </p>
          <div className="mt-6 flex flex-wrap gap-4">
            <button
              onClick={() => setActiveTab('new_inspection')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-sm transition-all shadow-lg shadow-cyan-600/30"
            >
              <Play className="w-4 h-4 fill-current" />
              Launch New Inspection
            </button>
            <button
              onClick={() => setActiveTab('models')}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700 font-medium text-sm transition-all"
            >
              Verify YOLO26 & Dataset
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-navy-900/60 border border-slate-800/80 rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Inspections</span>
            <FileVideo className="w-5 h-5 text-cyan-400" />
          </div>
          <div className="mt-3 text-3xl font-bold text-white">{totalInspections}</div>
          <div className="mt-1 text-xs text-slate-400 flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-cyan-500" /> Completed or archived
          </div>
        </div>

        <div className="bg-navy-900/60 border border-slate-800/80 rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Frames Analyzed</span>
            <Layers className="w-5 h-5 text-blue-400" />
          </div>
          <div className="mt-3 text-3xl font-bold text-white">{totalFramesProcessed}</div>
          <div className="mt-1 text-xs text-slate-400">Quality filtered & enhanced</div>
        </div>

        <div className="bg-navy-900/60 border border-slate-800/80 rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Crack Detections</span>
            <AlertTriangle className="w-5 h-5 text-amber-400" />
          </div>
          <div className="mt-3 text-3xl font-bold text-white">{totalDetections}</div>
          <div className="mt-1 text-xs text-slate-400">Identified via YOLO26</div>
        </div>

        <div className="bg-navy-900/60 border border-slate-800/80 rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">High Severity Alerts</span>
            <ShieldAlert className="w-5 h-5 text-rose-500" />
          </div>
          <div className="mt-3 text-3xl font-bold text-rose-400">{aggregatedSeverity.HIGH}</div>
          <div className="mt-1 text-xs text-slate-400">Urgent structural concern</div>
        </div>
      </div>

      {/* Visual Analytics & Severity Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Severity Distribution Chart */}
        <div className="lg:col-span-2 bg-navy-900/50 border border-slate-800 rounded-xl p-6 shadow-xl">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-bold text-white">Aggregated Defect Severity Distribution</h2>
              <p className="text-xs text-slate-400 mt-1">Classification across all processed frames and inspections</p>
            </div>
            <TrendingUp className="w-5 h-5 text-cyan-400" />
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={12} tickLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc' }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Severity Breakdown Card */}
        <div className="bg-navy-900/50 border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">Severity Criteria</h2>
            <p className="text-xs text-slate-400 mt-1">Multi-factor heuristic criteria</p>
            <div className="mt-4 space-y-3">
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs">
                <span className="font-bold text-rose-400 block mb-0.5">HIGH SEVERITY</span>
                Score 61–100: Large surface area ratio (&gt;8%), complex branching edges, high darkness depth.
              </div>
              <div className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/20 text-xs">
                <span className="font-bold text-orange-400 block mb-0.5">MEDIUM SEVERITY</span>
                Score 34–60: Moderate crack elongation, visible linear width, moderate depth.
              </div>
              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs">
                <span className="font-bold text-emerald-400 block mb-0.5">LOW SEVERITY</span>
                Score 0–33: Minor hairline crack, limited pixel extent.
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 italic">
            *Visual heuristic estimation for inspection prioritizing, not a licensed structural stamp.
          </div>
        </div>
      </div>

      {/* Recent Inspections Table */}
      <div className="bg-navy-900/50 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <h2 className="text-base font-bold text-white">Recent Inspection Runs</h2>
          </div>
          <button 
            onClick={() => setActiveTab('history')}
            className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
          >
            View all history <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-900/80 text-xs uppercase tracking-wider text-slate-400 border-b border-slate-800">
              <tr>
                <th className="px-6 py-3">Filename</th>
                <th className="px-6 py-3">Status</th>
                <th className="px-6 py-3">Frames</th>
                <th className="px-6 py-3">Detections</th>
                <th className="px-6 py-3">High / Med / Low</th>
                <th className="px-6 py-3">Confidence</th>
                <th className="px-6 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {history.length === 0 ? (
                <tr>
                  <td colSpan="7" className="px-6 py-8 text-center text-slate-400">
                    No inspections recorded yet. Click "Launch New Inspection" to start.
                  </td>
                </tr>
              ) : (
                history.slice(0, 5).map((item) => {
                  const dist = item.severity_distribution || {};
                  return (
                    <tr key={item.inspection_id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-6 py-4 font-medium text-white flex items-center gap-2">
                        <FileVideo className="w-4 h-4 text-cyan-400 shrink-0" />
                        <span className="truncate max-w-[200px]">{item.filename}</span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold ${
                          item.status === 'completed'
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                            : item.status === 'failed'
                            ? 'bg-rose-950 text-rose-400 border border-rose-800'
                            : 'bg-amber-950 text-amber-400 border border-amber-800'
                        }`}>
                          {item.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-slate-400">
                        {item.processed_frames} / {item.total_frames_extracted}
                      </td>
                      <td className="px-6 py-4 font-semibold text-white">
                        {item.total_detections}
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2 text-xs">
                          <span className="text-rose-400 font-semibold">{dist.HIGH || 0}H</span>
                          <span className="text-orange-400 font-semibold">{dist.MEDIUM || 0}M</span>
                          <span className="text-emerald-400 font-semibold">{dist.LOW || 0}L</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-slate-400">
                        {(item.avg_confidence * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={() => handleViewInspection(item.inspection_id)}
                          className="px-3 py-1 rounded bg-slate-800 hover:bg-cyan-600 hover:text-white text-xs font-medium text-slate-200 transition-colors"
                        >
                          View Report
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
