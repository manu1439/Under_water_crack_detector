import React, { useEffect, useState } from 'react';
import { 
  FileVideo, 
  Search, 
  Download, 
  ArrowRight, 
  Clock, 
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  FileText
} from 'lucide-react';
import { inspectionService } from '../services/api';

export default function HistoryPage({ setActiveTab, setSelectedInspectionId }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const res = await inspectionService.getHistory(50);
      setHistory(res.data || []);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  const filtered = history.filter((h) =>
    h.filename?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    h.inspection_id?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-white">Inspection Records & Archive</h1>
          <p className="text-xs text-slate-400 mt-1">Review historical underwater infrastructure inspection telemetry.</p>
        </div>
        <button
          onClick={loadHistory}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors w-fit"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh List
        </button>
      </div>

      {/* Filter / Search Bar */}
      <div className="relative max-w-md">
        <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          placeholder="Filter by filename or inspection ID..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full bg-navy-900/60 border border-slate-800 rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
        />
      </div>

      {/* History Table */}
      <div className="bg-navy-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900 text-slate-400 uppercase font-semibold border-b border-slate-800">
              <tr>
                <th className="px-6 py-3.5">Target Filename</th>
                <th className="px-6 py-3.5">Status</th>
                <th className="px-6 py-3.5">Date / Time</th>
                <th className="px-6 py-3.5">Frames</th>
                <th className="px-6 py-3.5">Defects</th>
                <th className="px-6 py-3.5">Severity Distribution</th>
                <th className="px-6 py-3.5">Avg Conf</th>
                <th className="px-6 py-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {loading ? (
                <tr>
                  <td colSpan="8" className="px-6 py-12 text-center text-slate-400">
                    Loading historical inspections...
                  </td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan="8" className="px-6 py-12 text-center text-slate-500">
                    No matching inspection records found.
                  </td>
                </tr>
              ) : (
                filtered.map((item) => {
                  const dist = item.severity_distribution || {};
                  return (
                    <tr key={item.inspection_id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-6 py-4 font-medium text-white flex items-center gap-2.5">
                        <FileVideo className="w-4 h-4 text-cyan-400 shrink-0" />
                        <div>
                          <div className="font-semibold text-white truncate max-w-[220px]">{item.filename}</div>
                          <div className="text-[10px] text-slate-500 font-mono">{item.inspection_id.slice(0, 14)}...</div>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          item.status === 'completed'
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                            : item.status === 'failed'
                            ? 'bg-rose-950 text-rose-400 border border-rose-800'
                            : 'bg-amber-950 text-amber-400 border border-amber-800'
                        }`}>
                          {item.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-slate-400 font-mono">
                        {item.created_at || '—'}
                      </td>
                      <td className="px-6 py-4 font-mono">
                        {item.processed_frames} / {item.total_frames_extracted}
                      </td>
                      <td className="px-6 py-4 font-bold text-white font-mono">
                        {item.total_detections}
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-1.5 text-[11px] font-mono">
                          <span className="px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-400 border border-rose-900/60">
                            {dist.HIGH || 0}H
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-orange-950/60 text-orange-400 border border-orange-900/60">
                            {dist.MEDIUM || 0}M
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-900/60">
                            {dist.LOW || 0}L
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 font-mono text-slate-300">
                        {(item.avg_confidence * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => {
                              setSelectedInspectionId(item.inspection_id);
                              setActiveTab('results');
                            }}
                            className="px-3 py-1 rounded-lg bg-cyan-600/20 hover:bg-cyan-600 text-cyan-300 hover:text-white border border-cyan-500/40 text-xs font-semibold transition-all"
                          >
                            View Details
                          </button>
                          <a
                            href={inspectionService.getReportDownloadUrl(item.inspection_id, 'pdf')}
                            download
                            title="Download PDF"
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
                          >
                            <FileText className="w-3.5 h-3.5" />
                          </a>
                        </div>
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
