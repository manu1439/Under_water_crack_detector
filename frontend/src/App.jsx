import React, { useState } from 'react';
import Navbar from './components/Navbar';
import DashboardPage from './pages/DashboardPage';
import NewInspectionPage from './pages/NewInspectionPage';
import ProcessingPage from './pages/ProcessingPage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';
import ModelManagementPage from './pages/ModelManagementPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedInspectionId, setSelectedInspectionId] = useState(null);

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6 sm:pt-8">
        {activeTab === 'dashboard' && (
          <DashboardPage
            setActiveTab={setActiveTab}
            setSelectedInspectionId={setSelectedInspectionId}
          />
        )}

        {activeTab === 'new_inspection' && (
          <NewInspectionPage
            setActiveTab={setActiveTab}
            setSelectedInspectionId={setSelectedInspectionId}
          />
        )}

        {activeTab === 'processing' && (
          <ProcessingPage
            inspectionId={selectedInspectionId}
            setActiveTab={setActiveTab}
            setSelectedInspectionId={setSelectedInspectionId}
          />
        )}

        {activeTab === 'results' && (
          <ResultsPage
            inspectionId={selectedInspectionId}
            setActiveTab={setActiveTab}
          />
        )}

        {activeTab === 'history' && (
          <HistoryPage
            setActiveTab={setActiveTab}
            setSelectedInspectionId={setSelectedInspectionId}
          />
        )}

        {activeTab === 'models' && (
          <ModelManagementPage />
        )}
      </main>

      {/* Technical Footer */}
      <footer className="border-t border-slate-900 bg-[#050811] py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            <span className="font-semibold text-slate-400">AquaGuard AI</span> — Autonomous Underwater Crack Detection & Severity Assessment
          </div>
          <div className="font-mono text-[11px] text-slate-600">
            Local Ultralytics YOLO26 Engine | OpenCV CLAHE | FastAPI | React
          </div>
        </div>
      </footer>
    </div>
  );
}
