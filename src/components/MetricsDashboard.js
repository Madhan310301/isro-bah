'use client';

import { Activity } from 'lucide-react';

export default function MetricsDashboard() {
  const chartConfigs = [
    {
      title: 'mAP',
      value: '89.4%',
      lineColor: '#F37023', // ISRO Orange
      gradId: 'mapGradient',
      path: 'M 0 50 C 40 30, 80 45, 120 20 C 160 5, 200 25, 240 18 C 260 15, 280 20, 300 15',
      fillPath: 'M 0 50 C 40 30, 80 45, 120 20 C 160 5, 200 25, 240 18 C 260 15, 280 20, 300 15 L 300 70 L 0 70 Z'
    },
    {
      title: 'F1-Score',
      value: '0.88',
      lineColor: '#005EA6', // ISRO Blue
      gradId: 'f1Gradient',
      path: 'M 0 55 C 30 20, 60 10, 90 40 C 120 60, 155 35, 190 38 C 220 40, 260 15, 300 25',
      fillPath: 'M 0 55 C 30 20, 60 10, 90 40 C 120 60, 155 35, 190 38 C 220 40, 260 15, 300 25 L 300 70 L 0 70 Z'
    },
    {
      title: 'Precision',
      value: '91.2%',
      lineColor: '#0083E8', // Light Blue
      gradId: 'precGradient',
      path: 'M 0 45 C 50 35, 90 20, 130 30 C 170 38, 210 18, 250 15 C 270 12, 285 10, 300 8',
      fillPath: 'M 0 45 C 50 35, 90 20, 130 30 C 170 38, 210 18, 250 15 C 270 12, 285 10, 300 8 L 300 70 L 0 70 Z'
    },
    {
      title: 'Recall',
      value: '87.5%',
      lineColor: '#FF8F3D', // Light Saffron
      gradId: 'recGradient',
      path: 'M 0 50 C 45 48, 80 30, 120 32 C 160 35, 200 40, 240 22 C 270 10, 285 15, 300 12',
      fillPath: 'M 0 50 C 45 48, 80 30, 120 32 C 160 35, 200 40, 240 22 C 270 10, 285 15, 300 12 L 300 70 L 0 70 Z'
    }
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-between items-center bg-[#EAF2FF] px-4 py-2 border-b border-border-light rounded-t-xl -mx-4 -mt-4 mb-2 select-none">
        <h2 className="text-xs font-bold text-primary-blue tracking-wider flex items-center gap-2 uppercase">
          <Activity className="w-4 h-4 text-isro-orange" />
          Accuracy Metrics
        </h2>
        <span className="text-[10px] text-slate-550 font-bold italic">Sentinel index analysis</span>
      </div>

      <div className="flex flex-col gap-3 max-h-[320px] overflow-y-auto pr-1">
        {chartConfigs.map((chart, idx) => (
          <div key={idx} className="dashboard-card p-3 rounded-lg flex flex-col gap-1 border border-border-light">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[10px] text-slate-550 font-bold uppercase tracking-wider">{chart.title}</span>
              <span className="text-sm font-extrabold" style={{ color: chart.lineColor }}>
                {chart.value}
              </span>
            </div>

            <div className="w-full h-[60px] relative overflow-hidden bg-slate-50 border border-border-light rounded mt-1">
              <svg viewBox="0 0 300 70" width="100%" height="100%" preserveAspectRatio="none" className="block">
                <defs>
                  <linearGradient id={chart.gradId} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={chart.lineColor} stopOpacity="0.2" />
                    <stop offset="100%" stopColor={chart.lineColor} stopOpacity="0" />
                  </linearGradient>
                </defs>

                {/* Light mode grid lines */}
                <line x1="0" y1="15" x2="300" y2="15" stroke="rgba(0, 59, 142, 0.08)" strokeWidth="1" />
                <line x1="0" y1="35" x2="300" y2="35" stroke="rgba(0, 59, 142, 0.08)" strokeWidth="1" />
                <line x1="0" y1="55" x2="300" y2="55" stroke="rgba(0, 59, 142, 0.08)" strokeWidth="1" />
                <line x1="60" y1="0" x2="60" y2="70" stroke="rgba(0, 59, 142, 0.04)" strokeWidth="1" />
                <line x1="120" y1="0" x2="120" y2="70" stroke="rgba(0, 59, 142, 0.04)" strokeWidth="1" />
                <line x1="180" y1="0" x2="180" y2="70" stroke="rgba(0, 59, 142, 0.04)" strokeWidth="1" />
                <line x1="240" y1="0" x2="240" y2="70" stroke="rgba(0, 59, 142, 0.04)" strokeWidth="1" />

                <path d={chart.fillPath} fill={`url(#${chart.gradId})`} />
                <path 
                  d={chart.path} 
                  fill="none" 
                  stroke={chart.lineColor} 
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </div>

            <div className="flex justify-between text-[8px] text-slate-400 font-mono mt-0.5">
              <span>10m</span>
              <span>22h</span>
              <span>15m</span>
              <span>16m</span>
              <span>28m</span>
              <span>11m</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
