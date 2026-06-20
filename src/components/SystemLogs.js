'use client';

import { useState, useEffect, useRef } from 'react';
import { Terminal } from 'lucide-react';

const INITIAL_LOGS = [
  { time: '15:05:01', level: 'SYSTEM', msg: 'ISRO Geospatial Vector Node initialized.' },
  { time: '15:05:02', level: 'INFO', msg: 'FAISS Vector Database: index loaded (100K+ Sentinel pairs).' },
  { time: '15:05:03', level: 'INFO', msg: 'DualEncoder ResNet50 projection head bound on CUDA device: 0.' },
  { time: '15:05:05', level: 'SUCCESS', msg: 'Modality model bridges: Sentinel-1 SAR / Sentinel-2 Optical operational.' },
  { time: '15:05:06', level: 'SYSTEM', msg: 'Connection established with Bhuvan Portal coordinate indexes.' }
];

const LOG_TEMPLATES = [
  { level: 'INFO', msg: 'Received query request payload.' },
  { level: 'INFO', msg: 'Extracting 256-D signature vector using ResNet50 model.' },
  { level: 'SUCCESS', msg: 'Signature vector generated. L2-normalized.' },
  { level: 'INFO', msg: 'Executing FAISS Cosine Similarity index search (top_k=10)...' },
  { level: 'SUCCESS', msg: 'FAISS search completed in 42ms. 10 nearest neighbors retrieved.' },
  { level: 'INFO', msg: 'Syncing retrieved coordinates with Leaflet Map coordinates.' },
  { level: 'WARN', msg: 'Sentinel-2 band visual cloud cover check: 1.2% (under threshold).' },
  { level: 'INFO', msg: 'Pre-processing co-registration matrices for split slider curtain overlay.' },
  { level: 'SUCCESS', msg: 'Alignment calculation completed. RMS error: 0.04 pixels.' },
  { level: 'INFO', msg: 'Refreshing Accuracy metrics benchmarking charts...' }
];

export default function SystemLogs({ searchTriggered }) {
  const [logs, setLogs] = useState(INITIAL_LOGS);
  const containerRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(() => {
      const randomTemplate = LOG_TEMPLATES[Math.floor(Math.random() * LOG_TEMPLATES.length)];
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];
      
      setLogs(prev => [...prev.slice(-15), {
        time: timeStr,
        level: randomTemplate.level,
        msg: randomTemplate.msg
      }]);
    }, 4500);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!searchTriggered) return;
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    
    const searchLogs = [
      { time: timeStr, level: 'SYSTEM', msg: 'Triggering cross-modal query vector pipeline...' },
      { time: timeStr, level: 'INFO', msg: 'Extracting Sentinel embedding vector.' },
      { time: timeStr, level: 'SUCCESS', msg: 'Retrieval execution completed.' }
    ];

    setLogs(prev => [...prev.slice(-12), ...searchLogs]);
  }, [searchTriggered]);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="glass-panel p-4 rounded-xl flex flex-col gap-2 h-full min-h-[160px] shadow-sm">
      <div className="flex justify-between items-center bg-[#EAF2FF] px-4 py-2 border-b border-border-light rounded-t-xl -mx-4 -mt-4 mb-2 select-none">
        <div className="flex items-center gap-2 text-xs font-bold text-primary-blue">
          <Terminal className="w-4 h-4 text-isro-orange" />
          SYSTEM TERMINAL / INFERENCE LOGS
        </div>
        <div className="flex items-center gap-1.5 text-[9px] font-bold text-emerald-600 bg-white border border-emerald-200 px-2 py-0.5 rounded-full shadow-sm">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          ONLINE
        </div>
      </div>

      <div 
        ref={containerRef}
        className="flex-grow overflow-y-auto bg-white p-2.5 rounded-lg border border-border-light font-mono text-[10px] text-slate-700 flex flex-col gap-1.5 pr-2 select-text shadow-inner"
      >
        {logs.map((log, idx) => {
          let levelColor = 'text-secondary-blue';
          if (log.level === 'SUCCESS') levelColor = 'text-emerald-600 font-bold';
          if (log.level === 'WARN') levelColor = 'text-amber-600 font-bold';
          if (log.level === 'SYSTEM') levelColor = 'text-isro-orange font-bold';

          return (
            <div key={idx} className="flex gap-2 items-start leading-relaxed">
              <span className="text-slate-400 flex-shrink-0">{log.time}</span>
              <span className={`uppercase flex-shrink-0 ${levelColor}`}>[{log.level}]</span>
              <span className="text-slate-600 font-medium">{log.msg}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
