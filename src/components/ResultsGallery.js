'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Eye } from 'lucide-react';

export default function ResultsGallery({ results, selectedResult, onSelectResult }) {
  const [activeFilter, setActiveFilter] = useState('all');
  const [activeSort, setActiveSort] = useState('score');

  if (!results || results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center text-slate-400 h-full min-h-[300px]">
        <Eye className="w-8 h-8 text-slate-500 mb-2 stroke-[1.5]" />
        <p className="text-xs font-semibold text-slate-350">Image Retrieval Gallery</p>
        <p className="text-[10px] text-slate-500 mt-0.5">Use the presets or file uploader to trigger searches</p>
      </div>
    );
  }

  const filteredResults = results.filter(result => {
    if (activeFilter === 'all') return true;
    if (activeFilter === 'sar') return result.satellite.includes('SAR') || result.satellite.includes('1');
    if (activeFilter === 'optical') return result.satellite.includes('Optical') || result.satellite.includes('2');
    return true;
  });

  const sortedResults = [...filteredResults].sort((a, b) => {
    if (activeSort === 'score') return b.score - a.score;
    return a.rank - b.rank;
  });

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.05 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    show: { opacity: 1, scale: 1, transition: { type: 'spring', stiffness: 120 } }
  };

  return (
    <div className="flex flex-col gap-4 h-full justify-between">
      <div className="flex flex-col gap-3">
        <div className="flex justify-between items-center bg-[#EAF2FF] px-4 py-2 border-b border-border-light rounded-t-xl -mx-4 -mt-4 mb-2 select-none">
          <h2 className="text-xs font-bold text-primary-blue tracking-wider flex items-center gap-2 uppercase">
            <Eye className="w-4 h-4 text-isro-orange" />
            Image Retrieval Gallery
          </h2>
          <span className="text-[10px] text-slate-550 font-bold italic">Retrieved pairs</span>
        </div>

        <div className="flex justify-between items-center gap-2">
          <div className="flex bg-white p-0.5 rounded border border-border-light shadow-sm">
            <button
              onClick={() => setActiveFilter('all')}
              className={`px-2.5 py-1 rounded text-[9px] font-extrabold uppercase transition-all ${
                activeFilter === 'all' ? 'bg-isro-orange text-white shadow-sm' : 'text-slate-550 hover:text-primary-blue'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setActiveFilter('sar')}
              className={`px-2.5 py-1 rounded text-[9px] font-extrabold uppercase transition-all ${
                activeFilter === 'sar' ? 'bg-isro-orange text-white shadow-sm' : 'text-slate-550 hover:text-primary-blue'
              }`}
            >
              SAR
            </button>
            <button
              onClick={() => setActiveFilter('optical')}
              className={`px-2.5 py-1 rounded text-[9px] font-extrabold uppercase transition-all ${
                activeFilter === 'optical' ? 'bg-isro-orange text-white shadow-sm' : 'text-slate-550 hover:text-primary-blue'
              }`}
            >
              Optical
            </button>
          </div>

          <select
            value={activeSort}
            onChange={(e) => setActiveSort(e.target.value)}
            className="bg-white border border-border-light rounded px-2 py-1 text-[9px] font-bold text-slate-700 outline-none cursor-pointer focus:border-secondary-blue shadow-sm"
          >
            <option value="score">Sorted by Similarity</option>
            <option value="rank">Sorted by Date</option>
          </select>
        </div>
      </div>

      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className="flex-grow overflow-y-auto pr-1 grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-3 max-h-[300px] mt-2"
      >
        {sortedResults.map((result) => {
          const isSelected = selectedResult?.id === result.id;
          const scorePercent = (result.score * 100).toFixed(2);
          const isSAR = result.satellite.includes('SAR') || result.satellite.includes('1');

          return (
            <motion.div
              key={result.id}
              variants={itemVariants}
              onClick={() => onSelectResult(result)}
              className={`dashboard-card p-2.5 rounded-xl cursor-pointer flex flex-col gap-2 transition-all duration-300 border border-t-[3px] ${
                isSelected 
                  ? 'border-isro-orange ring-1 ring-isro-orange/30 bg-[#EAF2FF] shadow-sm' 
                  : 'border-border-light border-t-[#F7941D] hover:border-secondary-blue hover:bg-[#EAF2FF]/40 shadow-sm hover:shadow'
              }`}
            >
              {/* Header/Date Badge */}
              <div className="flex justify-between items-center text-[8px] font-bold font-mono px-0.5 select-none">
                <span className="text-isro-orange font-extrabold uppercase tracking-wider">RETRIEVED TARGET</span>
                <span className="text-slate-400 bg-slate-100 px-1 py-0.2 rounded">20 JUN 2026</span>
              </div>

              <div className="relative aspect-square w-full bg-slate-100 rounded overflow-hidden border border-border-light flex-shrink-0">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img 
                  src={result.image} 
                  alt={result.name} 
                  className="w-full h-full object-cover"
                />
                
                <div className="absolute top-1.5 left-1.5 bg-white/95 text-[8px] font-extrabold text-slate-700 px-1.5 py-0.5 rounded border border-border-light uppercase flex items-center gap-1 shadow-sm">
                  <span className={`w-1.5 h-1.5 rounded-full ${isSAR ? 'bg-isro-orange' : 'bg-[#0E4FAF]'}`} />
                  {isSAR ? 'SAR SENSOR' : 'OPTICAL RGB'}
                </div>

                <div className="absolute bottom-1.5 right-1.5 bg-white/95 text-[8px] font-mono text-isro-orange px-1.5 py-0.5 rounded border border-border-light shadow-sm font-bold">
                  {scorePercent}% MATCH
                </div>
              </div>

              <div className="flex flex-col gap-0.5 text-[9px] text-slate-500 font-mono mt-0.5 leading-normal">
                <div className="flex justify-between">
                  <span className="text-slate-400">GRID ID</span>
                  <span className="text-slate-700 font-bold">#{result.rank * 11}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">SENSOR</span>
                  <span className="text-slate-700 truncate max-w-[90px]">{isSAR ? 'Sentinel-1 SAR' : 'Sentinel-2 RGB'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">LOCATION</span>
                  <span className="text-slate-700 truncate max-w-[90px] font-bold">{result.useCase.split(' ')[0]}</span>
                </div>
              </div>
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
}
