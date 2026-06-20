'use client';

import { useState, useRef, useEffect } from 'react';
import { Sliders } from 'lucide-react';

export default function SideBySideView({ queryImage, queryModality, selectedResult }) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isSliding, setIsSliding] = useState(false);
  const containerRef = useRef(null);

  const handleMove = (clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const position = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(position);
  };

  const handleTouchMove = (e) => {
    if (!isSliding) return;
    handleMove(e.touches[0].clientX);
  };

  const handleMouseMove = (e) => {
    if (!isSliding) return;
    handleMove(e.clientX);
  };

  useEffect(() => {
    const handleMouseUp = () => setIsSliding(false);
    
    if (isSliding) {
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isSliding]);

  if (!selectedResult || !queryImage) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center text-slate-400 h-full min-h-[300px]">
        <Sliders className="w-8 h-8 text-slate-500 mb-2 stroke-[1.5]" />
        <p className="text-xs font-semibold text-slate-300">Split-Screen Retrieval View</p>
        <p className="text-[10px] text-slate-500 mt-0.5">Click a retrieved match card to align co-registered overlay</p>
      </div>
    );
  }

  const queryLabel = queryModality.toUpperCase();
  const matchLabel = queryModality === 'sar' ? 'OPTICAL' : 'SAR';

  return (
    <div className="flex flex-col gap-4 h-full justify-between">
      {/* Title */}
      <div className="flex justify-between items-center bg-[#EAF2FF] px-4 py-2 border-b border-border-light rounded-t-xl -mx-4 -mt-4 mb-2 select-none">
        <h2 className="text-xs font-bold text-primary-blue tracking-wider flex items-center gap-2 uppercase">
          <Sliders className="w-4 h-4 text-isro-orange" />
          Split-Screen Retrieval View
        </h2>
        <span className="text-[10px] text-slate-550 font-bold italic">Curtain slider</span>
      </div>

      {/* Slider Canvas Container */}
      <div 
        ref={containerRef}
        className="relative w-full aspect-[4/3] max-w-[450px] mx-auto rounded-xl overflow-hidden border border-border-light bg-slate-50 select-none cursor-ew-resize mt-2 shadow-sm"
        onMouseDown={() => setIsSliding(true)}
        onTouchStart={() => setIsSliding(true)}
      >
        {/* Background Image: Retrieved Match */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img 
          src={selectedResult.image} 
          alt={selectedResult.name} 
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        />
        
        {/* Foreground Image: Query Image clipped dynamically */}
        <div 
          className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none"
          style={{ width: `${sliderPosition}%` }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img 
            src={queryImage} 
            alt="Query satellite patch" 
            className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            style={{ width: containerRef.current?.getBoundingClientRect().width || 450 }}
          />
        </div>

        {/* Thick Glowing Saffron Divider Line Labeled "ISRO Gold" in handle */}
        <div 
          className="absolute top-0 bottom-0 w-[4px] saffron-slider-bar pointer-events-none"
          style={{ left: `calc(${sliderPosition}% - 2px)` }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 saffron-slider-handle flex flex-col items-center justify-center font-mono">
            <span className="text-[7px] text-slate-400 font-bold tracking-tight mb-0.5">‹  ›</span>
            <span className="text-[6px] text-isro-orange font-bold scale-[0.8] leading-none uppercase tracking-widest rotate-90 whitespace-nowrap mt-1">
              ISRO Gold
            </span>
          </div>
        </div>

        {/* Text overlays at the bottom left & right corners */}
        <div className="absolute bottom-2 left-2 bg-white/95 text-[8px] font-mono text-slate-700 p-1.5 rounded border border-border-light leading-normal backdrop-blur-sm pointer-events-none shadow-sm">
          <div className="font-extrabold text-isro-orange">{queryLabel}</div>
          <div>12.01.2024</div>
          <div className="text-slate-400">Sentinel-1</div>
        </div>
        <div className="absolute bottom-2 right-2 bg-white/95 text-[8px] font-mono text-slate-700 p-1.5 rounded border border-border-light leading-normal text-right backdrop-blur-sm pointer-events-none shadow-sm">
          <div className="font-extrabold text-secondary-blue">{matchLabel}</div>
          <div>12.01.2024</div>
          <div className="text-slate-400">Sentinel-2</div>
        </div>
      </div>
    </div>
  );
}
