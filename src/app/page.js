'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import ImageUpload from '../components/ImageUpload';
import ResultsGallery from '../components/ResultsGallery';
import MetricsDashboard from '../components/MetricsDashboard';
import SideBySideView from '../components/SideBySideView';
import SystemLogs from '../components/SystemLogs';
import { Globe } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const MapView = dynamic(() => import('../components/MapView'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[220px] flex flex-col items-center justify-center text-slate-500 glass-card rounded-xl border border-slate-200">
      <Globe className="w-6 h-6 animate-spin text-isro-orange mb-1" />
      <p className="text-[10px] font-semibold">Loading Map Engine...</p>
    </div>
  ),
});

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [searchData, setSearchData] = useState(null);
  const [selectedResult, setSelectedResult] = useState(null);
  const [searchTriggered, setSearchTriggered] = useState(false);
  const [isMapFocused, setIsMapFocused] = useState(false);

  const handleSearchStart = () => {
    setIsLoading(true);
    setSearchTriggered(true);
    setSearchData(null);
    setSelectedResult(null);
    setTimeout(() => setSearchTriggered(false), 200);
  };

  const handleSearchComplete = (data) => {
    setIsLoading(false);
    setSearchData(data);
    if (data.results && data.results.length > 0) {
      setSelectedResult(data.results[0]);
    }
  };

  const handleCoordinateSearch = async (lat, lng) => {
    handleSearchStart();
    await new Promise(resolve => setTimeout(resolve, 800));

    const mockImage = `data:image/svg+xml;utf8,${encodeURIComponent(`
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="100%" height="100%">
        <rect width="256" height="256" fill="#e2e8f0" />
        <circle cx="128" cy="128" r="40" stroke="#F7941D" stroke-width="3" fill="none"/>
        <text x="50" y="240" fill="#475569" font-size="8">Anchor Location: ${lat.toFixed(2)}, ${lng.toFixed(2)}</text>
      </svg>
    `)}`;

    const data = {
      success: true,
      queryImage: mockImage,
      queryModality: 'sar',
      queryCoordinates: [lat, lng],
      queryName: `Coordinates Target (${lat.toFixed(2)}, ${lng.toFixed(2)})`,
      results: [
        {
          rank: 1,
          score: 0.9582,
          id: 'custom_1',
          name: `Custom Location (Zone A)`,
          useCase: 'Manual Coordinates Anchor',
          coordinates: [lat, lng],
          image: mockImage,
          isPerfectMatch: true,
          satellite: 'Sentinel-2 (Optical)'
        },
        {
          rank: 2,
          score: 0.7915,
          id: 'custom_2',
          name: `Custom Location (Zone B)`,
          useCase: 'Manual Coordinates Anchor',
          coordinates: [lat + 0.015, lng - 0.015],
          image: mockImage,
          isPerfectMatch: false,
          satellite: 'Sentinel-2 (Optical)'
        }
      ]
    };

    handleSearchComplete(data);
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F5F8FC]">
      {/* Top Tricolor Strip */}
      <div className="bg-gradient-to-r from-[#F7941D] via-white to-[#128807] h-[4px] w-full" />
      
      {/* Top Utility Bar */}
      <div className="bg-[#f0f4f9] text-[10px] text-slate-600 px-6 py-1.5 flex justify-between items-center border-b border-border-light font-medium select-none">
        {/* Left: Accessibility & Language */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <button className="hover:text-primary-blue transition-colors font-bold">Screen Reader Access</button>
            <span className="text-slate-350">|</span>
            <div className="flex gap-1.5 items-center font-bold">
              <button className="px-1 hover:bg-slate-200 rounded text-[9px]">A-</button>
              <button className="px-1 hover:bg-slate-200 rounded text-[9px]">A</button>
              <button className="px-1 hover:bg-slate-200 rounded text-[9px]">A+</button>
            </div>
          </div>
          <span className="text-slate-350">|</span>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-400">Skip to main content</span>
          </div>
        </div>

        {/* Right: Social icons, Language Toggle & Search */}
        <div className="flex items-center gap-4">
          {/* Language Toggle */}
          <div className="flex items-center gap-1.5">
            <button className="text-primary-blue font-bold hover:underline">English</button>
            <span className="text-slate-350">/</span>
            <button className="text-slate-500 hover:text-primary-blue transition-colors font-medium">हिन्दी</button>
          </div>
          
          <span className="text-slate-355">|</span>

          {/* Social Icons */}
          <div className="flex items-center gap-2 text-[9px] text-slate-500 font-bold">
            <a href="#" className="hover:text-primary-blue">FB</a>
            <a href="#" className="hover:text-primary-blue">YT</a>
            <a href="#" className="hover:text-primary-blue">X</a>
            <a href="#" className="hover:text-primary-blue">IG</a>
          </div>

          <span className="text-slate-355">|</span>

          {/* Utility Search */}
          <div className="relative flex items-center bg-white border border-slate-300 rounded px-2 py-0.5 h-5">
            <input 
              type="text" 
              placeholder="Search..." 
              className="bg-transparent text-[9px] text-slate-700 outline-none w-24 placeholder:text-slate-400"
            />
            <svg viewBox="0 0 24 24" className="w-2.5 h-2.5 text-slate-400 ml-1" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
          </div>
        </div>
      </div>

      {/* Main Header Container */}
      <header className="bg-white border-b border-slate-200 px-6 py-3 flex justify-between items-center">
        {/* Left: Official ISRO Logo & Name */}
        <div className="flex items-center gap-4">
          {/* SVG ISRO Logo */}
          <svg viewBox="0 0 140 50" className="h-10 w-auto flex-shrink-0">
            {/* Blue orbit arcs */}
            <path d="M10,38 Q35,12 85,26" fill="none" stroke="#003B8E" strokeWidth="2" strokeLinecap="round" />
            <path d="M50,42 Q90,14 130,22" fill="none" stroke="#003B8E" strokeWidth="1.5" strokeDasharray="1,2" />
            <path d="M25,45 Q75,10 120,38" fill="none" stroke="#0E4FAF" strokeWidth="1.8" />
            {/* Orange launch wedge rocket arrow */}
            <path d="M60,45 L76,8 L81,45 L74,38 Z" fill="#F7941D" />
            {/* Hindi Text: इसरो */}
            <text x="10" y="36" fontFamily="var(--font-inter), sans-serif" fontWeight="900" fontSize="16" fill="#F7941D">इसरो</text>
            {/* English Text: isro */}
            <text x="86" y="36" fontFamily="var(--font-inter), sans-serif" fontWeight="800" fontSize="18" fill="#003B8E">isro</text>
            <line x1="85" y1="19" x2="119" y2="19" stroke="#003B8E" strokeWidth="2" />
          </svg>

          {/* Bilingual Divider & Title */}
          <div className="flex flex-col border-l border-slate-300 pl-3">
            <span className="text-[10px] font-bold text-slate-700 leading-tight">भारतीय अंतरिक्ष अनुसंधान संगठन</span>
            <span className="text-[11px] font-extrabold text-primary-blue leading-tight uppercase">Indian Space Research Organisation</span>
            <span className="text-[8px] font-semibold text-slate-500 leading-none mt-0.5">अंतरिक्ष विभाग / Department of Space</span>
            <span className="text-[8px] font-semibold text-slate-500 leading-none">भारत सरकार / Government of India</span>
          </div>
        </div>

        {/* Center: Mission Title */}
        <div className="text-center py-1 flex flex-col items-center">
          <div className="bg-[#002147] px-4 py-1.5 rounded-lg border border-[#0E4FAF]/20 shadow-sm">
            <h1 className="text-xs sm:text-sm font-black text-white tracking-wide uppercase leading-none">
              Cross-Modal Satellite Image Retrieval
            </h1>
          </div>
          <span className="text-[9px] text-[#F7941D] font-extrabold uppercase tracking-widest block mt-1.5">
            National Remote Sensing Center • Bhuvan Portal
          </span>
        </div>

        {/* Right: National Emblem of India & Visit Portal Button */}
        <div className="flex items-center gap-4">
          {/* Government Emblem SVG */}
          <div className="flex items-center gap-2 border-r border-slate-200 pr-4">
            <svg viewBox="0 0 100 100" className="h-10 w-auto text-slate-755" fill="currentColor">
              <path d="M50,15 C45,15 42,18 42,22 C42,24 43,26 45,27 L45,35 L40,37 L40,42 L42,42 L42,65 L38,65 L38,70 L62,70 L62,65 L58,65 L58,42 L60,42 L60,37 L55,35 L55,27 C57,26 58,24 58,22 C58,18 55,15 50,15 Z" />
              <circle cx="50" cy="78" r="6" stroke="currentColor" strokeWidth="2" fill="none" />
              <path d="M50,72 L50,84 M44,78 L56,78" stroke="currentColor" strokeWidth="1.5" />
              <text x="50" y="93" fontSize="8" fontWeight="bold" textAnchor="middle" fill="currentColor">सत्यमेव जयते</text>
            </svg>
            <div className="flex flex-col text-[8px] font-bold text-slate-500 uppercase leading-none">
              <span>Satyameva</span>
              <span>Jayate</span>
            </div>
          </div>

          <a 
            href="https://www.isro.gov.in" 
            target="_blank" 
            rel="noopener noreferrer"
            className="bg-[#0E4FAF] hover:bg-[#003B8E] text-white font-bold text-[10px] uppercase tracking-wider px-4 py-2 rounded-lg transition-all border border-[#0E4FAF]/30 shadow-sm"
          >
            Visit ISRO Portal
          </a>
        </div>
      </header>

      {/* Navy Navigation Bar */}
      <nav className="bg-[#002147] px-6 text-white shadow-sm border-b border-[#001733] select-none">
        <div className="max-w-[1600px] mx-auto flex items-center gap-6 h-10">
          <a href="#" className="text-[10px] font-extrabold uppercase tracking-wider text-white border-b-2 border-[#F7941D] h-full flex items-center px-1">
            Home
          </a>
          <a href="#" className="text-[10px] font-bold uppercase tracking-wider text-slate-300 hover:text-white border-b-2 border-transparent hover:border-[#F7941D] h-full flex items-center px-1 transition-all">
            About Retrieval Node
          </a>
          <a href="#" className="text-[10px] font-bold uppercase tracking-wider text-slate-300 hover:text-white border-b-2 border-transparent hover:border-[#F7941D] h-full flex items-center px-1 transition-all">
            Missions
          </a>
          <a 
            href="https://bhuvan.nrsc.gov.in" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-[10px] font-bold uppercase tracking-wider text-slate-300 hover:text-white border-b-2 border-transparent hover:border-[#F7941D] h-full flex items-center px-1 transition-all"
          >
            Bhuvan Platform
          </a>
          <a 
            href="https://www.mosdac.gov.in" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-[10px] font-bold uppercase tracking-wider text-slate-300 hover:text-white border-b-2 border-transparent hover:border-[#F7941D] h-full flex items-center px-1 transition-all"
          >
            MOSDAC Portal
          </a>
          <a href="#" className="text-[10px] font-bold uppercase tracking-wider text-slate-300 hover:text-white border-b-2 border-transparent hover:border-[#F7941D] h-full flex items-center px-1 transition-all">
            Student Corner
          </a>
          <a href="#" className="text-[10px] font-bold uppercase tracking-wider text-slate-300 hover:text-white border-b-2 border-transparent hover:border-[#F7941D] h-full flex items-center px-1 transition-all">
            Contact
          </a>
        </div>
      </nav>

      {/* Announcements Ticker */}
      <div className="bg-[#FFF8F0] border-b border-[#F7941D]/20 h-8 flex items-center overflow-hidden px-6 select-none">
        <div className="max-w-[1600px] mx-auto w-full flex items-center gap-4">
          {/* Ticker Title Badge */}
          <div className="bg-[#F7941D] text-white text-[9px] font-extrabold uppercase px-2 py-0.5 rounded flex-shrink-0 tracking-wider shadow-sm z-10">
            Latest Telemetry
          </div>
          {/* Ticker Content */}
          <div className="flex-grow overflow-hidden relative w-full h-full flex items-center">
            <div className="absolute whitespace-nowrap animate-ticker text-[10px] text-slate-700 font-medium">
              <span className="mx-4 text-primary-blue font-bold">[ONLINE]</span> Sentinel-1 SAR & Sentinel-2 Optical dataset ingestion pipelines are fully operational.
              <span className="mx-8">•</span>
              <span className="mx-4 text-emerald-600 font-bold">[SPEED]</span> FAISS indexing optimized for sub-50ms retrieval query response time.
              <span className="mx-8">•</span>
              <span className="mx-4 text-[#F7941D] font-bold">[TELEMETRY]</span> Polar Satellite Launch Vehicle (PSLV) remote imagery node syncing successfully.
              <span className="mx-8">•</span>
              <span className="mx-4 text-primary-blue font-bold">[DATABASE]</span> Bhuvan map layers loaded with current high-resolution GIS coordinates.
              <span className="mx-8">•</span>
              <span className="mx-4 text-primary-blue font-bold">[ONLINE]</span> Sentinel-1 SAR & Sentinel-2 Optical dataset ingestion pipelines are fully operational.
              <span className="mx-8">•</span>
              <span className="mx-4 text-emerald-600 font-bold">[SPEED]</span> FAISS indexing optimized for sub-50ms retrieval query response time.
              <span className="mx-8">•</span>
              <span className="mx-4 text-[#F7941D] font-bold">[TELEMETRY]</span> Polar Satellite Launch Vehicle (PSLV) remote imagery node syncing successfully.
              <span className="mx-8">•</span>
              <span className="mx-4 text-primary-blue font-bold">[DATABASE]</span> Bhuvan map layers loaded with current high-resolution GIS coordinates.
            </div>
          </div>
        </div>
      </div>

      {/* Centered Map Focus Overlay (Modal) */}
      <AnimatePresence>
        {isMapFocused && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[999] bg-[#000c20]/60 backdrop-blur-md flex items-center justify-center p-4 sm:p-8 map-focus-overlay"
          >
            <motion.div 
              initial={{ scale: 0.95, y: 15 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 15 }}
              transition={{ type: 'spring', damping: 25, stiffness: 180 }}
              className="glass-panel p-5 rounded-2xl w-full max-w-[1100px] h-[82vh] flex flex-col justify-between border border-columbia/20 shadow-2xl"
            >
              <MapView 
                queryCoordinates={searchData?.queryCoordinates}
                results={searchData?.results}
                selectedResult={selectedResult}
                onCoordinateSearch={handleCoordinateSearch}
                isFocused={true}
                onToggleFocus={() => setIsMapFocused(false)}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <main className="flex-grow flex flex-col px-4 sm:px-6 py-4 max-w-[1600px] mx-auto w-full gap-4 relative select-text">
        {/* Row 1: Image Retrieval Gallery (Left) & Upload Dataset (Right) - Aligned Down Header */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-stretch">
          
          {/* Left: Image Retrieval Gallery */}
          <div className="glass-panel p-4 rounded-xl flex flex-col justify-between min-h-[380px]">
            {isLoading ? (
              <div className="flex-grow flex flex-col items-center justify-center py-20 text-center">
                <div className="relative w-12 h-12 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border-2 border-slate-800"></div>
                  <div className="absolute inset-0 rounded-full border-2 border-t-isro-orange border-r-transparent border-b-transparent border-l-transparent animate-spin"></div>
                  <Globe className="w-5 h-5 text-isro-orange animate-pulse" />
                </div>
                <p className="text-xs font-bold text-slate-700 mt-3">Executing Cosine Similarity Search</p>
                <p className="text-[9px] text-slate-550 mt-0.5">Calculating FAISS indexing matrices...</p>
              </div>
            ) : (
              <ResultsGallery 
                results={searchData?.results} 
                selectedResult={selectedResult}
                onSelectResult={setSelectedResult}
              />
            )}
          </div>

          {/* Right: Upload Dataset Widget */}
          <div className="glass-panel p-4 rounded-xl flex flex-col justify-between">
            <ImageUpload 
              onSearchStart={handleSearchStart} 
              onSearchComplete={handleSearchComplete} 
              isLoading={isLoading} 
            />
          </div>

        </div>

        {/* Row 2: Target Area Monitoring Map - Large Centered Map Down Row 1 */}
        <div className="glass-panel p-4 rounded-xl min-h-[380px] w-full flex flex-col justify-between relative group">
          <MapView 
            queryCoordinates={searchData?.queryCoordinates}
            results={searchData?.results}
            selectedResult={selectedResult}
            onCoordinateSearch={handleCoordinateSearch}
            isFocused={false}
            onToggleFocus={() => setIsMapFocused(true)}
          />
          {!searchData && (
            <div className="absolute top-12 right-12 z-20 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-350 bg-white/95 text-slate-700 px-2 py-1 rounded text-[8px] font-bold text-isro-orange border border-border-light shadow-lg backdrop-blur-sm">
              Click map to focus full-screen
            </div>
          )}
        </div>

        {/* Row 3: Opposing Modalities & logs - Placed Down Map */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-stretch">
          
          {/* Column 1: Accuracy Metrics Dashboard */}
          <div className="glass-panel p-4 rounded-xl flex flex-col justify-between min-h-[400px]">
            <MetricsDashboard />
          </div>

          {/* Column 2: Split-Screen Comparison Slider */}
          <div className="glass-panel p-4 rounded-xl flex flex-col justify-between">
            <SideBySideView 
              queryImage={searchData?.queryImage}
              queryModality={searchData?.queryModality}
              selectedResult={selectedResult}
            />
          </div>

          {/* Column 3: System Logs Terminal */}
          <div className="flex flex-col h-full justify-between">
            <SystemLogs searchTriggered={searchTriggered} />
          </div>

        </div>

        {/* Copyright branding footer */}
        <div className="py-2 flex justify-between items-center text-[9px] text-slate-550 border-t border-border-light font-mono mt-1 select-none">
          <span>© 2026 ISRO Remote Control Center</span>
          <div className="flex gap-3">
            <a href="https://bhuvan.nrsc.gov.in" target="_blank" rel="noopener noreferrer" className="hover:text-isro-orange transition-colors">Bhuvan Platform</a>
            <span>•</span>
            <a href="https://www.isro.gov.in" target="_blank" rel="noopener noreferrer" className="hover:text-isro-orange transition-colors">Official ISRO Site</a>
          </div>
        </div>
      </main>
    </div>
  );
}
