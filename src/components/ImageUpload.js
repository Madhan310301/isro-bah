'use client';

import { useState } from 'react';
import { Upload, CheckCircle2, Layers } from 'lucide-react';

const PRESET_TILES = [
  { id: 'bangalore', name: 'Bangalore Urban', region: 'Karnataka', type: 'urban', icon: '🏙️' },
  { id: 'kerala', name: 'Kerala Floods', region: 'Alappuzha', type: 'flood', icon: '🌊' },
  { id: 'rajasthan', name: 'Bhadla Solar Park', region: 'Rajasthan', type: 'desert', icon: '☀️' },
  { id: 'mumbai', name: 'Mumbai Port', region: 'Maharashtra', type: 'coastal', icon: '⚓' },
  { id: 'sundarbans', name: 'Sundarbans Delta', region: 'West Bengal', type: 'delta', icon: '🌳' },
];

export default function ImageUpload({ onSearchStart, onSearchComplete, isLoading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [modality, setModality] = useState('sar'); // 'sar' means query is SAR -> retrieve Optical
  const [outputText, setOutputText] = useState('');
  const [progress, setProgress] = useState(78); // Static 78% as in screenshot

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setOutputText(file.name);
      processFile(file);
    }
  };

  const processFile = async (file) => {
    onSearchStart();
    const formData = new FormData();
    formData.append('file', file);
    formData.append('modality', modality);

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.success) {
        onSearchComplete(data);
      } else {
        alert('Extraction failed');
      }
    } catch (error) {
      console.error(error);
    }
  };

  const handlePresetSelect = async (presetId) => {
    onSearchStart();
    setSelectedFile(null);
    setOutputText(PRESET_TILES.find(t => t.id === presetId).name + ' Template');

    const formData = new FormData();
    formData.append('presetId', presetId);
    formData.append('modality', modality);

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.success) {
        onSearchComplete(data);
      }
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Title */}
      <div className="flex justify-between items-center bg-[#EAF2FF] px-4 py-2 border-b border-border-light rounded-t-xl -mx-4 -mt-4 mb-2 select-none">
        <h2 className="text-xs font-bold text-primary-blue tracking-wider flex items-center gap-2 uppercase">
          <Layers className="w-4 h-4 text-isro-orange" />
          Upload Dataset
        </h2>
        <span className="text-[10px] text-slate-550 font-bold italic">Satellite inputs</span>
      </div>

      {/* Input box upload */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Input</label>
          <div className="relative flex items-center bg-white border border-border-light focus-within:border-secondary-blue rounded-lg overflow-hidden h-9 px-3 transition-colors shadow-sm">
            <input
              type="text"
              readOnly
              value={selectedFile ? selectedFile.name : 'Upload Dataset'}
              className="bg-transparent text-xs text-slate-700 outline-none w-full pr-10 cursor-default font-medium"
            />
            <input
              type="file"
              id="dataset-file-input"
              className="hidden"
              onChange={handleFileChange}
              disabled={isLoading}
            />
            <label 
              htmlFor="dataset-file-input" 
              className="absolute right-1 top-1 bottom-1 w-7 bg-secondary-blue hover:bg-primary-blue text-white rounded flex items-center justify-center cursor-pointer transition-colors shadow-sm"
            >
              <Upload className="w-3.5 h-3.5" />
            </label>
          </div>
        </div>

        {/* Output box dataset */}
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Output</label>
          <input
            type="text"
            placeholder="Enter your dataset..."
            value={outputText}
            onChange={(e) => setOutputText(e.target.value)}
            className="bg-white border border-border-light focus:border-secondary-blue rounded-lg h-9 px-3 text-xs text-slate-700 outline-none transition-colors shadow-sm"
          />
        </div>

        {/* Progress Bar (as seen in screenshot: saffron/blue bar at 78%) */}
        <div className="flex flex-col gap-1.5 mt-1">
          <div className="flex justify-between items-center text-[10px] font-bold">
            <span className="text-slate-500">Progress</span>
            <span className="text-isro-orange">{progress}%</span>
          </div>
          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden border border-slate-300/40">
            <div 
              className="h-full bg-isro-orange rounded-full transition-all duration-1000"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="text-[9px] text-slate-400 font-semibold mt-0.5">Sentinel-1/2 Processing</span>
        </div>

        {/* Modality toggle */}
        <div className="grid grid-cols-2 gap-2 mt-1">
          <button
            onClick={() => setModality('sar')}
            className={`py-1.5 rounded text-[10px] font-bold transition-all border ${
              modality === 'sar'
                ? 'bg-isro-orange text-white border-isro-orange/50 shadow-sm'
                : 'bg-white text-slate-500 border-border-light hover:text-primary-blue hover:bg-card-blue'
            }`}
          >
            S1 SAR Query
          </button>
          <button
            onClick={() => setModality('optical')}
            className={`py-1.5 rounded text-[10px] font-bold transition-all border ${
              modality === 'optical'
                ? 'bg-isro-orange text-white border-isro-orange/50 shadow-sm'
                : 'bg-white text-slate-500 border-border-light hover:text-primary-blue hover:bg-card-blue'
            }`}
          >
            S2 Optical Query
          </button>
        </div>

        {/* Quick Presets for Demo */}
        <div className="border-t border-border-light pt-3 mt-1">
          <span className="text-[9px] text-slate-400 uppercase font-bold tracking-wider block mb-2">Preset Coordinates</span>
          <div className="flex flex-wrap gap-1.5">
            {PRESET_TILES.map(tile => (
              <button
                key={tile.id}
                onClick={() => handlePresetSelect(tile.id)}
                disabled={isLoading}
                className="bg-white border border-border-light hover:border-secondary-blue hover:bg-card-blue text-[9px] font-bold text-slate-700 px-2.5 py-1 rounded transition-all disabled:opacity-50 shadow-sm hover:shadow hover:text-primary-blue"
              >
                {tile.name}
              </button>
            ))}
          </div>
        </div>

        {/* Uploader Details */}
        <div className="flex justify-between items-center text-[9px] text-slate-450 mt-2 border-t border-border-light pt-2 font-mono">
          <span>Control Node: ISRO Scientist</span>
          <span className="flex items-center gap-1 text-emerald-600 font-bold">
            <CheckCircle2 className="w-3 h-3" /> System Verified
          </span>
        </div>
      </div>
    </div>
  );
}
