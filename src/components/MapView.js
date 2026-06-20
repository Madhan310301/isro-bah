'use client';

import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Navigation, Globe, Layers, Maximize2, Minimize2 } from 'lucide-react';

export default function MapView({ 
  queryCoordinates, 
  results, 
  selectedResult, 
  onCoordinateSearch,
  isFocused = false,
  onToggleFocus
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersGroupRef = useRef(null);
  const tileLayerRef = useRef(null);

  const [latVal, setLatVal] = useState('28.6139');
  const [lngVal, setLngVal] = useState('77.2090');
  const [mapType, setMapType] = useState('satellite');

  const tileUrls = {
    satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    streets: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
  };

  const handleCoordSubmit = (e) => {
    e.preventDefault();
    const lat = parseFloat(latVal);
    const lng = parseFloat(lngVal);
    
    if (isNaN(lat) || isNaN(lng)) return;

    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([lat, lng], 12);
    }
    if (onCoordinateSearch) {
      onCoordinateSearch(lat, lng);
    }
  };

  const toggleMapLayer = (type) => {
    setMapType(type);
    if (mapInstanceRef.current && tileLayerRef.current) {
      mapInstanceRef.current.removeLayer(tileLayerRef.current);
      const attribution = type === 'satellite' 
        ? 'Tiles &copy; Esri &mdash; Source: Esri, USDA, USGS'
        : '&copy; OpenStreetMap contributors';

      tileLayerRef.current = L.tileLayer(tileUrls[type], { attribution }).addTo(mapInstanceRef.current);
    }
  };

  useEffect(() => {
    if (typeof window === 'undefined' || !mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      mapInstanceRef.current = L.map(mapContainerRef.current, {
        center: queryCoordinates || [28.6139, 77.2090],
        zoom: 7,
        zoomControl: true,
      });

      tileLayerRef.current = L.tileLayer(tileUrls.satellite, {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, USDA, USGS',
        maxZoom: 20
      }).addTo(mapInstanceRef.current);

      markersGroupRef.current = L.layerGroup().addTo(mapInstanceRef.current);
      mapInstanceRef.current.doubleClickZoom.disable();
    }

    if (markersGroupRef.current) {
      markersGroupRef.current.clearLayers();
    }

    const bounds = [];

    const createHtmlIcon = (color, label, isSelected = false) => {
      const size = isSelected ? 'w-8 h-8' : 'w-6 h-6';
      const shadow = isSelected ? 'shadow-lg shadow-isro-orange/50 scale-125' : '';
      const border = isSelected ? 'border-2 border-isro-orange' : 'border border-white/20';

      return L.divIcon({
        className: 'custom-div-icon',
        html: `
          <div class="relative flex items-center justify-center">
            <div class="${size} rounded-full flex items-center justify-center text-[10px] font-bold ${border} ${shadow}" style="background-color: ${color}; color: #ffffff;">
              ${label}
            </div>
            ${isSelected ? '<div class="absolute -inset-1 rounded-full border border-isro-orange animate-ping opacity-60"></div>' : ''}
          </div>
        `,
        iconSize: isSelected ? [32, 32] : [24, 24],
        iconAnchor: isSelected ? [16, 16] : [12, 12],
      });
    };

    if (queryCoordinates) {
      const queryIcon = createHtmlIcon('#F37023', 'Q', selectedResult?.isPerfectMatch);
      L.marker(queryCoordinates, { icon: queryIcon })
        .bindPopup(`<b>Query Anchor</b><br/>GPS: ${queryCoordinates[0].toFixed(4)}, ${queryCoordinates[1].toFixed(4)}`)
        .addTo(markersGroupRef.current);
      
      bounds.push(queryCoordinates);
      setLatVal(queryCoordinates[0].toFixed(4));
      setLngVal(queryCoordinates[1].toFixed(4));
    }

    if (results && results.length > 0) {
      results.forEach((result) => {
        if (result.isPerfectMatch && queryCoordinates) return;

        const isCurrentSelected = selectedResult?.id === result.id;
        const color = isCurrentSelected ? '#FF8F3D' : '#005EA6';
        const icon = createHtmlIcon(color, result.rank.toString(), isCurrentSelected);
        
        L.marker(result.coordinates, { icon })
          .bindPopup(`<b>Rank #${result.rank}: ${result.name}</b>`)
          .addTo(markersGroupRef.current);

        bounds.push(result.coordinates);
      });
    }

    if (bounds.length > 0) {
      mapInstanceRef.current.fitBounds(bounds, { padding: [50, 50] });
    }

    if (selectedResult) {
      mapInstanceRef.current.flyTo(selectedResult.coordinates, isFocused ? 14 : 12, {
        animate: true,
        duration: 1.5,
      });
      setLatVal(selectedResult.coordinates[0].toFixed(4));
      setLngVal(selectedResult.coordinates[1].toFixed(4));
    }

    setTimeout(() => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
      }
    }, 300);

  }, [queryCoordinates, results, selectedResult, isFocused]);

  useEffect(() => {
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  return (
    <div className="flex flex-col gap-4 h-full">
      <div className="flex justify-between items-center bg-[#EAF2FF] px-4 py-2 border-b border-border-light rounded-t-xl -mx-4 -mt-4 mb-2 select-none">
        <h2 className="text-xs font-bold text-primary-blue tracking-wider flex items-center gap-2 uppercase">
          <Navigation className="w-4 h-4 text-isro-orange" />
          Target Area Monitoring
        </h2>
        
        <div className="flex items-center gap-2">
          <div className="flex bg-white p-0.5 rounded border border-border-light shadow-sm">
            <button
              onClick={() => toggleMapLayer('satellite')}
              type="button"
              className={`p-1 rounded text-[8px] font-bold uppercase transition-all ${
                mapType === 'satellite' ? 'bg-primary-blue text-white' : 'text-slate-500 hover:text-primary-blue'
              }`}
            >
              <Globe className="w-3 h-3" />
            </button>
            <button
              onClick={() => toggleMapLayer('streets')}
              type="button"
              className={`p-1 rounded text-[8px] font-bold uppercase transition-all ${
                mapType === 'streets' ? 'bg-primary-blue text-white' : 'text-slate-500 hover:text-primary-blue'
              }`}
            >
              <Layers className="w-3 h-3" />
            </button>
          </div>

          {onToggleFocus && (
            <button
              onClick={onToggleFocus}
              type="button"
              className="p-1.5 rounded bg-white border border-border-light text-slate-500 hover:text-isro-orange hover:border-secondary-blue hover:shadow-sm transition-all"
            >
              {isFocused ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
            </button>
          )}
        </div>
      </div>

      {!isFocused && (
        <form onSubmit={handleCoordSubmit} className="grid grid-cols-2 gap-3 select-text">
          <div className="flex flex-col gap-1">
            <label className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Latitude</label>
            <input
              type="text"
              value={`${latVal}° N`}
              onChange={(e) => setLatVal(e.target.value.replace(/[^0-9.-]/g, ''))}
              className="bg-white border border-border-light rounded-lg h-9 px-3 text-xs text-slate-700 outline-none font-mono focus:border-secondary-blue transition-colors shadow-sm"
            />
          </div>
          <div className="flex flex-col gap-1 relative">
            <label className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Longitude</label>
            <div className="relative flex items-center bg-white border border-border-light focus-within:border-secondary-blue rounded-lg overflow-hidden h-9 px-3 transition-colors shadow-sm">
              <input
                type="text"
                value={`${lngVal}° E`}
                onChange={(e) => setLngVal(e.target.value.replace(/[^0-9.-]/g, ''))}
                className="bg-transparent text-xs text-slate-700 outline-none w-full font-mono"
              />
              <button 
                type="submit"
                className="absolute right-1 top-1 bottom-1 w-7 bg-secondary-blue hover:bg-primary-blue text-white rounded flex items-center justify-center transition-colors shadow-sm"
              >
                <Globe className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </form>
      )}

      <div 
        onClick={() => {
          if (!isFocused && onToggleFocus) {
            onToggleFocus();
          }
        }}
        className={`relative flex-grow rounded-xl overflow-hidden border border-border-light shadow-sm mt-1 ${
          !isFocused ? 'cursor-pointer hover:border-secondary-blue/40 transition-all duration-350' : ''
        }`}
      >
        <div ref={mapContainerRef} className="w-full h-full min-h-[220px] z-10" />

        {/* Animated Floating ISRO Logo with Orbit Rings Overlay */}
        <div className="absolute top-4 right-4 z-[400] pointer-events-none select-none animate-float bg-white/95 p-2 rounded-2xl border border-border-light shadow-md flex items-center justify-center gap-2">
          <div className="relative w-12 h-12 flex items-center justify-center">
            {/* Spinning Orbit Rings */}
            <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full animate-spin-slow text-primary-blue/35">
              <ellipse cx="50" cy="50" rx="45" ry="12" fill="none" stroke="currentColor" strokeWidth="1.2" transform="rotate(-30 50 50)" />
              <ellipse cx="50" cy="50" rx="45" ry="12" fill="none" stroke="currentColor" strokeWidth="1.2" strokeDasharray="2,3" transform="rotate(30 50 50)" />
              <circle cx="50" cy="50" r="3" fill="#F7941D" />
            </svg>
            {/* Floating central ISRO logo arrow */}
            <svg viewBox="0 0 40 40" className="w-7 h-7">
              <path d="M15,33 L20,7 L25,33 L20,29 Z" fill="#F7941D" />
              <path d="M6,23 Q18,-2 34,17" fill="none" stroke="#003B8E" strokeWidth="1.8" />
            </svg>
          </div>
          <div className="flex flex-col font-bold uppercase text-[8px] text-primary-blue tracking-wider leading-tight pr-1">
            <span>ISRO</span>
            <span className="text-[7px] text-slate-500 font-medium">Bhuvan Node</span>
          </div>
        </div>

        <div className="absolute bottom-3 left-3 z-[400] bg-white px-2.5 py-1 rounded border border-border-light text-[9px] font-mono text-isro-orange flex items-center gap-1.5 shadow-md font-bold">
          <span className="w-1.5 h-1.5 rounded-full bg-isro-orange animate-pulse" />
          Layer: {mapType.toUpperCase()}
        </div>
      </div>
    </div>
  );
}
