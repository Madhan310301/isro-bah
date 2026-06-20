import { NextResponse } from 'next/server';

// Preset locations with SVG visual mockups of SAR and Optical images
const PRESETS = {
  bangalore: {
    id: 'bangalore',
    name: 'Bangalore Urban Area',
    useCase: 'Urban Development & Layout Analysis',
    coordinates: [12.9716, 77.5946],
    description: 'High-density urban sprawl showing major arterial roads, commercial zones, and surrounding lakes. SAR displays radar backscatter from concrete structures, while Optical displays typical red roofs and vegetation.',
    sar: createSvgSatellite('urban', 'sar'),
    optical: createSvgSatellite('urban', 'optical')
  },
  kerala: {
    id: 'kerala',
    name: 'Kerala Flood Zone (Kuttanad)',
    useCase: 'Disaster Response & Damage Assessment',
    coordinates: [9.4981, 76.3388],
    description: 'Low-lying agricultural backwater region. SAR penetrates cloud cover to clearly highlight flooded regions (black, flat surfaces), while Optical shows green vegetative canopy and flooded waterways.',
    sar: createSvgSatellite('flood', 'sar'),
    optical: createSvgSatellite('flood', 'optical')
  },
  rajasthan: {
    id: 'rajasthan',
    name: 'Bhadla Solar Park, Rajasthan',
    useCase: 'Infrastructure & Solar Resource Monitoring',
    coordinates: [27.3506, 72.0326],
    description: 'Massive solar energy harvesting facility in the Thar Desert. SAR highlights the metallic frames of solar arrays (strong bright reflections), while Optical shows the dark blue panels against yellow desert sand.',
    sar: createSvgSatellite('desert', 'sar'),
    optical: createSvgSatellite('desert', 'optical')
  },
  mumbai: {
    id: 'mumbai',
    name: 'Mumbai Harbor & Coastal Port',
    useCase: 'Maritime Security & Ship Monitoring',
    coordinates: [18.9500, 72.8500],
    description: 'Busy shipping lanes and docklands. SAR is highly sensitive to metallic hull reflections, showing ships as bright star-like points on a dark sea. Optical shows true ocean color and ship silhouettes.',
    sar: createSvgSatellite('coastal', 'sar'),
    optical: createSvgSatellite('coastal', 'optical')
  },
  sundarbans: {
    id: 'sundarbans',
    name: 'Sundarbans Mangrove Delta',
    useCase: 'Ecological Conservation & Wetland Forestry',
    coordinates: [21.9497, 89.1833],
    description: 'World\'s largest mangrove forest delta. SAR shows roughness of forest canopy vs smooth river channels (good for boundary detection). Optical shows dense dark green foliage and brown sediment rivers.',
    sar: createSvgSatellite('delta', 'sar'),
    optical: createSvgSatellite('delta', 'optical')
  }
};

// SVG Generator Helper
function createSvgSatellite(type, modality) {
  let colors = {};
  if (modality === 'sar') {
    // SAR is grayscale, grainy, radar reflections
    colors = {
      bg: '#111827',
      primary: '#D1D5DB', // metal/concrete
      secondary: '#4B5563', // forest/dirt
      water: '#030712', // smooth water is black in SAR
      active: '#FFFFFF' // high backscatter
    };
  } else {
    // Optical is natural color
    colors = {
      bg: '#1A361D', // green base
      primary: '#92400E', // brown roofs/roads
      secondary: '#15803D', // bright green vegetation
      water: '#1D4ED8', // blue water
      active: '#F59E0B' // gold solar panels/sand
    };
  }

  let content = '';

  if (type === 'urban') {
    if (modality === 'sar') {
      content = `
        <!-- Grids of streets -->
        <path d="M 0,50 L 256,50 M 0,150 L 256,150 M 50,0 L 50,256 M 180,0 L 180,256" stroke="${colors.primary}" stroke-width="2" stroke-dasharray="1,4" />
        <path d="M 0,100 L 256,120 M 100,0 L 120,256" stroke="${colors.active}" stroke-width="3" />
        <!-- Clusters of buildings (bright dots in SAR) -->
        <rect x="60" y="60" width="30" height="20" fill="none" stroke="${colors.active}" stroke-width="1.5" stroke-dasharray="2,2" />
        <circle cx="70" cy="70" r="2" fill="${colors.active}" />
        <circle cx="80" cy="65" r="1.5" fill="${colors.active}" />
        <circle cx="150" cy="80" r="3" fill="${colors.primary}" />
        <circle cx="160" cy="190" r="2.5" fill="${colors.active}" />
        <circle cx="210" cy="90" r="2" fill="${colors.active}" />
        <!-- Water body (smooth, absorbs radar, very dark) -->
        <path d="M 120,160 Q 140,140 180,160 T 240,150 L 256,160 L 256,256 L 120,256 Z" fill="${colors.water}" opacity="0.9" />
        <text x="10" y="240" fill="${colors.secondary}" font-size="8">S1-SAR (VV+VH)</text>
      `;
    } else {
      content = `
        <!-- Grids of streets -->
        <path d="M 0,50 L 256,50 M 0,150 L 256,150 M 50,0 L 50,256 M 180,0 L 180,256" stroke="${colors.primary}" stroke-width="3" />
        <path d="M 0,100 L 256,120 M 100,0 L 120,256" stroke="#475569" stroke-width="4" />
        <!-- Vegetation zones -->
        <circle cx="120" cy="80" r="40" fill="${colors.secondary}" opacity="0.6" filter="blur(4px)" />
        <!-- Buildings (red/brown blocks) -->
        <rect x="60" y="60" width="30" height="20" fill="${colors.primary}" rx="2" />
        <rect x="145" y="75" width="10" height="10" fill="#B45309" />
        <rect x="200" y="85" width="20" height="15" fill="${colors.primary}" />
        <rect x="150" y="180" width="25" height="25" fill="#9A3412" />
        <!-- Water body (blue) -->
        <path d="M 120,160 Q 140,140 180,160 T 240,150 L 256,160 L 256,256 L 120,256 Z" fill="${colors.water}" />
        <text x="10" y="240" fill="#94A3B8" font-size="8">S2-Optical (RGB)</text>
      `;
    }
  } else if (type === 'flood') {
    if (modality === 'sar') {
      content = `
        <!-- Low-backscatter flat flood water (pitch black) -->
        <rect x="0" y="0" width="256" height="256" fill="${colors.secondary}" />
        <path d="M 10,20 C 50,40 80,10 120,50 C 160,90 140,150 180,180 C 220,210 230,240 256,256 L 0,256 L 0,0 Z" fill="${colors.water}" />
        <!-- Non-flooded high ground (grainy grey structure) -->
        <path d="M 150,20 Q 200,60 220,10 T 256,70 L 256,0 Z" fill="${colors.primary}" opacity="0.4" />
        <circle cx="180" cy="35" r="4" fill="${colors.active}" />
        <circle cx="210" cy="25" r="3" fill="${colors.active}" />
        <text x="10" y="240" fill="${colors.primary}" font-size="8">S1-SAR (Flood Extent)</text>
      `;
    } else {
      content = `
        <!-- Lush vegetation background -->
        <rect x="0" y="0" width="256" height="256" fill="${colors.bg}" />
        <!-- River and flooded channels (blue-green mud) -->
        <path d="M 10,20 C 50,40 80,10 120,50 C 160,90 140,150 180,180 C 220,210 230,240 256,256 L 0,256 L 0,0 Z" fill="#1E40AF" />
        <!-- Silt / sediment deposits -->
        <path d="M 60,68 Q 80,80 100,65" stroke="#D97706" stroke-width="2" fill="none" opacity="0.5" />
        <!-- Dry land clusters (green fields & homes) -->
        <path d="M 150,20 Q 200,60 220,10 T 256,70 L 256,0 Z" fill="${colors.secondary}" />
        <rect x="180" y="25" width="6" height="6" fill="#F59E0B" />
        <text x="10" y="240" fill="#94A3B8" font-size="8">S2-Optical (True Color)</text>
      `;
    }
  } else if (type === 'desert') {
    if (modality === 'sar') {
      content = `
        <!-- Flat sand background (dark noise) -->
        <rect x="0" y="0" width="256" height="256" fill="${colors.bg}" />
        <!-- Sand dunes ridges (subtle wave backscatter) -->
        <path d="M -20,80 Q 80,120 180,60 T 280,90" stroke="${colors.secondary}" stroke-width="2" fill="none" stroke-dasharray="2,3" />
        <path d="M -20,160 Q 90,200 190,140 T 280,170" stroke="${colors.secondary}" stroke-width="2" fill="none" stroke-dasharray="2,3" />
        <!-- Solar panel arrays: Metal frame reflections (High double-bounce, very bright) -->
        <g fill="none" stroke="${colors.active}" stroke-width="1.5">
          <line x1="40" y1="40" x2="140" y2="40" stroke-dasharray="2,2" />
          <line x1="40" y1="50" x2="140" y2="50" stroke-dasharray="2,2" />
          <line x1="40" y1="60" x2="140" y2="60" stroke-dasharray="2,2" />
          <line x1="40" y1="70" x2="140" y2="70" stroke-dasharray="2,2" />
          
          <line x1="60" y1="120" x2="200" y2="120" stroke-dasharray="2,2" />
          <line x1="60" y1="130" x2="200" y2="130" stroke-dasharray="2,2" />
          <line x1="60" y1="140" x2="200" y2="140" stroke-dasharray="2,2" />
        </g>
        <text x="10" y="240" fill="${colors.primary}" font-size="8">S1-SAR (Polarimetric)</text>
      `;
    } else {
      content = `
        <!-- Sandy orange-yellow desert -->
        <rect x="0" y="0" width="256" height="256" fill="#FCD34D" />
        <!-- Sand dunes shadow lines -->
        <path d="M -20,80 Q 80,120 180,60 T 280,90" stroke="#F59E0B" stroke-width="4" fill="none" opacity="0.3" />
        <path d="M -20,160 Q 90,200 190,140 T 280,170" stroke="#F59E0B" stroke-width="4" fill="none" opacity="0.3" />
        <!-- Solar Panels (Dark blue glass grids) -->
        <g fill="${colors.water}" stroke="#475569" stroke-width="0.5">
          <rect x="40" y="38" width="100" height="34" rx="1" fill="#1E3A8A" />
          <rect x="60" y="118" width="140" height="26" rx="1" fill="#1E3A8A" />
        </g>
        <!-- Power substation -->
        <rect x="210" y="40" width="20" height="20" fill="#94A3B8" />
        <line x1="140" y1="55" x2="210" y2="50" stroke="#475569" stroke-width="1" />
        <text x="10" y="240" fill="#78350F" font-size="8">S2-Optical (Solar Array)</text>
      `;
    }
  } else if (type === 'coastal') {
    if (modality === 'sar') {
      content = `
        <!-- Water background (flat, absorbs radar, black) -->
        <rect x="0" y="0" width="256" height="256" fill="${colors.water}" />
        <!-- Rough sea wave patterns -->
        <path d="M 0,200 Q 50,195 100,200 T 200,195 T 256,200" stroke="#1F2937" stroke-width="1" fill="none" />
        <!-- Coastal shore wall (strong returns, bright white/grey) -->
        <path d="M 0,80 L 100,80 L 180,140 L 256,140" stroke="${colors.primary}" stroke-width="3" fill="none" />
        <rect x="30" y="40" width="40" height="40" fill="${colors.secondary}" opacity="0.3" />
        <!-- Ship targets (dihedral radar reflections: bright cross shapes) -->
        <!-- Ship 1 -->
        <g transform="translate(80, 160)">
          <path d="M -5,0 L 5,0 M 0,-5 L 0,5" stroke="${colors.active}" stroke-width="1.5" />
          <circle cx="0" cy="0" r="2" fill="${colors.active}" />
        </g>
        <!-- Ship 2 -->
        <g transform="translate(160, 220)">
          <path d="M -4,0 L 4,0 M 0,-4 L 0,4" stroke="${colors.active}" stroke-width="1.2" />
          <circle cx="0" cy="0" r="1.5" fill="${colors.active}" />
        </g>
        <!-- Ship 3 -->
        <g transform="translate(210, 110)">
          <circle cx="0" cy="0" r="1.5" fill="${colors.primary}" />
        </g>
        <text x="10" y="240" fill="${colors.primary}" font-size="8">S1-SAR (Coastal Guard)</text>
      `;
    } else {
      content = `
        <!-- Sea Water (deep blue) -->
        <rect x="0" y="0" width="256" height="256" fill="${colors.water}" />
        <!-- Land area -->
        <path d="M 0,80 L 100,80 L 180,140 L 256,140 L 256,0 L 0,0 Z" fill="#064E3B" />
        <!-- Port structures & buildings -->
        <rect x="30" y="40" width="40" height="40" fill="#334155" />
        <rect x="40" y="50" width="10" height="15" fill="#D4A843" />
        <rect x="120" y="60" width="25" height="15" fill="#475569" />
        <!-- Ship targets -->
        <!-- Ship 1 (White hull, wake behind it) -->
        <g transform="translate(80, 160) rotate(15)">
          <path d="M -10,-1 C -5,-4 5,-4 10,-1 L 8,3 C 3,4 -3,4 -8,3 Z" fill="#FFFFFF" />
          <path d="M -10,-1 L -25,-5 M -10,1 L -25,5" stroke="#FFFFFF" stroke-width="1" opacity="0.4" />
        </g>
        <!-- Ship 2 -->
        <g transform="translate(160, 220) rotate(-30)">
          <path d="M -8,-1 C -4,-3 4,-3 8,-1 L 7,2 C 2,3 -2,3 -6,2 Z" fill="#EF4444" />
        </g>
        <text x="10" y="240" fill="#94A3B8" font-size="8">S2-Optical (Coastal Port)</text>
      `;
    }
  } else {
    // Delta / Forestry
    if (modality === 'sar') {
      content = `
        <!-- Grainy canopy texture (forest) -->
        <rect x="0" y="0" width="256" height="256" fill="${colors.secondary}" />
        <!-- Smooth rivers (very dark) -->
        <path d="M 100,0 C 80,60 180,120 120,180 C 90,210 50,200 40,256" stroke="${colors.water}" stroke-width="16" fill="none" stroke-linecap="round" />
        <path d="M 150,90 Q 220,100 256,120" stroke="${colors.water}" stroke-width="8" fill="none" stroke-linecap="round" />
        <path d="M 90,140 Q 30,160 0,150" stroke="${colors.water}" stroke-width="6" fill="none" stroke-linecap="round" />
        <text x="10" y="240" fill="${colors.primary}" font-size="8">S1-SAR (Coherent Radar)</text>
      `;
    } else {
      content = `
        <!-- Rich mangrove canopy (dark forest green) -->
        <rect x="0" y="0" width="256" height="256" fill="#064E3B" />
        <!-- Silt rivers (muddy brown-green) -->
        <path d="M 100,0 C 80,60 180,120 120,180 C 90,210 50,200 40,256" stroke="#78350F" stroke-width="16" fill="none" stroke-linecap="round" />
        <path d="M 150,90 Q 220,100 256,120" stroke="#78350F" stroke-width="8" fill="none" stroke-linecap="round" />
        <path d="M 90,140 Q 30,160 0,150" stroke="#78350F" stroke-width="6" fill="none" stroke-linecap="round" />
        <!-- Canopy variations -->
        <circle cx="50" cy="50" r="30" fill="#047857" opacity="0.3" />
        <circle cx="210" cy="180" r="40" fill="#022C22" opacity="0.4" />
        <text x="10" y="240" fill="#94A3B8" font-size="8">S2-Optical (Bio-diversity)</text>
      `;
    }
  }

  // Combine into SVG wrapper
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="100%" height="100%">
    <rect width="256" height="256" fill="${colors.bg}" />
    <!-- Noise filter simulation -->
    <filter id="noise">
      <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch" />
      <feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.15 0" />
      <feBlend mode="overlay" in2="SourceGraphic" />
    </filter>
    <rect width="256" height="256" filter="url(#noise)" opacity="${modality === 'sar' ? 0.35 : 0.1}" pointer-events="none" />
    ${content}
  </svg>`;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

// Generates secondary search matches based on a target match
function generateSearchResults(targetPreset, queryModality, count = 10) {
  const targetModality = queryModality === 'sar' ? 'optical' : 'sar';
  
  // The first match is the true paired image at that coordinate
  const results = [
    {
      rank: 1,
      score: 0.9482,
      id: targetPreset.id,
      name: targetPreset.name,
      useCase: targetPreset.useCase,
      coordinates: targetPreset.coordinates,
      image: targetPreset[targetModality],
      isPerfectMatch: true,
      satellite: targetModality === 'sar' ? 'Sentinel-1 (SAR)' : 'Sentinel-2 (Optical)'
    }
  ];

  // Rest are similar locations with slightly altered coordinates/visuals
  const presetsList = Object.values(PRESETS).filter(p => p.id !== targetPreset.id);
  
  for (let i = 1; i < count; i++) {
    // Mix other presets or generate nearby synthetic coords
    const template = presetsList[i % presetsList.length];
    const latOffset = (Math.random() - 0.5) * 0.05;
    const lngOffset = (Math.random() - 0.5) * 0.05;
    
    // Similarity score decrements with rank
    const score = 0.85 - (i * 0.04) - (Math.random() * 0.02);

    results.push({
      rank: i + 1,
      score: parseFloat(score.toFixed(4)),
      id: `${template.id}_alt_${i}`,
      name: `${template.name} (Sector ${String.fromCharCode(65 + i)})`,
      useCase: template.useCase,
      coordinates: [template.coordinates[0] + latOffset, template.coordinates[1] + lngOffset],
      image: template[targetModality], // Retrieve opposing sensor representation
      isPerfectMatch: false,
      satellite: targetModality === 'sar' ? 'Sentinel-1 (SAR)' : 'Sentinel-2 (Optical)'
    });
  }

  return results;
}

export async function POST(req) {
  try {
    const formData = await req.formData();
    const file = formData.get('file');
    const modality = formData.get('modality') || 'sar';
    const presetId = formData.get('presetId');

    // Simulate model inference and FAISS search delay (800ms)
    await new Promise((resolve) => setTimeout(resolve, 800));

    let selectedPreset = PRESETS.bangalore; // fallback
    
    if (presetId && PRESETS[presetId]) {
      selectedPreset = PRESETS[presetId];
    } else if (file) {
      // If user uploaded a custom file, pick a preset based on the file name hash or random
      const hash = file.name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
      const keys = Object.keys(PRESETS);
      const chosenKey = keys[hash % keys.length];
      selectedPreset = PRESETS[chosenKey];
    }

    const results = generateSearchResults(selectedPreset, modality);

    return NextResponse.json({
      success: true,
      queryImage: selectedPreset[modality],
      queryModality: modality,
      queryCoordinates: selectedPreset.coordinates,
      queryName: selectedPreset.name,
      results: results
    });
  } catch (error) {
    console.error('Search API Error:', error);
    return NextResponse.json(
      { success: false, error: 'Internal Server Error during cross-modal search' },
      { status: 500 }
    );
  }
}
