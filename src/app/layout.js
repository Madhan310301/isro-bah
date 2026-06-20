import './globals.css';

export const metadata = {
  title: 'ISRO Cross-Modal Satellite Image Retrieval Dashboard',
  description: 'Cross-modal image retrieval using Sentinel-1 (SAR) and Sentinel-2 (Optical) remote sensing data. Developed for Bharatiya Antariksh Hackathon 2026.',
  keywords: ['satellite', 'remote sensing', 'SAR', 'optical', 'cross-modal retrieval', 'deep learning', 'ISRO', 'BAH 2026'],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
      </head>
      <body className="antialiased min-h-screen bg-bg-light text-slate-800 selection:bg-isro-orange selection:text-white">
        <div className="relative z-10 min-h-screen flex flex-col justify-between">
          {children}
        </div>
      </body>
    </html>
  );
}
