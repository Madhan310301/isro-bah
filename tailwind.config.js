/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        space: {
          darkest: '#02050b',
          dark: '#0B1D3A',
          medium: '#1E293B',
          light: '#334155',
        },
        isro: {
          orange: '#F7941D',
          orangeLight: '#FFA834',
          blue: '#003B8E',
          blueLight: '#0E4FAF',
          gold: '#D4A843',
        },
        'primary-blue': '#003B8E',
        'secondary-blue': '#0E4FAF',
        'isro-orange': '#F7941D',
        'card-blue': '#D2E2F7',
        'bg-light': '#F5F8FC',
        'border-light': '#D6E3F5',
        columbia: {
          DEFAULT: '#B0CBEF',
          light: '#D2E2F7',
          dark: '#D2E2F7',
          darker: '#F5F8FC',
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
};
