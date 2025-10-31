import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          blue: '#22799B',
          black: '#000000',
          white: '#FFFFFF'
        }
      },
      boxShadow: {
        soft: '0 8px 24px rgba(0,0,0,0.08)'
      },
      borderRadius: {
        xl2: '1.25rem'
      }
    },
  },
  plugins: [],
}
export default config