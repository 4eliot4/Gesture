import './globals.css'
import type { Metadata } from 'next'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'

export const metadata: Metadata = {
  title: 'GESTURE',
  description: 'Download models and explore ASL-powered use cases (HAND & SIGNIT).',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main className="container py-10">{children}</main>
        <Footer />
      </body>
    </html>
  )
}
