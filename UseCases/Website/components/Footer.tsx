export default function Footer() {
  return (
    <footer className="mt-16 border-t bg-white">
      <div className="container py-6 text-sm text-gray-500 flex flex-col md:flex-row gap-3 md:items-center md:justify-between">
        <p>© {new Date().getFullYear()} GESTURE. All rights reserved.</p>
        <p>
          Built with <span className="text-[color:var(--brand)]">Next.js</span> & Tailwind. Theme: #22799B / #000 / #FFF.
        </p>
      </div>
    </footer>
  )
}
