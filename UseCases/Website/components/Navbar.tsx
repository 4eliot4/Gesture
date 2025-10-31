import Link from 'next/link'

export default function Navbar() {
  return (
    <header className="border-b bg-white">
      <div className="container flex items-center justify-between py-4">
        <Image src="/logo.svg" alt="GESTURE " width={80} height={80} />
        <nav className="flex items-center gap-6 text-sm">
          <Link href="/downloads" className="hover:underline">Downloads</Link>
          <Link href="/use-cases" className="hover:underline">Use Cases</Link>
          <a href="https://vercel.com" target="_blank" className="hover:underline">Deploy</a>
        </nav>
      </div>
    </header>
  )
}
