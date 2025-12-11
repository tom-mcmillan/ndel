'use client'

import { useState } from 'react'

type Language = 'python' | 'sql'

export default function Home() {
  const [source, setSource] = useState('')
  const [language, setLanguage] = useState<Language>('python')
  const [output, setOutput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runDescribe = async () => {
    if (!source.trim()) return
    setIsLoading(true)
    setError(null)
    try {
      const resp = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source, language })
      })

      const data = await resp.json()
      if (!resp.ok) {
        setError(data.error || 'Describe failed')
        setOutput('')
      } else {
        setOutput(data.output || '')
      }
    } catch (err) {
      setError('Unable to reach backend. Is api_server.py running on port 5000?')
      setOutput('')
    }
    setIsLoading(false)
  }

  const copyOutput = async () => {
    if (!output) return
    await navigator.clipboard.writeText(output)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="border-b bg-white">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">NDEL Playground</h1>
            <p className="text-sm text-gray-600">Post-facto, code-to-description DSL for Python and SQL</p>
          </div>
          <a
            href="https://github.com/tom-mcmillan/ndel"
            className="text-sm text-blue-600 hover:underline"
          >
            NDEL repo
          </a>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-6">
        <section className="bg-white rounded-xl shadow-sm border p-4 space-y-3">
          <h2 className="text-lg font-semibold">How it works</h2>
          <p className="text-sm text-gray-700">
            Paste Python or SQL that represents a data/ML pipeline. NDEL will statically analyze it and
            render a descriptive DSL showing datasets, transformations, features, models, and metrics. No code is executed.
          </p>
        </section>

        <section className="grid md:grid-cols-2 gap-4">
          <div className="bg-white rounded-xl shadow-sm border p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="space-x-2 text-sm">
                <label className="font-medium">Language:</label>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value as Language)}
                  className="border rounded px-2 py-1 text-sm"
                >
                  <option value="python">Python</option>
                  <option value="sql">SQL</option>
                </select>
              </div>
              <button
                onClick={runDescribe}
                disabled={isLoading || !source.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
              >
                {isLoading ? 'Describing…' : 'Describe'}
              </button>
            </div>
            <textarea
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder={language === 'python' ? 'Paste Python pipeline code…' : 'Paste SQL query…'}
              className="w-full h-96 border rounded-lg p-3 font-mono text-sm focus:outline-none focus:ring"
              spellCheck={false}
            />
          </div>

          <div className="bg-white rounded-xl shadow-sm border p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-sm">NDEL output</h3>
              <button
                onClick={copyOutput}
                disabled={!output}
                className="px-3 py-1 bg-gray-900 text-white rounded text-sm disabled:opacity-50"
              >
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
            {error && (
              <div className="text-sm text-red-600">{error}</div>
            )}
            <textarea
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              placeholder="NDEL description will appear here"
              className="w-full h-96 border rounded-lg p-3 font-mono text-sm bg-gray-50 focus:outline-none"
              spellCheck={false}
            />
          </div>
        </section>
      </main>
    </div>
  )
}
