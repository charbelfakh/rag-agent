import { gql } from '@apollo/client'
import { useSubscription } from '@apollo/client/react'
import { useState, type FormEvent } from 'react'
import type { Source } from './types'

// Corpus vendors + document types mirror the main UI's dropdowns
// (`ui/index.html` chat toolbar + upload modal) so the two clients offer the
// same query-scoping options. Both filters map to args the GraphQL `askStream`
// subscription already accepts.
const VENDORS = ['pekat', 'mechmind', 'zivid', 'lmi', 'basler', 'photoneo'] as const

const DOC_TYPES: ReadonlyArray<readonly [string, string]> = [
  ['user_manual', 'User manual'],
  ['quick_start', 'Quick start'],
  ['api_reference', 'API reference'],
  ['datasheet', 'Datasheet'],
  ['release_notes', 'Release notes'],
  ['integration_guide', 'Integration guide'],
  ['troubleshooting', 'Troubleshooting'],
  ['specification', 'Specification'],
]

const TOP_K_OPTIONS = [3, 5, 8, 10] as const

const ASK_STREAM = gql`
  subscription AskStream(
    $question: String!
    $vendor_filter: String
    $document_type_filter: String
    $top_k: Int
  ) {
    askStream(
      question: $question
      vendor_filter: $vendor_filter
      document_type_filter: $document_type_filter
      top_k: $top_k
    ) {
      token
      done
      error
      sources {
        source
        file_name
        vendor
        page
        section
        content_type
        text
      }
    }
  }
`

interface AskStreamEvent {
  token?: string | null
  done: boolean
  error?: string | null
  sources?: Source[] | null
}

interface AskStreamData {
  askStream: AskStreamEvent
}

interface AskStreamVariables {
  question: string
  vendor_filter?: string | null
  document_type_filter?: string | null
  top_k?: number | null
}

function sourceDetail(source: Source): string {
  return [
    source.vendor,
    source.page != null ? `p. ${source.page}` : null,
    source.section,
    source.content_type,
  ]
    .filter(Boolean)
    .join(' · ')
}

const selectClasses =
  'rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500'

/**
 * Owns one streaming run. Mounted with a fresh `key` per submission so that
 * re-asking the identical question restarts the subscription cleanly.
 */
function AnswerStream({ variables }: { variables: AskStreamVariables }) {
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [streaming, setStreaming] = useState(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  useSubscription<AskStreamData, AskStreamVariables>(ASK_STREAM, {
    variables,
    onData: ({ data }) => {
      const event = data.data?.askStream
      if (!event) return
      if (event.token != null) setAnswer((prev) => prev + event.token)
      if (event.done) {
        if (event.error) setErrorMessage(event.error)
        if (event.sources) setSources(event.sources)
        setStreaming(false)
      }
    },
    onError: (err) => {
      setErrorMessage(err.message)
      setStreaming(false)
    },
    // The server may close the stream without a done event (e.g. cancelled
    // generation); make sure the cursor stops blinking either way.
    onComplete: () => {
      setStreaming(false)
    },
  })

  return (
    <section className="flex flex-col gap-4">
      {errorMessage && (
        <div
          className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-800"
          role="alert"
        >
          {errorMessage}
        </div>
      )}

      {streaming && answer.length === 0 && !errorMessage && (
        <p className="text-gray-500" aria-live="polite">
          Thinking…
        </p>
      )}

      {answer.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white px-4 py-3 whitespace-pre-wrap text-gray-900 shadow-sm">
          {answer}
          {streaming && <span className="animate-pulse">▋</span>}
        </div>
      )}

      {sources.length > 0 && (
        <div>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
            Sources
          </h2>
          <ul className="flex flex-col gap-2">
            {sources.map((source, index) => {
              const detail = sourceDetail(source)
              const snippet = source.text?.trim()
              return (
                <li
                  key={index}
                  className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm"
                >
                  <span className="font-medium text-gray-900">
                    {source.file_name || source.source}
                  </span>
                  {detail && (
                    <span className="mt-0.5 block text-gray-500">{detail}</span>
                  )}
                  {snippet && (
                    <p className="mt-1 line-clamp-3 text-gray-600">{snippet}</p>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </section>
  )
}

function App() {
  const [question, setQuestion] = useState('')
  const [vendorFilter, setVendorFilter] = useState('')
  const [docTypeFilter, setDocTypeFilter] = useState('')
  const [topK, setTopK] = useState(5)
  const [run, setRun] = useState<{ id: number; variables: AskStreamVariables } | null>(
    null,
  )

  const trimmed = question.trim()
  const canSubmit = trimmed.length > 0

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (!canSubmit) return
    setRun((prev) => ({
      id: (prev?.id ?? 0) + 1,
      variables: {
        question: trimmed,
        vendor_filter: vendorFilter || null,
        document_type_filter: docTypeFilter || null,
        top_k: topK,
      },
    }))
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-2xl flex-col gap-6 px-4 py-10">
      <header>
        <h1 className="text-2xl font-semibold text-gray-900">RAG Agent</h1>
        <p className="mt-1 text-sm text-gray-500">
          Ask a question about your document corpus.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="flex flex-col gap-2">
        <div className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question…"
            className="min-w-0 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            Ask
          </button>
        </div>

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          <label className="flex flex-col gap-1 text-xs font-medium text-gray-500">
            Vendor
            <select
              value={vendorFilter}
              onChange={(e) => setVendorFilter(e.target.value)}
              className={selectClasses}
            >
              <option value="">All vendors</option>
              {VENDORS.map((vendor) => (
                <option key={vendor} value={vendor}>
                  {vendor}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-xs font-medium text-gray-500">
            Document type
            <select
              value={docTypeFilter}
              onChange={(e) => setDocTypeFilter(e.target.value)}
              className={selectClasses}
            >
              <option value="">All types</option>
              {DOC_TYPES.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-xs font-medium text-gray-500">
            Results (top-k)
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className={selectClasses}
            >
              {TOP_K_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        </div>
      </form>

      {run && <AnswerStream key={run.id} variables={run.variables} />}
    </div>
  )
}

export default App
