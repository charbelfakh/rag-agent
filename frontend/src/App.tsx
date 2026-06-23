import { gql } from '@apollo/client'
import { useLazyQuery } from '@apollo/client/react'
import { useState, type FormEvent } from 'react'
import type { AskQueryData, AskQueryVariables, Source } from './types'

const ASK_QUERY = gql`
  query Ask($question: String!) {
    ask(question: $question) {
      answer
      sources {
        source
        vendor
        page
        content_type
      }
    }
  }
`

function sourceDetail(source: Source): string {
  return [source.vendor, source.page != null ? `p. ${source.page}` : null, source.content_type]
    .filter(Boolean)
    .join(' · ')
}

function App() {
  const [question, setQuestion] = useState('')
  const [ask, { data, loading, error, called }] = useLazyQuery<
    AskQueryData,
    AskQueryVariables
  >(ASK_QUERY)

  const trimmed = question.trim()
  const canSubmit = trimmed.length > 0 && !loading

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (!canSubmit) return
    void ask({ variables: { question: trimmed } })
  }

  const sources = data?.ask.sources ?? []

  return (
    <div className="mx-auto flex min-h-screen max-w-2xl flex-col gap-6 px-4 py-10">
      <header>
        <h1 className="text-2xl font-semibold text-gray-900">RAG Agent</h1>
        <p className="mt-1 text-sm text-gray-500">Ask a question about your document corpus.</p>
      </header>

      <form onSubmit={handleSubmit} className="flex gap-2">
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
      </form>

      {called && (
        <section className="flex flex-col gap-4">
          {loading && (
            <p className="text-gray-500" aria-live="polite">
              Thinking…
            </p>
          )}

          {!loading && error && (
            <div
              className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-800"
              role="alert"
            >
              {error.message}
            </div>
          )}

          {!loading && !error && data?.ask && (
            <>
              <div className="rounded-lg border border-gray-200 bg-white px-4 py-3 whitespace-pre-wrap text-gray-900 shadow-sm">
                {data.ask.answer}
              </div>

              {sources.length > 0 && (
                <div>
                  <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
                    Sources
                  </h2>
                  <ul className="flex flex-col gap-2">
                    {sources.map((source, index) => {
                      const detail = sourceDetail(source)
                      return (
                        <li
                          key={index}
                          className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm"
                        >
                          <span className="font-medium text-gray-900">{source.source}</span>
                          {detail && (
                            <span className="mt-0.5 block text-gray-500">{detail}</span>
                          )}
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}
            </>
          )}
        </section>
      )}
    </div>
  )
}

export default App
