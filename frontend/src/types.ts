export interface Source {
  source: string
  vendor?: string | null
  page?: number | null
  content_type?: string | null
}

export interface AskResult {
  answer: string
  sources: Source[]
}

export interface AskQueryData {
  ask: AskResult
}

export interface AskQueryVariables {
  question: string
}
