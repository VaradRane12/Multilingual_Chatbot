import { useEffect, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

const initialMessages = [
  {
    id: 1,
    role: 'assistant',
    content:
      'Hello! Try one of the Indian language examples below, or send your own message.',
    time: 'Now',
    meta: {
      detectedLang: 'en',
      intent: 'greeting',
      confidence: 1,
      translationBackend: 'none',
      translatedToPivot: false,
      translatedFromPivot: false,
    },
  },
]

const fallbackExamplePrompts = [
  'Hello, please summarize this paragraph.',
  'नमस्ते, क्या आप मेरी मदद कर सकते हैं?',
  'নমস্কার, এই লেখাটি সংক্ষেপে বলুন।',
  'வணக்கம், இந்த உரையை சுருக்கமாக கூறுங்கள்.',
  'ہیلو، براہ کرم اس متن کی وضاحت کریں۔',
]

const getNowTime = () =>
  new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })

function App() {
  const [messages, setMessages] = useState(initialMessages)
  const [draft, setDraft] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [examplePrompts, setExamplePrompts] = useState(fallbackExamplePrompts)

  useEffect(() => {
    let isMounted = true

    const loadExamples = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/examples`)
        if (!response.ok) {
          throw new Error('Failed to load examples')
        }

        const data = await response.json()
        const fromApi = (data.examples ?? [])
          .map((item) => {
            if (typeof item === 'string') {
              return item
            }
            return item?.text
          })
          .filter((text) => typeof text === 'string' && text.trim().length > 0)

        if (isMounted && fromApi.length > 0) {
          setExamplePrompts(fromApi)
        }
      } catch {
        // Keep fallback prompts when backend is unavailable.
      }
    }

    loadExamples()

    return () => {
      isMounted = false
    }
  }, [])

  const pushConversation = async (text) => {
    const cleanText = text.trim()
    if (!cleanText || isTyping) {
      return
    }

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: cleanText,
      time: getNowTime(),
    }

    setMessages((previous) => [...previous, userMessage])
    setDraft('')
    setIsTyping(true)

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: cleanText }),
      })

      if (!response.ok) {
        throw new Error('Chat request failed')
      }

      const data = await response.json()
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.assistant_message ?? 'Sorry, no response was returned.',
        time: getNowTime(),
        meta: {
          detectedLang: data.detected_lang ?? 'en',
          intent: data.intent ?? 'fallback',
          confidence: typeof data.confidence === 'number' ? data.confidence : 0,
          translationBackend: data.translation_backend ?? 'none',
          translatedToPivot: Boolean(data.translated_to_pivot),
          translatedFromPivot: Boolean(data.translated_from_pivot),
        },
      }

      setMessages((previous) => [...previous, assistantMessage])
    } catch {
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content:
          'Could not reach the backend API. Start FastAPI server and try again.',
        time: getNowTime(),
        meta: {
          detectedLang: 'en',
          intent: 'system_error',
          confidence: 0,
          translationBackend: 'none',
          translatedToPivot: false,
          translatedFromPivot: false,
        },
      }
      setMessages((previous) => [...previous, errorMessage])
    } finally {
      setIsTyping(false)
    }
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    pushConversation(draft)
  }

  const handlePromptClick = (prompt) => {
    pushConversation(prompt)
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      pushConversation(draft)
    }
  }

  return (
    <main className="chatbox-page">
      <section className="chatbox-card">
        <header className="chatbox-header">
          <h1>Multilingual Chatbox</h1>
          <p>Click an example or write your own message.</p>
        </header>

        <section className="examples" aria-label="example prompts">
          {examplePrompts.map((prompt) => (
            <button key={prompt} type="button" onClick={() => handlePromptClick(prompt)}>
              {prompt}
            </button>
          ))}
        </section>

        <section className="chat-thread" aria-live="polite" aria-label="chat thread">
          {messages.map((message, index) => (
            <article
              key={message.id}
              className={`message ${message.role}`}
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <p>{message.content}</p>
              {message.role === 'assistant' && message.meta && (
                <div className="message-meta">
                  <span>lang: {message.meta.detectedLang}</span>
                  <span>intent: {message.meta.intent}</span>
                  <span>conf: {Number(message.meta.confidence).toFixed(2)}</span>
                  <span>mt: {message.meta.translationBackend}</span>
                  <span>
                    pivot: {message.meta.translatedToPivot ? 'in' : 'skip'}/
                    {message.meta.translatedFromPivot ? 'out' : 'skip'}
                  </span>
                </div>
              )}
              <time>{message.time}</time>
            </article>
          ))}

          {isTyping && (
            <article className="message assistant typing-indicator" aria-label="assistant typing">
              <div>
                <span />
                <span />
                <span />
              </div>
            </article>
          )}
        </section>

        <form className="composer" onSubmit={handleSubmit} aria-label="message composer">
          <textarea
            placeholder="Send a message..."
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
          />

          <div className="composer-actions">
            <button type="submit" className="send-btn" disabled={!draft.trim() || isTyping}>
              Send
            </button>
          </div>
        </form>
      </section>
    </main>
  )
}

export default App
