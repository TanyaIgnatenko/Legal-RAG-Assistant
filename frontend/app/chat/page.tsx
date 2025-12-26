"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { ArrowLeft, Settings, Send, Bot, Sparkles, User, ChevronDown, ChevronRight, Scale } from "lucide-react"
import ReactMarkdown from "react-markdown"

interface Message {
  role: "user" | "assistant"
  content: string
  chunks?: Array<{
    metadata: Record<string, unknown>
    text: string
    score: number
  }>
}

const exampleQuestions = [
  "What are the main principles of data processing under GDPR?",
  "What rights do data subjects have?",
  "What are the penalties for GDPR violations?",
  "When is a Data Protection Officer required?",
  "How long does an organization have to report a data breach?",
]

export default function ChatPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const doc = searchParams.get("doc") || "gdpr.pdf"
  const chunksCount = searchParams.get("chunks") || "26"
  const docName = doc

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [expandedChunks, setExpandedChunks] = useState<Record<number, boolean>>({})
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (question: string) => {
    if (!question.trim()) return

    const userMessage: Message = { role: "user", content: question }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
          top_k: 3,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to get answer")
      }

      const data = await response.json()

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        chunks: data.chunks,
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        role: "assistant",
        content:
          "Sorry, I encountered an error while processing your question. Please make sure the backend server is running and try again.",
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleQuestionClick = (question: string) => {
    handleSend(question)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    handleSend(input)
  }

  const toggleChunks = (messageIndex: number) => {
    setExpandedChunks((prev) => ({
      ...prev,
      [messageIndex]: !prev[messageIndex],
    }))
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50/30 to-white flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-white">
        <div className="flex items-center gap-4">
          <button onClick={() => router.push("/")} className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5 text-gray-500" />
          </button>
          <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-50 rounded-full flex items-center justify-center">
            <Scale className="w-6 h-6 text-blue-600" />
          </div>
            <div>
              <h1 className="font-semibold text-gray-900">Legal RAG Assistant</h1>
              <p className="text-sm text-gray-500">
                <span className="inline-flex items-center gap-1">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-gray-400">
                    <path
                      d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <polyline
                      points="14,2 14,8 20,8"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  {docName}
                </span>
              </p>
            </div>
          </div>
        </div>
        <button className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
          <Settings className="w-5 h-5 text-gray-500" />
        </button>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-3xl mx-auto">
          {/* Welcome Section - Always visible */}
          <div className="text-center mb-8">
            <div className="w-14 h-14 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Bot className="w-7 h-7 text-blue-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Ready to analyze {docName}</h2>
            <p className="text-gray-500 flex items-center justify-center gap-1">
              <Sparkles className="w-4 h-4" />
              Try asking one of these questions:
            </p>
          </div>

          {/* Example Questions - Always visible */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
            {exampleQuestions.slice(0, 4).map((question, index) => (
              <button
                key={index}
                onClick={() => handleQuestionClick(question)}
                className="p-4 text-left bg-white border border-gray-200 rounded-2xl hover:border-gray-300 hover:shadow-sm transition-all text-gray-700 text-sm"
              >
                {question}
              </button>
            ))}
          </div>
          <div className="mb-8">
            <button
              onClick={() => handleQuestionClick(exampleQuestions[4])}
              className="w-full p-4 text-left bg-white border border-gray-200 rounded-2xl hover:border-gray-300 hover:shadow-sm transition-all text-gray-700 text-sm"
            >
              {exampleQuestions[4]}
            </button>
          </div>

          {/* Messages */}
          {messages.length > 0 && (
            <div className="space-y-6 mb-8">
              {messages.map((message, index) => (
                <div key={index} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                  {message.role === "assistant" && (
                    <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Bot className="w-4 h-4 text-blue-600" />
                    </div>
                  )}
                  <div
                    className={`max-w-xl px-4 py-3 rounded-2xl ${
                      message.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-white border border-gray-200 text-gray-700"
                    }`}
                  >
                    {message.role === "assistant" ? (
                      <div className="text-sm prose prose-sm prose-gray max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-2 prose-strong:text-gray-900">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    )}
                    {message.chunks && message.chunks.length > 0 && (
                      <div className="mt-4 border-t border-gray-100 pt-3">
                        <button
                          onClick={() => toggleChunks(index)}
                          className="flex items-center gap-2 text-gray-600 hover:text-gray-800 font-medium text-sm transition-colors"
                        >
                          {expandedChunks[index] ? (
                            <ChevronDown className="w-4 h-4" />
                          ) : (
                            <ChevronRight className="w-4 h-4" />
                          )}
                          <span>Sources ({message.chunks.length})</span>
                        </button>
                        {expandedChunks[index] && (
                          <ul className="mt-3 space-y-3">
                            {message.chunks.map((chunk, chunkIndex) => (
                              <li
                                key={chunkIndex}
                                className="text-gray-500 text-xs bg-gray-50 p-3 rounded-lg border border-gray-100"
                              >
                                <p className="line-clamp-4">{chunk.text}</p>
                                {chunk.score !== undefined && (
                                  <p className="text-gray-400 text-xs mt-2">
                                    Relevance: {chunk.score.toFixed(1)}%
                                  </p>
                                )}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    )}
                  </div>
                  {message.role === "user" && (
                    <div className="w-8 h-8 bg-gray-200 rounded-lg flex items-center justify-center flex-shrink-0">
                      <User className="w-4 h-4 text-gray-600" />
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-blue-600" />
                  </div>
                  <div className="bg-white border border-gray-200 px-4 py-3 rounded-2xl">
                    <div className="flex gap-1">
                      <span
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      ></span>
                      <span
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      ></span>
                      <span
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      ></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      {/* Input Area */}
      <div className="border-t border-gray-100 bg-white px-6 py-4">
        <div className="max-w-3xl mx-auto">
          <form onSubmit={handleSubmit} className="relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about the document..."
              className="w-full px-5 py-4 pr-14 bg-gray-50 border border-gray-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 placeholder-gray-400"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 rounded-xl flex items-center justify-center transition-colors"
            >
              <Send className="w-5 h-5 text-white" />
            </button>
          </form>
          <p className="text-center text-xs text-gray-400 mt-3">
            AI responses may vary. Check important information against the source document.
          </p>
        </div>
      </div>
    </div>
  )
}
