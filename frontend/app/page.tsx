"use client"

import type React from "react"

import { useRouter } from "next/navigation"
import { FileText, Upload, Lock, Scale } from "lucide-react"
import { useState, useRef } from "react"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export default function HomePage() {
  const router = useRouter()
  const [isDragging, setIsDragging] = useState(false)
  const [isLoadingGDPR, setIsLoadingGDPR] = useState(false)
  const [isLoadingUpload, setIsLoadingUpload] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleUseGDPR = async () => {
    setIsLoadingGDPR(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/load-gdpr`, {
        method: "POST",
      })

      if (!response.ok) {
        throw new Error("Failed to load GDPR document")
      }

      const data = await response.json()
      router.push(`/chat?doc=gdpr.pdf&chunks=${data.chunks}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load document")
    } finally {
      setIsLoadingGDPR(false)
    }
  }

  const handleFileUpload = async (file: File) => {
    setIsLoadingUpload(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch(`${API_BASE_URL}/api/upload-document`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to upload document")
      }

      const data = await response.json()
      router.push(`/chat?doc=${encodeURIComponent(data.document)}&chunks=${data.chunks}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload document")
    } finally {
      setIsLoadingUpload(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && (file.type === "application/pdf" || file.type === "text/plain")) {
      handleFileUpload(file)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleBrowseClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50/50 to-white relative overflow-hidden">
      {/* Color Spots */}
      <div className="absolute top-20 left-90 w-96 h-96 bg-blue-200/15 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 left-150 w-96 h-96 bg-pink-200/15 rounded-full blur-3xl"></div>

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-white relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-50 rounded-full flex items-center justify-center">
            <Scale className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h1 className="font-semibold text-gray-900">Legal RAG Assistant</h1>
            <p className="text-sm text-gray-500">AI-powered legal document analysis</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-6 py-20 relative z-10">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4 text-balance">Understand Legal Documents with AI</h2>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto text-balance">
            Upload a legal document or use our pre-loaded GDPR file to start asking questions and get AI-powered answers
            with source citations.
          </p>
        </div>

        {error && (
          <div className="max-w-2xl mx-auto mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm">
            {error}
          </div>
        )}

        {/* Cards */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          {/* GDPR Document Card */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center mb-6">
              <FileText className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">GDPR Document</h3>
            <p className="text-gray-500 mb-8">
              Pre-loaded General Data Protection Regulation for quick start analysis.
            </p>
            <button
              onClick={handleUseGDPR}
              disabled={isLoadingGDPR || isLoadingUpload}
              className="w-full py-3 px-4 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-100 disabled:cursor-not-allowed rounded-full text-gray-700 font-medium transition-colors flex items-center justify-center gap-2"
            >
              {isLoadingGDPR ? (
                <>
                  <span className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></span>
                  Loading...
                </>
              ) : (
                <>
                  Use GDPR
                  <span>→</span>
                </>
              )}
            </button>
          </div>

          {/* Upload Document Card */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <div className="w-12 h-12 bg-purple-50 rounded-xl flex items-center justify-center mb-6">
              <Upload className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Upload Document</h3>
            <p className="text-gray-500 mb-6">Upload your own PDF or text file to analyze specific legal contracts.</p>
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={handleBrowseClick}
              className={`border-2 border-dashed rounded-full py-4 px-6 text-center cursor-pointer transition-colors ${
                isDragging ? "border-blue-400 bg-blue-50" : "border-gray-200 hover:border-gray-300"
              } ${isLoadingUpload || isLoadingGDPR ? "pointer-events-none opacity-50" : ""}`}
            >
              {isLoadingUpload ? (
                <p className="text-gray-600 text-sm font-medium">Uploading...</p>
              ) : (
                <>
                  <p className="text-gray-600 text-sm font-medium">Drag & drop or click to browse</p>
                  <p className="text-gray-400 text-xs">PDF, TXT up to 10MB</p>
                </>
              )}
            </div>
            <input ref={fileInputRef} type="file" accept=".pdf,.txt" onChange={handleFileChange} className="hidden" />
          </div>
        </div>

        {/* Privacy Notice */}
        <div className="flex items-center justify-center gap-2 text-gray-400 text-sm">
          <Lock className="w-4 h-4" />
          <span>Your documents are processed locally and never stored on our servers.</span>
        </div>
      </main>
    </div>
  )
}