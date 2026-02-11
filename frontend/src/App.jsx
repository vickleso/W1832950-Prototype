import { useState } from 'react'
import './App.css'
import axios from 'axios'

function App() {
  const [tweetUrl, setTweetUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const analyzeTweet = async () => {
    if (!tweetUrl.trim()) {
      setError('Please enter a tweet URL')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post('http://localhost:8000/analyse', {
        url: tweetUrl
      })

      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyse tweet')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      analyzeTweet()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>X Misinformation Detector</h1>
        <p>Paste a tweet link to analyse it for misinformation</p>
      </header>

      <div className="main-container">
        {/* Left Panel - Input */}
        <div className="input-panel">
          <h2>Enter Tweet URL</h2>
          
          <input
            type="text"
            className="url-input"
            placeholder="https://x.com/username/status/123456789"
            value={tweetUrl}
            onChange={(e) => setTweetUrl(e.target.value)}
            onKeyDown={handleKeyPress}
          />

          <button 
            className="analyse-btn" 
            onClick={analyzeTweet}
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Analyse Tweet'}
          </button>

          {error && (
            <div className="error-box">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="results-panel">
          <h2>Analysis Results</h2>

          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Analyzing tweet</p>
            </div>
          )}

          {result && !loading && (
            <div className="result-content">
              <div className="classification-badge">
                <span className={result.classification === 'Misinformation' ? 'badge-fake' : 'badge-real'}>
                  {result.classification}
                </span>
                <span className="confidence">
                  Confidence: {(result.confidence * 100).toFixed(0)}%
                </span>
              </div>

              <div className="result-section">
                <h3>Tweet Information</h3>
                <p><strong>Author:</strong> @{result.author}</p>
                <p><strong>Text:</strong> {result.text}</p>
              </div>

              <div className="result-section">
                <h3>Engagement Metrics</h3>
                <p><strong>Likes:</strong> {result.likes?.toLocaleString()}</p>
                <p><strong>Retweets:</strong> {result.retweets?.toLocaleString()}</p>
              </div>

              <div className="result-section">
                <h3>Explanation</h3>
                <p className="explanation">
                  {result.classification === 'Misinformation' 
                    ? 'This post has been classified as potentially misleading or false based on analysis of its content.'
                    : 'This post appears to contain factual information based on the analysis.'}
                </p>
              </div>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="placeholder">
              <p>Enter a tweet URL and click Analyse to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
