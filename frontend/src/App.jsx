import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const suggestions = [
  { icon: '✦', title: 'Explain a complex idea', prompt: 'Explain how an inference gateway chooses the best model for a request.' },
  { icon: '⌘', title: 'Help me write code', prompt: 'Write a Python function that retries a request with exponential backoff.' },
  { icon: '↗', title: 'Summarize a document', prompt: 'Summarize the key points and action items in this document.' },
]

function createConversationId() {
  return globalThis.crypto.randomUUID()
}
const demoPlans = [
  {
    id: 'usage',
    name: 'Pay as you go',
    price: 'Usage-based',
    description: 'Pay only for the model usage you need.',
  },
  {
    id: 'pro',
    name: 'Pro',
    price: '₹2000 / month',
    description: 'Higher usage limits for individual workflows.',
    popular: true,
  },
  {
    id: 'team',
    name: 'Team',
    price: '₹5000 / user / month',
    description: 'Shared access and controls for a team.',
  },
]

function fileToAttachment(file) {
  const extension = file.name.split('.').pop()?.toLowerCase()
  const inferredType = {
    pdf: 'application/pdf',
    txt: 'text/plain',
    md: 'text/markdown',
    csv: 'text/csv',
    json: 'application/json',
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    webp: 'image/webp',
    gif: 'image/gif',
  }[extension]
  const mimeType = file.type || inferredType || 'application/octet-stream'

  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`))
    reader.onload = () => {
      if (mimeType.startsWith('text/') || mimeType === 'application/json') {
        resolve({ name: file.name, mime_type: mimeType, text: reader.result })
      } else {
        resolve({ name: file.name, mime_type: mimeType, data: reader.result })
      }
    }
    if (mimeType.startsWith('text/') || mimeType === 'application/json') {
      reader.readAsText(file)
    } else {
      reader.readAsDataURL(file)
    }
  })
}

function App() {
  const [conversationId, setConversationId] = useState(createConversationId)
  const [prompt, setPrompt] = useState('')
  const [mode, setMode] = useState('fast')
  const [files, setFiles] = useState([])
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [judgeOpen, setJudgeOpen] = useState(false)
  const [judgeModels, setJudgeModels] = useState([])
  const [judgeModelId, setJudgeModelId] = useState('')
  const [judgeAvailable, setJudgeAvailable] = useState(true)
  const [judgeLatency, setJudgeLatency] = useState('')
  const [judgeBusy, setJudgeBusy] = useState(false)
  const [judgeMessage, setJudgeMessage] = useState('')
  const [judgeError, setJudgeError] = useState('')
  const [planModal, setPlanModal] = useState(null)
  const [selectedPlan, setSelectedPlan] = useState('')
  const [planConfirmed, setPlanConfirmed] = useState(false)
  const fileInput = useRef(null)
  const messageEnd = useRef(null)
  const promptInput = useRef(null)

  function openPlanModal(reason) {
    setPlanModal(reason)
    setSelectedPlan('')
    setPlanConfirmed(false)
  }

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error('Backend is unavailable')
        return response.json()
      })
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline' }))
  }, [])

  useEffect(() => {
    messageEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, busy])

  async function sendMessage(text = prompt, attachedFiles = files) {
    const trimmed = text.trim()
    if ((!trimmed && attachedFiles.length === 0) || busy) return

    const userMessage = {
      role: 'user',
      content: trimmed,
      files: attachedFiles.map((file) => file.name),
    }
    setMessages((current) => [...current, userMessage])
    setPrompt('')
    setFiles([])
    setError('')
    setBusy(true)

    try {
      const attachments = await Promise.all(attachedFiles.map(fileToAttachment))
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          prompt: trimmed,
          mode,
          attachments,
          allow_paid: false,
        }),
      })
      const responseText = await response.text()
      let result
      try {
        result = responseText ? JSON.parse(responseText) : {}
      } catch {
        result = {
          detail: `Backend returned HTTP ${response.status}: ${responseText || response.statusText}`,
        }
      }

      if (response.status === 402 && result.paywall_required) {
        setMessages((current) => [...current, {
          role: 'assistant',
          content: 'No free model meets every constraint for this request. A paid model can handle it within your cost limit.',
          explanation: result,
          paywall: true,
        }])
        openPlanModal('required')
        return
      }

      if (!response.ok) {
        const detail = result.detail
        setMessages((current) => [...current, {
          role: 'assistant',
          content: typeof detail === 'string'
            ? detail
            : detail?.message || 'No model could complete this request while satisfying the constraints.',
          explanation: detail?.explanation || result,
          attempts: detail?.attempts || result.attempts,
          failed: true,
        }])
        return
      }

      setMessages((current) => [...current, {
        role: 'assistant',
        content: result.answer,
        images: result.images || [],
        explanation: result,
      }])
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the ModelRouter backend.')
    } finally {
      setBusy(false)
    }
  }

  async function refreshJudgeModels() {
    try {
      const response = await fetch(`${API_BASE}/api/judge/models`)
      const result = await response.json()
      if (!response.ok) {
        throw new Error(result.detail || 'Could not load judge controls.')
      }
      setJudgeModels(result.models)
      const selected = result.models.find((item) => item.id === judgeModelId)
        || result.models[0]
      if (selected) {
        setJudgeModelId(selected.id)
        setJudgeAvailable(selected.available)
        setJudgeLatency(selected.measured_latency_ms ?? '')
      }
      setJudgeError('')
    } catch (requestError) {
      setJudgeError(requestError.message || 'Could not load judge controls.')
    }
  }

  async function saveJudgeOverride() {
    if (!judgeModelId || judgeBusy) return
    setJudgeBusy(true)
    setJudgeMessage('')
    setJudgeError('')
    try {
      const response = await fetch(`${API_BASE}/api/judge/overrides`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: judgeModelId,
          available: judgeAvailable,
          measured_latency_ms: judgeLatency === '' ? null : Number(judgeLatency),
        }),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Could not save override.')
      await refreshJudgeModels()
      setJudgeMessage('Override saved. It will affect the next routing request.')
    } catch (requestError) {
      setJudgeError(requestError.message || 'Could not save override.')
    } finally {
      setJudgeBusy(false)
    }
  }

  async function resetJudgeOverride() {
    if (!judgeModelId || judgeBusy) return
    setJudgeBusy(true)
    setJudgeMessage('')
    setJudgeError('')
    try {
      const response = await fetch(`${API_BASE}/api/judge/overrides/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: judgeModelId }),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Could not reset override.')
      await refreshJudgeModels()
      setJudgeMessage('Override reset to catalogue defaults.')
    } catch (requestError) {
      setJudgeError(requestError.message || 'Could not reset override.')
    } finally {
      setJudgeBusy(false)
    }
  }

  async function handleFiles(event) {
    const selected = Array.from(event.target.files || [])
    setFiles((current) => [...current, ...selected].slice(0, 10))
    event.target.value = ''
  }

  function handleSubmit(event) {
    event.preventDefault()
    sendMessage()
  }

  function confirmPlan() {
    if (!selectedPlan) return
    setPlanConfirmed(true)
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="ModelRouter home">
          <span className="brand-mark">M</span>
          <span>model<span className="brand-light">router</span></span>
        </a>
        <div className="topbar-right">
          <span className="routing-label"><span className="status-dot" /> AUTO ROUTING</span>
          <span className={`backend-status ${health?.status === 'ok' ? 'online' : ''}`}>
            {health?.status === 'ok'
              ? `${health.model_count.toLocaleString()} models`
              : health?.status === 'offline' ? 'Backend offline' : 'Connecting'}
          </span>
          <button
            className="new-chat-button judge-toggle"
            type="button"
            aria-expanded={judgeOpen}
            onClick={() => {
              const opening = !judgeOpen
              setJudgeOpen(opening)
              if (opening) refreshJudgeModels()
            }}
          >
            Judge controls
          </button>
          <button
            className="new-chat-button"
            type="button"
            disabled={busy}
            onClick={() => {
              setConversationId(createConversationId())
              setMessages([])
              setPrompt('')
              setFiles([])
              setError('')
            }}
          >
            + New chat
          </button>
        </div>
      </header>

      {judgeOpen && (
        <section className="judge-panel" aria-label="Judge model controls">
          <div className="judge-panel-heading">
            <div>
              <h2>Judge controls</h2>
              <p>Simulate model availability and latency for the next routing decision.</p>
            </div>
            <span>Local machine only</span>
          </div>
          <div className="judge-controls">
            <label>
              <span>Model</span>
              <select
                value={judgeModelId}
                onChange={(event) => {
                  const selected = judgeModels.find((item) => item.id === event.target.value)
                  setJudgeModelId(event.target.value)
                  setJudgeAvailable(selected?.available ?? true)
                  setJudgeLatency(selected?.measured_latency_ms ?? '')
                  setJudgeMessage('')
                }}
              >
                {judgeModels.map((item) => (
                  <option value={item.id} key={item.id}>
                    {item.name}{item.overridden ? ' · overridden' : ''}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Simulated latency (ms)</span>
              <input
                type="number"
                min="0"
                max="120000"
                step="100"
                value={judgeLatency}
                onChange={(event) => setJudgeLatency(event.target.value)}
                placeholder="Unknown"
              />
            </label>
            <label className="judge-availability">
              <input
                type="checkbox"
                checked={judgeAvailable}
                onChange={(event) => setJudgeAvailable(event.target.checked)}
              />
              <span>Model available</span>
            </label>
            <div className="judge-actions">
              <button type="button" onClick={saveJudgeOverride} disabled={judgeBusy || !judgeModelId}>
                Save override
              </button>
              <button
                className="judge-reset"
                type="button"
                onClick={resetJudgeOverride}
                disabled={judgeBusy || !judgeModels.find((item) => item.id === judgeModelId)?.overridden}
              >
                Reset to catalogue
              </button>
            </div>
          </div>
          {judgeError && <p className="judge-status judge-error">{judgeError}</p>}
          {judgeMessage && <p className="judge-status">{judgeMessage}</p>}
        </section>
      )}

      <section className={`conversation ${messages.length ? 'has-messages' : ''}`}>
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="eyebrow"><span className="eyebrow-line" /> ONE REQUEST. THE RIGHT MODEL. <span className="eyebrow-line" /></div>
            <h1>What can I help<br />you <span>figure out?</span></h1>
            <p className="welcome-copy">Ask anything. ModelRouter estimates the task, checks every hard constraint, then picks and calls the best available model.</p>
            <div className="suggestions">
              {suggestions.map((item) => (
                <button
                  className="suggestion"
                  key={item.title}
                  onClick={() => {
                    setPrompt(item.prompt)
                    promptInput.current?.focus()
                  }}
                  type="button"
                >
                  <span className="suggestion-icon">{item.icon}</span>
                  <span>{item.title}</span>
                  <span className="suggestion-arrow">↗</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((message, index) => (
              <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                <div className="avatar">{message.role === 'user' ? 'Y' : 'M'}</div>
                <div className="message-body">
                  <div className="message-author">{message.role === 'user' ? 'You' : 'ModelRouter'}</div>
                  {message.content && (
                    <div className={`message-content ${message.failed ? 'message-error' : ''} ${message.role === 'assistant' ? 'markdown-content' : ''}`}>
                      {message.role === 'assistant' ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      ) : message.content}
                    </div>
                  )}
                  {message.images?.length > 0 && (
                    <div className="generated-images">
                      {message.images.map((image, imageIndex) => (
                        <img
                          key={`${index}-${imageIndex}`}
                          src={image}
                          alt={`Generated image ${imageIndex + 1}`}
                          loading="lazy"
                        />
                      ))}
                    </div>
                  )}
                  {message.files?.length > 0 && (
                    <div className="message-files">
                      {message.files.map((name) => <span className="file-pill" key={name}>▧ {name}</span>)}
                    </div>
                  )}
                  {message.explanation?.selected_model && (
                    <div className="answer-meta">
                      <span className="selected-model"><span className="meta-dot" />{message.explanation.selected_model}</span>
                      <span>{message.explanation.latency_ms} ms</span>
                      {message.explanation.actual_cost_usd != null && <span>${message.explanation.actual_cost_usd.toFixed(6)}</span>}
                    </div>
                  )}
                  {message.paywall && (
                    <div className="paywall-card">
                      <div className="paywall-title">Paid model needed</div>
                      <p>This request needs a paid model. Review the demo plans; paid model requests are disabled while billing is not connected.</p>
                      <button
                        className="paid-button"
                        type="button"
                        onClick={() => openPlanModal('required')}
                      >
                        View plans <span>→</span>
                      </button>
                    </div>
                  )}
                  {message.explanation && (
                    <details className="explanation">
                      <summary>Routing explanation</summary>
                      <Explanation data={message.explanation} />
                    </details>
                  )}
                </div>
              </article>
            ))}
            {busy && (
              <div className="thinking-row">
                <div className="avatar assistant-avatar">M</div>
                <div><span className="typing-dots"><i /><i /><i /></span><span className="thinking-label">Checking constraints and routing</span></div>
              </div>
            )}
            <div ref={messageEnd} />
          </div>
        )}
      </section>

      <div className="composer-wrap">
        {error && <div className="connection-error">{error}</div>}
        <form className="composer" onSubmit={handleSubmit}>
          {files.length > 0 && (
            <div className="attachment-list">
              {files.map((file, index) => (
                <span className="attachment-chip" key={`${file.name}-${index}`}>
                  <span>▧</span>{file.name}
                  <button type="button" aria-label={`Remove ${file.name}`} onClick={() => setFiles((current) => current.filter((_, i) => i !== index))}>×</button>
                </span>
              ))}
            </div>
          )}
          <textarea
            ref={promptInput}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                sendMessage()
              }
            }}
            placeholder="Message ModelRouter..."
            rows="2"
            aria-label="Message ModelRouter"
          />
          <div className="composer-controls">
            <div className="composer-left">
              <button className="icon-button attach-button" onClick={() => fileInput.current?.click()} type="button" aria-label="Attach a file" title="Attach a file">＋</button>
              <input ref={fileInput} type="file" multiple hidden onChange={handleFiles} />
              <span className="control-divider" />
              <label className="mode-control">
                <span aria-hidden="true">{mode === 'fast' ? '↯' : '✦'}</span>
                <select
                  aria-label="Routing preference"
                  value={mode}
                  onChange={(event) => setMode(event.target.value)}
                  title="Choose the model ranking preference"
                >
                  <option value="fast">Fast</option>
                  <option value="thinking">Thinking</option>
                </select>
              </label>
            </div>
            <div className="composer-right">
              <label className="paid-toggle" title="View demo plans; paid model requests are currently disabled">
                <input type="checkbox" checked={false} onChange={() => openPlanModal('manual')} />
                <span>Paid</span>
              </label>
              <button className="send-button" type="submit" disabled={busy || (!prompt.trim() && files.length === 0)} aria-label="Send message">
                {busy ? <span className="send-spinner" /> : <span>↑</span>}
              </button>
            </div>
          </div>
        </form>
        <div className="disclaimer">ModelRouter can make mistakes. Every route is filtered against your hard constraints.</div>
      </div>
      {planModal && (
        <div
          className="plan-modal-backdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget) setPlanModal(null)
          }}
        >
          <section
            className="plan-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="plan-modal-title"
          >
            <button
              className="plan-modal-close"
              type="button"
              aria-label="Close plans"
              onClick={() => setPlanModal(null)}
            >
              ×
            </button>
            {planConfirmed ? (
              <div className="plan-confirmation">
                <span className="plan-confirmation-icon">✓</span>
                <p className="plan-eyebrow">DEMO ONLY</p>
                <h2 id="plan-modal-title">Plan selected</h2>
                <p>
                  {demoPlans.find((plan) => plan.id === selectedPlan)?.name} is
                  selected for this demo. No payment was taken. Billing is not
                  connected, so paid model requests remain disabled.
                </p>
                <button
                  className="plan-primary-button"
                  type="button"
                  onClick={() => setPlanModal(null)}
                >
                  Done
                </button>
              </div>
            ) : (
              <>
                <p className="plan-eyebrow">MODELROUTER PLANS · DEMO</p>
                <h2 id="plan-modal-title">
                  {planModal === 'required' ? 'A paid model is needed' : 'Choose a plan'}
                </h2>
                <p className="plan-modal-copy">
                  Select a plan to preview the flow. This is a UI demo only;
                  payments and paid model access are not connected.
                </p>
                <div className="plan-options">
                  {demoPlans.map((plan) => (
                    <label
                      className={`plan-option ${selectedPlan === plan.id ? 'selected' : ''}`}
                      key={plan.id}
                    >
                      <input
                        type="radio"
                        name="demo-plan"
                        value={plan.id}
                        checked={selectedPlan === plan.id}
                        onChange={() => setSelectedPlan(plan.id)}
                      />
                      <span className="plan-option-main">
                        <span className="plan-option-name">
                          {plan.name}
                          {plan.popular && <span className="plan-popular">POPULAR</span>}
                        </span>
                        <span className="plan-option-description">{plan.description}</span>
                      </span>
                      <span className="plan-option-price">{plan.price}</span>
                    </label>
                  ))}
                </div>
                <div className="plan-modal-footer">
                  <span>No charge · demo selection only</span>
                  <button
                    className="plan-primary-button"
                    type="button"
                    disabled={!selectedPlan}
                    onClick={confirmPlan}
                  >
                    Select plan
                  </button>
                </div>
              </>
            )}
          </section>
        </div>
      )}
    </main>
  )
}

function Explanation({ data }) {
  const analysis = data.analysis
  const constraints = data.constraints || {}
  const rejected = data.rejected || []
  return (
    <div className="explanation-content">
      {analysis && (
        <div className="analysis-summary">
          <span>{analysis.complexity} complexity</span>
          <span>{analysis.complexity_confidence} confidence</span>
          <span>~{analysis.estimated_input_tokens.toLocaleString()} input tokens</span>
          <span>task: {analysis.task}</span>
          {analysis.automatic_min_capability != null && (
            <span>complexity floor: {analysis.automatic_min_capability} capability</span>
          )}
          {analysis.complexity_signals?.length > 0 && (
            <p>Signals: {analysis.complexity_signals.join(', ')}</p>
          )}
          <p>{analysis.estimation_note}</p>
        </div>
      )}
      {data.stage_counts && (
        <p className="filter-counts">
          Checked {data.stage_counts.catalogue} models · {data.stage_counts.eligible} eligible
        </p>
      )}
      {data.ranked_models.slice(0, 5).map((model, index) => (
        <div className="rejection-row" key={model.id}>
          <span>{index + 1}. {model.name}{model.id === data.selected_model ? ' · selected' : ''}</span>
          <span>
            {model.measured_latency_ms != null ? `${model.measured_latency_ms} ms` : 'latency unknown'}
            {' · '}
            {model.estimated_cost_usd != null ? `$${model.estimated_cost_usd.toFixed(6)} est.` : 'cost unknown'}
          </span>
        </div>
      ))}
      {data.ranked_models.length > 5 && (
        <p>Showing 5 of {data.ranked_models.length} ranked models.</p>
      )}
      {constraints && (
        <div className="constraint-tags">
          <span>{constraints.mode || 'fast'} preference</span>
          <span>input: {(constraints.input_modalities || ['text']).join(' + ')}</span>
          <span>output: {constraints.output_modality || 'text'}</span>
          <span>{constraints.max_latency_ms?.toLocaleString()} ms budget</span>
          <span>${constraints.max_cost_usd?.toFixed(4)} max cost</span>
          <span>{constraints.required_context_tokens?.toLocaleString()} context tokens</span>
          {constraints.min_capability != null && <span>capability ≥ {constraints.min_capability}</span>}
          <span>{constraints.free_only ? 'free only' : 'paid allowed'}</span>
        </div>
      )}
      {data.paid_alternatives?.length > 0 && (
        <div className="rejection-list">
          <strong>Paid alternatives that pass the other constraints</strong>
          {data.paid_alternatives.map((model) => (
            <div className="rejection-row" key={model.id}>
              <span>{model.name}</span><span>${model.estimated_cost_usd?.toFixed(6)} est.</span>
            </div>
          ))}
        </div>
      )}
      {/* {attempts?.length > 0 && (
        <div className="rejection-list">
          <strong>Inference attempts</strong>
          {attempts.map((attempt, index) => (
            <div className="rejection-row" key={`${attempt.model}-${index}`}>
              <span>{attempt.model}</span><span>{attempt.error || `${attempt.latency_ms} ms`}</span>
            </div>
          ))}
        </div>
      )} */}
      {rejected.length > 0 && (
        <div className="rejection-list">
          <strong>Rejected models ({rejected.length})</strong>
          {rejected.slice(0, 5).map((item) => (
            <div className="rejection-row" key={item.id}>
              <span>{item.name || item.id}</span>
              <span>{item.reasons.map((reason) => reason.reason).join(' · ')}</span>
            </div>
          ))}
          {rejected.length > 5 && <p>Showing 5 of {rejected.length} rejected models.</p>}
        </div>
      )}
    </div>
  )
}

export default App
