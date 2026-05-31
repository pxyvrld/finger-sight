import { useRef, useState, useEffect } from 'react'
import { useHandDetector } from './hooks/useHandDetector'
import './App.css'

const KNOWN_LETTERS = ['A','B','C','E','I','L','M','N','O','P','R','S','T','U','W','Y']
const CALIB_DURATION = 2500
const CALIB_INTERVAL = 100

function App() {
  const videoRef = useRef(null)
  const [history, setHistory] = useState([])
  const [running, setRunning] = useState(false)
  const [learningMode, setLearningMode] = useState(false)
  const [practised, setPractised] = useState(new Set())
  const [calibState, setCalibState] = useState('idle') // idle | prepare | running | ok | fail
  const [calibProgress, setCalibProgress] = useState(0)
  const [calibCountdown, setCalibCountdown] = useState(3)
  const calibIntervalRef = useRef(null)

  const { handDetected, letter, confidence, ready, startDetection, stopDetection } = useHandDetector(videoRef)
  const handDetectedRef = useRef(handDetected)
  useEffect(() => { handDetectedRef.current = handDetected }, [handDetected])

  const timerRef = useRef(null)
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    if (!letter) return
    timerRef.current = setTimeout(() => {
      setHistory(h => [...h.slice(-29), letter])
      if (learningMode) setPractised(p => new Set([...p, letter]))
    }, 1000)
    return () => clearTimeout(timerRef.current)
  }, [letter, learningMode])

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    videoRef.current.srcObject = stream
    videoRef.current.play()
    startDetection()
    setRunning(true)
  }

  const stopCamera = () => {
    videoRef.current?.srcObject?.getTracks().forEach(t => t.stop())
    stopDetection()
    setRunning(false)
    if (calibIntervalRef.current) clearInterval(calibIntervalRef.current)
    setCalibState('idle')
  }

  const calibrate = () => {
    if (!running) {
      setCalibState('ok')
      setTimeout(() => setCalibState('idle'), 1500)
      return
    }

    setCalibState('prepare')
    setCalibCountdown(3)

    let tick = 3
    const countdownInterval = setInterval(() => {
      tick--
      setCalibCountdown(tick)
      if (tick <= 0) {
        clearInterval(countdownInterval)
        startCalibMeasure()
      }
    }, 1000)
  }

  const startCalibMeasure = () => {
    setCalibState('running')
    setCalibProgress(0)

    let detectedCount = 0
    let step = 0
    const steps = CALIB_DURATION / CALIB_INTERVAL

    calibIntervalRef.current = setInterval(() => {
      if (handDetectedRef.current) detectedCount++
      step++
      setCalibProgress(Math.round((step / steps) * 100))

      if (step >= steps) {
        clearInterval(calibIntervalRef.current)
        const ratio = detectedCount / steps
        setCalibState(ratio > 0.5 ? 'ok' : 'fail')
        setTimeout(() => setCalibState('idle'), 2500)
      }
    }, CALIB_INTERVAL)
  }

  const display = !ready ? '…' : !handDetected ? '—' : (letter ?? '?')
  const isLetter = ready && handDetected && letter
  const pct = Math.round(confidence * 100)
  const barColor = confidence > 0.8 ? '#4ade80' : confidence > 0.6 ? '#facc15' : '#f87171'
  const barLabel = confidence > 0.8 ? 'wysoka' : confidence > 0.6 ? 'średnia' : 'niska'

  const calibLabel = {
    idle:    'Kalibracja',
    prepare: `Przygotuj się… ${calibCountdown}`,
    running: `Sprawdzanie… ${calibProgress}%`,
    ok:      '✓ Kalibracja OK',
    fail:    '✗ Popraw oświetlenie',
  }[calibState]

  const calibClass = {
    idle:    'btn--ghost',
    prepare: 'btn--ghost',
    running: 'btn--ghost',
    ok:      'btn--calib-ok',
    fail:    'btn--calib-fail',
  }[calibState]

  return (
    <div className="app">

      <header className="header">
        <h1 className="header__title">FingerSight</h1>
        <p className="header__subtitle">Polski Alfabet Palcowy</p>
      </header>

      <main className="layout">

        {/* Camera */}
        <section className="camera" aria-label="Podgląd kamery">
          <video ref={videoRef} className="camera__video" muted aria-label="Obraz z kamery" />
          {!running && <div className="camera__placeholder" aria-hidden="true">🖐</div>}
          {running && (
            <div
              className={`camera__badge ${handDetected ? 'camera__badge--detected' : 'camera__badge--none'}`}
              role="status"
              aria-live="polite"
              aria-atomic="true"
            >
              {handDetected ? 'Dłoń wykryta' : 'Brak dłoni'}
            </div>
          )}
          {calibState === 'prepare' && (
            <div className="calib-overlay" aria-live="assertive">
              <p className="calib-overlay__msg">Ustaw dłoń przed kamerą</p>
              <span className="calib-overlay__count">{calibCountdown}</span>
            </div>
          )}
          {calibState === 'running' && (
            <div className="calib-bar" aria-hidden="true">
              <div className="calib-bar__fill" style={{ width: `${calibProgress}%` }} />
            </div>
          )}
          {(calibState === 'ok' || calibState === 'fail') && (
            <div className={`calib-result ${calibState === 'ok' ? 'calib-result--ok' : 'calib-result--fail'}`} aria-live="polite">
              {calibState === 'ok' ? '✓ Kalibracja OK' : '✗ Popraw oświetlenie'}
            </div>
          )}
        </section>

        {/* Panel */}
        <div className="panel">

          {/* Letter + confidence */}
          <div className="card letter-card">
            <span
              className={`letter ${isLetter ? 'letter--active' : ''}`}
              role="status"
              aria-live="polite"
              aria-atomic="true"
              aria-label={isLetter ? `Rozpoznana litera: ${letter}` : 'Brak rozpoznanej litery'}
            >
              {display}
            </span>

            <div className="confidence" role="group" aria-label="Wskaźnik pewności">
              <div className="confidence__header">
                <span className="confidence__label" id="conf-label">Pewność</span>
                <span
                  className="confidence__value"
                  style={{ color: isLetter ? barColor : '#2a2a2a' }}
                  aria-hidden="true"
                >
                  {isLetter ? `${pct}%` : '—'}
                </span>
              </div>
              <div
                className="confidence__track"
                role="progressbar"
                aria-labelledby="conf-label"
                aria-valuenow={pct}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuetext={isLetter ? `${pct}%, pewność ${barLabel}` : 'brak danych'}
              >
                <div className="confidence__bar" style={{ width: `${pct}%`, backgroundColor: barColor }} />
              </div>
            </div>
          </div>

          {/* History */}
          <div className="card" role="region" aria-label="Historia rozpoznanych liter">
            <p className="history__label" id="history-label">Historia</p>
            <p className="history__text" aria-labelledby="history-label" aria-live="polite" aria-relevant="additions">
              {history.join('') || '—'}
            </p>
          </div>

          {/* Controls */}
          <div className="controls" role="group" aria-label="Sterowanie">
            {!running ? (
              <button className="btn btn--primary" onClick={startCamera}>Uruchom kamerę</button>
            ) : (
              <button className="btn btn--secondary" onClick={stopCamera}>Zatrzymaj</button>
            )}
            <button
              className={`btn ${calibClass}`}
              onClick={calibrate}
              disabled={calibState === 'running' || calibState === 'prepare'}
              aria-label="Kalibracja — sprawdź jakość wykrywania dłoni"
            >
              {calibLabel}
            </button>
            <button
              className={`btn ${learningMode ? 'btn--active' : 'btn--ghost'}`}
              onClick={() => setLearningMode(m => !m)}
              aria-pressed={learningMode}
            >
              Tryb nauki
            </button>
            <button
              className="btn btn--ghost"
              onClick={() => setHistory([])}
              aria-label="Wyczyść historię rozpoznanych liter"
            >
              Wyczyść historię
            </button>
          </div>

        </div>
      </main>

      {/* Learning mode panel */}
      {learningMode && (
        <section className="learn-panel" aria-label="Tryb nauki — siatka liter">
          <div className="learn-panel__header">
            <span className="learn-panel__title">Ćwicz litery</span>
            <span className="learn-panel__progress" aria-live="polite">
              {practised.size} / {KNOWN_LETTERS.length} zaliczonych
            </span>
          </div>
          <div className="learn-grid" role="list">
            {KNOWN_LETTERS.map(l => (
              <div
                key={l}
                role="listitem"
                className={`learn-cell ${letter === l ? 'learn-cell--active' : practised.has(l) ? 'learn-cell--done' : ''}`}
                aria-label={`Litera ${l}${practised.has(l) ? ', zaliczona' : ''}${letter === l ? ', aktualnie wykrywana' : ''}`}
              >
                {l}
              </div>
            ))}
          </div>
        </section>
      )}

    </div>
  )
}

export default App
