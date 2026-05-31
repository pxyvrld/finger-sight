import { useRef, useState, useEffect } from 'react'
import { useHandDetector } from './hooks/useHandDetector'

function App() {
  const videoRef = useRef(null)
  const [history, setHistory] = useState([])
  const { handDetected, letter, confidence, ready, startDetection, stopDetection } = useHandDetector(videoRef)
  const timerRef = useRef(null)

  // Add letter to history after 1 second of stable gesture
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    if (!letter) return
    timerRef.current = setTimeout(() => {
      setHistory(h => [...h.slice(-19), letter])
    }, 1000)
    return () => clearTimeout(timerRef.current)
  }, [letter])

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    videoRef.current.srcObject = stream
    videoRef.current.play()
    startDetection()
  }

  const stopCamera = () => {
    videoRef.current?.srcObject?.getTracks().forEach(t => t.stop())
    stopDetection()
  }

  const display = !ready ? '⏳' : !handDetected ? '—' : (letter ?? '?')
  const isLetter = ready && handDetected && letter

  return (
    <div style={{minHeight:'100vh', background:'#f5f0eb', display:'flex', flexDirection:'column', alignItems:'center', padding:'24px', fontFamily:'sans-serif'}}>
      <h2 style={{color:'#5a4a3a'}}>🖐 FingerSight</h2>
      <div style={{display:'flex', gap:'20px'}}>
        <div style={{width:'400px', height:'300px', borderRadius:'16px', overflow:'hidden', background:'#d9cfc7', border:'2px solid #c4b5a5'}}>
          <video ref={videoRef} style={{width:'100%', height:'100%', objectFit:'cover'}} muted />
        </div>
        <div style={{display:'flex', flexDirection:'column', gap:'12px', width:'140px'}}>
          <div style={{background:'white', borderRadius:'16px', border:'2px solid #c4b5a5', height:'120px', display:'flex', alignItems:'center', justifyContent:'center', padding:'8px'}}>
            <span style={{
              fontSize: display.length === 1 ? '72px' : '28px',
              fontWeight: 'bold',
              color: isLetter ? '#4a7c59' : '#8a7a6a',
              textAlign: 'center',
              lineHeight: 1,
            }}>
              {display}
            </span>
          </div>
          <div style={{background:'white', borderRadius:'16px', border:'2px solid #c4b5a5', padding:'12px'}}>
            <p style={{margin:'0 0 6px 0', color:'#888', fontSize:'13px'}}>Pewność:</p>
            <div style={{background:'#e8e0d8', borderRadius:'8px', height:'10px', overflow:'hidden'}}>
              <div style={{width:`${Math.round(confidence * 100)}%`, height:'100%', background: confidence > 0.8 ? '#4a7c59' : confidence > 0.6 ? '#e0a020' : '#c0392b', transition:'width 0.2s'}} />
            </div>
            <p style={{margin:'4px 0 0 0', fontSize:'12px', color:'#888', textAlign:'right'}}>{Math.round(confidence * 100)}%</p>
          </div>
          <div style={{background:'white', borderRadius:'16px', border:'2px solid #c4b5a5', padding:'12px'}}>
            <p style={{margin:'0 0 6px 0', color:'#888', fontSize:'13px'}}>Historia:</p>
            <p style={{margin:0, fontSize:'14px', wordBreak:'break-all'}}>{history.join(' ') || '—'}</p>
          </div>
        </div>
      </div>
      <div style={{display:'flex', gap:'10px', marginTop:'20px'}}>
        <button onClick={startCamera} style={{padding:'10px 20px', borderRadius:'20px', border:'2px solid #c4b5a5', background:'white', cursor:'pointer'}}>Start</button>
        <button onClick={stopCamera} style={{padding:'10px 20px', borderRadius:'20px', border:'2px solid #c4b5a5', background:'white', cursor:'pointer'}}>Stop</button>
        <button onClick={() => setHistory([])} style={{padding:'10px 20px', borderRadius:'20px', border:'2px solid #c4b5a5', background:'white', cursor:'pointer'}}>Kalibracja</button>
        <button style={{padding:'10px 20px', borderRadius:'20px', border:'2px solid #c4b5a5', background:'white', cursor:'pointer'}}>Tryb Nauki</button>
      </div>
    </div>
  )
}

export default App
