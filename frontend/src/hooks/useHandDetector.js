import { useEffect, useRef, useState } from 'react'
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { classifyHand } from '../classifier'

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

export function useHandDetector(videoRef) {
  const [handDetected, setHandDetected] = useState(false)
  const [letter, setLetter] = useState(null)
  const [ready, setReady] = useState(false)
  const landmarkerRef = useRef(null)
  const rafRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function init() {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 1,
      })
      if (!cancelled) {
        landmarkerRef.current = landmarker
        setReady(true)
      }
    }

    init()

    return () => {
      cancelled = true
      landmarkerRef.current?.close()
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  function startDetection() {
    function detect() {
      const video = videoRef.current
      const landmarker = landmarkerRef.current
      if (video && landmarker && video.readyState >= 2) {
        const result = landmarker.detectForVideo(video, performance.now())
        const detected = result.landmarks.length > 0
        setHandDetected(detected)
        setLetter(detected ? classifyHand(result.landmarks[0]) : null)
      }
      rafRef.current = requestAnimationFrame(detect)
    }
    detect()
  }

  function stopDetection() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    setHandDetected(false)
    setLetter(null)
  }

  return { handDetected, letter, ready, startDetection, stopDetection }
}
