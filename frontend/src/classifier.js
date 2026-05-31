// Geometry helpers
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

// MediaPipe landmark indices:
//   Thumb:  1(CMC) 2(MCP) 3(IP)  4(TIP)
//   Index:  5(MCP) 6(PIP) 7(DIP) 8(TIP)
//   Middle: 9(MCP) 10(PIP) 11(DIP) 12(TIP)
//   Ring:   13(MCP) 14(PIP) 15(DIP) 16(TIP)
//   Pinky:  17(MCP) 18(PIP) 19(DIP) 20(TIP)
//
// y increases downward — tip.y < pip.y means finger is extended upward.

export function classifyHand(lm) {
  const indexExt  = lm[8].y  < lm[6].y
  const middleExt = lm[12].y < lm[10].y
  const ringExt   = lm[16].y < lm[14].y
  const pinkyExt  = lm[20].y < lm[18].y

  // Thumb spread relative to palm width (index MCP → pinky MCP)
  const palmWidth = dist(lm[5], lm[17])
  const thumbOut  = dist(lm[4], lm[5]) > palmWidth * 0.6

  const noFingerExt = !indexExt && !middleExt && !ringExt && !pinkyExt

  if (noFingerExt) {
    // C: fingers curved forward (tip above MCP) but NOT fully extended
    // A: fingers curled tightly into fist (tip below MCP)
    const indexForward = lm[8].y < lm[5].y
    return indexForward ? 'C' : 'A'
  }

  // B: all 4 fingers up, thumb tucked across palm
  if (indexExt && middleExt && ringExt && pinkyExt && !thumbOut) return 'B'

  // L: index up + thumb out, other 3 bent
  if (indexExt && !middleExt && !ringExt && !pinkyExt && thumbOut) return 'L'

  // Y: pinky up + thumb out, other 3 bent (shaka)
  if (!indexExt && !middleExt && !ringExt && pinkyExt && thumbOut) return 'Y'

  return null
}
