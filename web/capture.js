// Opens the mic and delivers 20ms PCM16 frames (640 bytes) via onFrame.

export async function startCapture(onFrame) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true, // stops the agent's own voice being picked up later
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  let ctx;
  try {
    // ask the browser for 16kHz directly, it resamples the mic for us
    ctx = new AudioContext({ sampleRate: 16000 });
    await ctx.audioWorklet.addModule("worklet.js?v=p5-1");

    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "pcm-frame-processor");
    node.port.onmessage = (e) => onFrame(e.data);
    source.connect(node);

    return {
      stop() {
        node.disconnect();
        source.disconnect();
        stream.getTracks().forEach((track) => track.stop());
        ctx.close();
      },
    };
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    await ctx?.close();
    throw error;
  }
}
