// Plays raw PCM16 audio chunks back to back, with no gaps, through Web Audio.

export class Player {
  constructor() {
    this.ctx = new AudioContext();
    this.nextTime = 0;
    this.pendingByte = null;
  }

  reset() {
    this.nextTime = 0;
    this.pendingByte = null;
  }

  enqueue(arrayBuffer, sampleRate) {
    let bytes = new Uint8Array(arrayBuffer);
    if (this.pendingByte !== null) {
      const combined = new Uint8Array(bytes.length + 1);
      combined[0] = this.pendingByte;
      combined.set(bytes, 1);
      bytes = combined;
      this.pendingByte = null;
    }

    if (bytes.length % 2) {
      this.pendingByte = bytes[bytes.length - 1];
      bytes = bytes.subarray(0, bytes.length - 1);
    }
    if (!bytes.length) return null;

    const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
    const buffer = this.ctx.createBuffer(1, int16.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < int16.length; i++) channel[i] = int16[i] / 32768;

    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);
    const startAt = Math.max(this.ctx.currentTime, this.nextTime);
    source.start(startAt);
    this.nextTime = startAt + buffer.duration;
    return performance.now() + (startAt - this.ctx.currentTime) * 1000;
  }
}
