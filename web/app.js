import { startCapture } from "./capture.js?v=p8-1";
import { Player } from "./playback.js?v=p8-1";

const $ = (id) => document.getElementById(id);

let ws = null;
let player = null;
let capture = null;
let speechEndAt = 0;
let sampleRate = 0;
let firstChunk = true;
let turn = 0;
let startupComplete = false;
let playbackTimer = null;
let turnDone = false;

function armPlaybackDone(socket) {
  if (playbackTimer !== null) clearTimeout(playbackTimer);
  playbackTimer = setTimeout(() => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "playback_done" }));
      $("status").textContent = "listening";
    }
    playbackTimer = null;
  }, player.remainingMs());
}

async function start() {
  let socket;
  startupComplete = false;
  try {
    player = new Player();
    player.ctx.resume().catch((error) => console.error("Audio context resume failed", error));
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${location.host}/ws`);
  } catch (error) {
    showStartError(`Could not start: ${error.message}`);
    return;
  }

  ws = socket;
  socket.binaryType = "arraybuffer";
  $("status").textContent = "connecting...";

  socket.onopen = async () => {
    if (ws !== socket) return;
    $("status").textContent = "requesting microphone...";
    try {
      const activeCapture = await startCapture((frame) => {
        if (socket.readyState === WebSocket.OPEN) socket.send(frame);
      });
      if (ws !== socket || socket.readyState !== WebSocket.OPEN) {
        activeCapture.stop();
        return;
      }
      capture = activeCapture;
      $("status").textContent = "listening";
      startupComplete = true;
    } catch (error) {
      console.error("Microphone setup failed", error);
      const detail = error.name === "NotAllowedError"
        ? "Allow microphone access in the browser, then retry."
        : error.message;
      showStartError(`Microphone setup failed: ${detail}`);
      socket.close();
    }
  };

  socket.onerror = () => {
    if (ws === socket && !startupComplete) {
      showStartError("WebSocket connection failed. Check that the P7 server is running, then retry.");
    }
  };

  socket.onclose = () => {
    if (ws !== socket) return;
    $("start").disabled = false;
    if (playbackTimer !== null) clearTimeout(playbackTimer);
    playbackTimer = null;
    capture?.stop();
    capture = null;
    if (!startupComplete) {
      if ($("status").textContent === "connecting..." ||
          $("status").textContent === "requesting microphone...") {
        $("status").textContent = "Connection closed before startup completed. Retry.";
      }
    } else {
      $("status").textContent = "disconnected";
    }
  };

  socket.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      if (!player) return;
      const playAt = player.enqueue(e.data, sampleRate);
      if (firstChunk && playAt !== null) {
        $("v2v").textContent = Math.round(playAt - speechEndAt) + " ms";
        firstChunk = false;
      }
      return;
    }
    const msg = JSON.parse(e.data);
    if (msg.type === "speech_start") {
      $("status").textContent = "you are speaking...";
      $("live").textContent = "";
    } else if (msg.type === "speech_end") {
      speechEndAt = performance.now() - msg.turn_detection_ms;
      $("status").textContent = "thinking...";
    } else if (msg.type === "audio_start") {
      sampleRate = msg.sample_rate;
      firstChunk = true;
      turnDone = false;
      player.reset();
      $("status").textContent = "speaking...";
    } else if (msg.type === "pause_audio") {
      player.pause().catch((error) => console.error("Audio pause failed", error));
      if (playbackTimer !== null) clearTimeout(playbackTimer);
      playbackTimer = null;
      $("status").textContent = `paused while listening (${msg.after_ms} ms)`;
    } else if (msg.type === "resume_audio") {
      player.resume().catch((error) => console.error("Audio resume failed", error));
      if (turnDone) armPlaybackDone(socket);
      $("status").textContent = "speaking...";
    } else if (msg.type === "clear_audio") {
      const playedMs = player.clear();
      if (playbackTimer !== null) clearTimeout(playbackTimer);
      playbackTimer = null;
      socket.send(JSON.stringify({ type: "played", ms: playedMs }));
      $("status").textContent = `interrupted after ${Math.round(playedMs)} ms`;
    } else if (msg.type === "transcript") {
      $("live").textContent = msg.text;
      $("live").className = msg.final ? "final" : "interim";
    } else if (msg.type === "turn_done") {
      if (!msg.greeting && msg.transcript) addTurn(msg);
      turnDone = true;
      if (!player.paused) armPlaybackDone(socket);
    } else if (msg.type === "error") {
      $("status").textContent = msg.message;
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "playback_done" }));
    }
  };
}

function showStartError(message) {
  $("status").textContent = message;
  $("start").disabled = false;
}

function addTurn({ transcript, reply, written, provider, speculated, tool_ms, timings: t }) {
  if (provider) reply += " (" + provider + ")";
  if (written) reply += " (model wrote: " + written + ")";
  if (speculated) reply += " [LLM started early]";
  if (tool_ms) reply += " [tool call: " + tool_ms + " ms]";
  const v2v = $("v2v").textContent;
  const timing = (name) => Number.isFinite(t?.[name]) ? t[name] : "—";
  const bar = (name, ms) => `<span class="${name}" style="width:${Number.isFinite(ms) ? ms / 10 : 0}px"></span>`;
  const expectedTimings = ["turn_detection", "stt", "llm_first_token", "llm_first_sentence", "tts_first_byte", "first_audio", "total"];
  const missingTimings = expectedTimings.filter((name) => !Number.isFinite(t?.[name]));
  if (missingTimings.length) {
    console.warn("P7 timing fields missing from turn_done:", missingTimings, t);
  }
  $("turns").insertAdjacentHTML("afterbegin", `
    <tr>
      <td>${++turn}</td>
      <td>${transcript}</td>
      <td>${reply}</td>
      <td>${timing("turn_detection")}</td><td>${timing("stt")}</td><td>${timing("llm_first_token")}</td><td>${timing("llm_first_sentence")}</td><td>${timing("tts_first_byte")}</td>
      <td>${timing("first_audio")}</td><td>${timing("total")}</td><td>${v2v}</td>
      <td><div class="bar">${bar("vad", t?.turn_detection)}${bar("stt", t?.stt)}${bar("ttft", t?.llm_first_token)}${bar("llm", t?.llm_first_sentence)}${bar("tts", t?.tts_first_byte)}</div></td>
    </tr>`);
}

$("start").onclick = () => {
  $("start").disabled = true;
  start();
};
