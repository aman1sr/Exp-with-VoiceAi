import { startCapture } from "./capture.js?v=p2-2";
import { Player } from "./playback.js?v=p2-2";

const $ = (id) => document.getElementById(id);

let ws = null;
let player = null;
let capture = null;
let talking = false;
let speechEndAt = 0;
let sampleRate = 0;
let firstChunk = true;
let turn = 0;
let startupComplete = false;

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
        if (talking && socket.readyState === WebSocket.OPEN) socket.send(frame);
      });
      if (ws !== socket || socket.readyState !== WebSocket.OPEN) {
        activeCapture.stop();
        return;
      }
      capture = activeCapture;
      $("talk").disabled = false;
      $("status").textContent = "ready";
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
    if (ws === socket && $("talk").disabled) {
      showStartError("WebSocket connection failed. Check that the P2 server is running, then retry.");
    }
  };

  socket.onclose = () => {
    if (ws !== socket) return;
    talking = false;
    $("talk").disabled = true;
    $("start").disabled = false;
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
    if (msg.type === "audio_start") {
      sampleRate = msg.sample_rate;
      firstChunk = true;
      player.reset();
    }
    else if (msg.type === "turn_done") addTurn(msg);
    else if (msg.type === "error") $("status").textContent = msg.message;
  };
}

function showStartError(message) {
  $("status").textContent = message;
  $("start").disabled = false;
  $("talk").disabled = true;
}

function addTurn({ transcript, reply, timings: t }) {
  const v2v = $("v2v").textContent;
  const timing = (name) => Number.isFinite(t?.[name]) ? t[name] : "—";
  const bar = (name, ms) => `<span class="${name}" style="width:${Number.isFinite(ms) ? ms / 10 : 0}px"></span>`;
  const expectedTimings = ["stt", "llm_first_token", "llm_first_sentence", "tts_first_byte", "first_audio", "total"];
  const missingTimings = expectedTimings.filter((name) => !Number.isFinite(t?.[name]));
  if (missingTimings.length) {
    console.warn("P2 timing fields missing from turn_done:", missingTimings, t);
  }
  $("turns").insertAdjacentHTML("afterbegin", `
    <tr>
      <td>${++turn}</td>
      <td>${transcript}</td>
      <td>${reply}</td>
      <td>${timing("stt")}</td><td>${timing("llm_first_token")}</td><td>${timing("llm_first_sentence")}</td><td>${timing("tts_first_byte")}</td>
      <td>${timing("first_audio")}</td><td>${timing("total")}</td><td>${v2v}</td>
      <td><div class="bar">${bar("stt", t?.stt)}${bar("ttft", t?.llm_first_token)}${bar("llm", t?.llm_first_sentence)}${bar("tts", t?.tts_first_byte)}</div></td>
    </tr>`);
  $("status").textContent = "connected";
}

$("start").onclick = () => {
  $("start").disabled = true;
  start();
};

const talk = $("talk");
talk.onpointerdown = () => {
  if (ws?.readyState !== WebSocket.OPEN || talk.disabled) return;
  talking = true;
  $("status").textContent = "listening...";
  ws.send(JSON.stringify({ type: "start" }));
};
talk.onpointerup = () => {
  if (!talking || ws?.readyState !== WebSocket.OPEN) return;
  talking = false;
  speechEndAt = performance.now();
  $("status").textContent = "thinking...";
  ws.send(JSON.stringify({ type: "end_of_speech" }));
};
talk.onpointercancel = talk.onpointerup;
