async function loadStatus() {
  try {
    const s = await fetch("/api/status").then(r => r.json());
    const el = document.getElementById("statusChips");
    if (!el) return;
    el.innerHTML = [
      chip(s.faceDetector, "YuNet 人臉偵測"),
      chip(s.faceRecognizer, "SFace 身份特徵"),
      chip(s.yolo, s.yolo ? "YOLO ONNX 已載入" : "YOLO 待匯入"),
      `<span class="chip">${s.identities} 人已建檔</span>`
    ].join("");
  } catch { /* ignore */ }
}
function chip(ok, label) {
  return `<span class="chip ${ok ? "ok" : "warn"}">${label}</span>`;
}
loadStatus();

function toast(msg, isErr) {
  const n = document.createElement("div");
  n.textContent = msg;
  n.style.cssText = `position:fixed;right:18px;bottom:18px;z-index:40;padding:10px 14px;border-radius:10px;background:${isErr ? "#4a1f1c" : "#16332e"};border:1px solid ${isErr ? "#e85d4c" : "#3ecfbf"}`;
  document.body.appendChild(n);
  setTimeout(() => n.remove(), 3200);
}

async function postForm(url, form) {
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res.text();
}

function inferFlags(form) {
  ["faces", "plates", "identify", "ocr", "yolo"].forEach(k => {
    const box = document.getElementById("opt-" + k);
    if (box) form.set(k, box.checked ? "1" : "0");
  });
}
