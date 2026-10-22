import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
// usage: node render-turn.mjs <turn.html path> <frames dir> <nframes> <maxDeg>
const [,, page, outDir, nStr = '120', degStr = '18'] = process.argv;
const N = +nStr, A = +degStr;
const port = 9400 + Math.floor(Math.random() * 400);
const profile = mkdtempSync(join(tmpdir(), 'cdp-profile-'));
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--allow-file-access-from-files', `--remote-debugging-port=${port}`, '--window-size=600,620', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
async function getWs() { for (let i = 0; i < 60; i++) { try { const l = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); const p = l.find(t => t.type === 'page'); if (p) return p.webSocketDebuggerUrl; } catch {} await new Promise(r => setTimeout(r, 200)); } throw new Error('no chrome'); }
const ws = new WebSocket(await getWs()); await new Promise(r => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const evalJs = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails)); return r.result?.result?.value; };
await send('Page.enable');
await send('Page.navigate', { url: `file://${page}` });
await new Promise(r => setTimeout(r, 800));
await evalJs('window.renderReady');
const t0 = Date.now();
for (let f = 0; f < N; f++) {
  const t = N === 1 ? 0 : f / (N - 1);                 // 0..1
  const theta = -A + 2 * A * t;                         // -A .. +A, even speed
  const url = await evalJs(`window.render(${theta})`);
  writeFileSync(`${outDir}/f${String(f).padStart(3, '0')}.png`, Buffer.from(url.split(',')[1], 'base64'));
  if (f % 20 === 0) console.log(`frame ${f}/${N} theta=${theta.toFixed(2)}`);
}
console.log(`rendered ${N} frames in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
ws.close(); chrome.kill();
