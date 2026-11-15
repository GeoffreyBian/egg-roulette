// Squeeze-force trace for the egg in the gripper: one line per policy on screen, with the two numbers that
// make the task hard drawn as reference lines.
const T_MAX = 11;          // s: 330 control steps at 30 Hz
const F_MAX = 6;           // N

export class ForceChart {
  constructor(canvas, tooltip) {
    this.c = canvas; this.tip = tooltip;
    this.series = [];        // [{label, color, t:[], f:[], dashed}]
    this.cursor = null;      // t in s, during replay
    this.hover = null;
    canvas.addEventListener('pointermove', (e) => { this.hover = e.offsetX; this.draw(); });
    canvas.addEventListener('pointerleave', () => { this.hover = null; this.tip.hidden = true; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  css(name) { return getComputedStyle(this.c).getPropertyValue(name).trim(); }

  draw() {
    const c = this.c, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const L = 34, R = 10, T = 10, B = 22, pw = W - L - R, ph = H - T - B;
    const x = (t) => L + (t / T_MAX) * pw, y = (f) => T + ph - (Math.min(f, F_MAX) / F_MAX) * ph;
    const ink = this.css('--ink-2'), muted = this.css('--ink-3'), grid = this.css('--grid');
    g.font = '11px system-ui, sans-serif';
    // grid + axes
    g.strokeStyle = grid; g.lineWidth = 1; g.fillStyle = muted;
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (let f = 0; f <= F_MAX; f += 2) {
      g.beginPath(); g.moveTo(L, y(f) + 0.5); g.lineTo(W - R, y(f) + 0.5); g.stroke();
      g.fillText(`${f}`, L - 6, y(f));
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (let t = 0; t <= T_MAX; t += 2) g.fillText(`${t}s`, x(t), T + ph + 6);
    g.save(); g.translate(10, T + ph / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle';
    g.fillText('squeeze, N', 0, 0); g.restore();
    // the two thresholds
    const ref = (f, label, color) => {
      g.strokeStyle = color; g.setLineDash([4, 4]); g.lineWidth = 1.25;
      g.beginPath(); g.moveTo(L, y(f) + 0.5); g.lineTo(W - R, y(f) + 0.5); g.stroke(); g.setLineDash([]);
      g.fillStyle = color; g.textAlign = 'right'; g.textBaseline = 'bottom';
      g.fillText(label, W - R - 2, y(f) - 2);
    };
    ref(2.2, 'raw egg cracks above 2.2 N', this.css('--bad'));
    ref(3.5, 'stone slips below ~3.5 N', this.css('--warn'));
    // traces
    for (const s of this.series) {
      if (s.t.length < 2) continue;
      g.strokeStyle = s.color; g.lineWidth = 2; g.lineJoin = 'round';
      g.beginPath();
      s.t.forEach((t, i) => (i ? g.lineTo(x(t), y(s.f[i])) : g.moveTo(x(t), y(s.f[i]))));
      g.stroke();
      const i = s.t.length - 1;
      g.fillStyle = s.color; g.beginPath(); g.arc(x(s.t[i]), y(s.f[i]), 3, 0, 2 * Math.PI); g.fill();
    }
    // replay cursor / hover crosshair with tooltip
    const cx = this.hover ?? (this.cursor !== null ? x(this.cursor) : null);
    if (cx !== null && cx >= L && cx <= W - R) {
      g.strokeStyle = ink; g.lineWidth = 1; g.globalAlpha = 0.5;
      g.beginPath(); g.moveTo(Math.round(cx) + 0.5, T); g.lineTo(Math.round(cx) + 0.5, T + ph); g.stroke();
      g.globalAlpha = 1;
      if (this.hover !== null) {
        const t = ((cx - L) / pw) * T_MAX;
        const rows = this.series.filter((s) => s.t.length).map((s) => {
          let k = 0;
          while (k < s.t.length - 1 && s.t[k + 1] <= t) k++;
          return s.t[k] <= t + 0.05 ? `<span class="sw" style="background:${s.color}"></span>${s.label}: <b>${s.f[k].toFixed(2)} N</b>` : null;
        }).filter(Boolean);
        if (rows.length) {
          this.tip.innerHTML = `<div class="tt-t">${t.toFixed(2)} s</div>${rows.join('<br>')}`;
          this.tip.hidden = false;
          const left = cx + 12 + this.tip.offsetWidth > W ? cx - 12 - this.tip.offsetWidth : cx + 12;
          this.tip.style.left = `${left}px`; this.tip.style.top = `${T}px`;
        } else this.tip.hidden = true;
      }
    }
  }
}
