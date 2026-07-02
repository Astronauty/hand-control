"""Live metrics dashboard for kinova_leap_pick_place.py.

Runs a pyqtgraph GUI in a SEPARATE PROCESS, fed by a multiprocessing.Queue, so the
MuJoCo passive viewer's GLFW event loop (main process) and the Qt event loop never
share a thread. Two GUI event loops in one thread — or Qt in a background thread —
is fragile; a separate process decouples them entirely and keeps the sim real-time.

The process is spawned (not forked) so the GUI interpreter starts clean and doesn't
inherit the parent's MuJoCo/GL state.

Message protocol (plain dicts put on the queue):
    {'type': 'dist',  't': float, 'd': {finger: signed_distance_m}}   # streaming, per step
    {'type': 'rrt',   'n_wp': int, 'plan_time': float, 'fallback': bool}
    {'type': 'ipopt', 'phase': str, 'status': str, 'iters': int|str,
                      'max_site_mm': float, 'max_pad_deg': float|None,
                      'min_slack_mm': float|None}
    {'type': 'active', 'label': str}     # active-object/phase label (optional)
    None                                 # sentinel: quit

Usage from the sim process:
    dash = Dashboard(FINGER_SET, horizon_s=10.0)
    dash.start()
    dash.push({'type': 'dist', 't': t, 'd': {'index': 0.03, 'thumb': 0.02}})
    ...
    dash.close()
"""
import multiprocessing as mp

# Distinct, high-contrast colour per finger (RGB 0-255). Matches the four LEAP fingers;
# unknown finger names fall back to grey.
FINGER_COLORS = {
    'index':  (60, 170, 255),
    'middle': (110, 220, 110),
    'ring':   (255, 200, 60),
    'thumb':  (255, 95, 95),
}


def _run(queue, fingers, horizon_s, dt_hint):
    """Dashboard process entry point. Imports Qt lazily so the parent never loads it."""
    from collections import deque
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore

    app = pg.mkQApp("hand-control dashboard")
    pg.setConfigOptions(antialias=True)

    win = QtWidgets.QWidget()
    win.setWindowTitle("hand-control — live dashboard")
    grid = QtWidgets.QGridLayout(win)

    # --- (1) scrolling fingertip→object distance plot ---
    plot = pg.PlotWidget(title="Fingertip → active object  (signed geom distance)")
    plot.addLegend(offset=(-10, 10))
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.setLabel('bottom', 'time', 's')
    plot.setLabel('left', 'distance', 'm')
    plot.addLine(y=0.0, pen=pg.mkPen((150, 150, 150), width=1,
                                     style=QtCore.Qt.DashLine))  # contact / penetration line

    maxlen = max(10, int(horizon_s / max(dt_hint, 1e-3)))
    tbuf = deque(maxlen=maxlen)
    curves, dbuf = {}, {}
    for f in fingers:
        col = FINGER_COLORS.get(f, (200, 200, 200))
        curves[f] = plot.plot(pen=pg.mkPen(col, width=2), name=f)
        dbuf[f] = deque(maxlen=maxlen)

    # --- (2) RRT metrics + (3) IPOPT metrics: monospace scrollback logs ---
    def _log_widget():
        w = QtWidgets.QPlainTextEdit()
        w.setReadOnly(True)
        w.setMaximumBlockCount(200)
        w.setStyleSheet("font-family: monospace; font-size: 11px;")
        return w

    rrt_log = _log_widget()
    ipopt_log = _log_widget()
    active_lbl = QtWidgets.QLabel("active: —")
    active_lbl.setStyleSheet("font-weight: bold;")

    grid.addWidget(active_lbl, 0, 0, 1, 2)
    grid.addWidget(plot, 1, 0, 1, 2)
    grid.addWidget(QtWidgets.QLabel("<b>RRT solutions</b>"), 2, 0)
    grid.addWidget(QtWidgets.QLabel("<b>IPOPT solutions</b>"), 2, 1)
    grid.addWidget(rrt_log, 3, 0)
    grid.addWidget(ipopt_log, 3, 1)
    grid.setRowStretch(1, 3)
    grid.setRowStretch(3, 1)
    win.resize(950, 680)
    win.show()

    def drain():
        got_dist = False
        # Bounded drain so a burst can't starve the event loop.
        for _ in range(20000):
            try:
                msg = queue.get_nowait()
            except Exception:
                break
            if msg is None:
                app.quit()
                return
            mt = msg.get('type')
            if mt == 'dist':
                tbuf.append(msg['t'])
                for f in fingers:
                    dbuf[f].append(msg['d'].get(f, float('nan')))
                got_dist = True
            elif mt == 'rrt':
                kind = "linear-fallback" if msg.get('fallback') else "RRT"
                rrt_log.appendPlainText(
                    f"{kind}: {msg.get('n_wp', '?')} wp  "
                    f"{msg.get('plan_time', float('nan'))*1e3:.0f} ms")
            elif mt == 'ipopt':
                pad = msg.get('max_pad_deg')
                slack = msg.get('min_slack_mm')
                ipopt_log.appendPlainText(
                    f"{msg.get('phase', '?'):<9} {msg.get('status', '?')[:22]:<22} "
                    f"it={msg.get('iters', '?')!s:>4}  "
                    f"site={msg.get('max_site_mm', float('nan')):.1f}mm  "
                    f"pad={'—' if pad is None else f'{pad:.1f}°'}  "
                    f"slack={'—' if slack is None else f'{slack:.1f}mm'}")
            elif mt == 'active':
                active_lbl.setText(f"active: {msg.get('label', '—')}")
        if got_dist:
            x = list(tbuf)
            for f in fingers:
                curves[f].setData(x, list(dbuf[f]))

    timer = QtCore.QTimer()
    timer.timeout.connect(drain)
    timer.start(33)  # ~30 Hz refresh
    app.exec_()


class Dashboard:
    """Handle to the dashboard process. push() is non-blocking and drops messages if
    the queue is full, so instrumenting the sim loop can never stall the sim."""

    def __init__(self, fingers, horizon_s=10.0, dt_hint=0.001, maxsize=20000):
        self._fingers = list(fingers)
        # 'spawn' → clean interpreter, no inherited MuJoCo/GL/threads from the parent.
        self._ctx = mp.get_context('spawn')
        self._q = self._ctx.Queue(maxsize=maxsize)
        self._p = self._ctx.Process(
            target=_run, args=(self._q, self._fingers, horizon_s, dt_hint), daemon=True)

    def start(self):
        self._p.start()

    def push(self, msg):
        try:
            self._q.put_nowait(msg)
        except Exception:
            pass  # queue full (GUI slow / closed) — drop, never block the sim

    def is_alive(self):
        return self._p.is_alive()

    def close(self):
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        self._p.join(timeout=1.5)
        if self._p.is_alive():
            self._p.terminate()
