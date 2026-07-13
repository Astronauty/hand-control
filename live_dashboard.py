"""Live metrics dashboard for kinova_leap_pick_place.py.

Runs a pyqtgraph GUI in a SEPARATE PROCESS, fed by a multiprocessing.Queue, so the
MuJoCo passive viewer's GLFW event loop (main process) and the Qt event loop never
share a thread. Two GUI event loops in one thread — or Qt in a background thread —
is fragile; a separate process decouples them entirely and keeps the sim real-time.

The process is spawned (not forked) so the GUI interpreter starts clean and doesn't
inherit the parent's MuJoCo/GL state.

Message protocol (plain dicts put on the queue):
    {'type': 'dist',    't': float, 'd': {finger: signed_distance_m}}   # streaming
    {'type': 'wrench',  't': float, 'f': [fx,fy,fz], 'tau': [tx,ty,tz]} # streaming;
                        #   rendered as 3D (Fx,Fy,Fz)/(τx,τy,τz) traces over horizon_s
    {'type': 'normals', 't': float, 'n': {finger: normal_force_N}}      # streaming
    {'type': 'rrt',     'object': str, 'n_wp': int, 'plan_time': float,
                        'fallback': bool}
    {'type': 'ipopt',   'object': str, 'status': str, 'iters': int|str,
                        'max_site_mm': float, 'max_pad_deg': float|None,
                        'min_slack_mm': float|None,
                        'dls_ms': float|None, 'ipopt_ms': float|None}
    {'type': 'mode',    'mode': str, 'target': str}   # Approach / Grasp / Transport
    {'type': 'active_obj', 'name': str}               # proximity-based active object
    {'type': 'squeeze', 'on': bool, 'gamma': float,   # GRASP internal-force toggle;
                        'f_contact': float}           #   commanded N per contact
    {'type': 'wrench_cone',                           # composite grasp wrench cone at
        'force':  {'verts': (nv,3), 'faces': (nf,3)} | None,   # solved gamma, projected
        'torque': {'verts': (nv,3), 'faces': (nf,3)} | None}   # to force / torque 3-space
    None                                              # sentinel: quit

Usage from the sim process:
    dash = Dashboard(FINGER_SET, horizon_s=5.0)
    dash.start()
    dash.push({'type': 'dist', 't': t, 'd': {'index': 0.03, 'thumb': 0.02}})
    ...
    dash.close()
"""
import multiprocessing as mp
import numpy as np

# Distinct, high-contrast colour per finger (RGB 0-255). Matches the four LEAP fingers;
# unknown finger names fall back to grey.
FINGER_COLORS = {
    'index':  (60, 170, 255),
    'middle': (110, 220, 110),
    'ring':   (255, 200, 60),
    'thumb':  (255, 95, 95),
}


def _time_gradient_rgba(n, base_rgb):
    """(n,4) RGBA ramp: oldest sample dim+transparent, newest bright+opaque, so a 3D
    trace reads as a fading trail with the current point at the head. base_rgb 0-1."""
    if n <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    t = np.linspace(0.15, 1.0, n)          # brightness/alpha ramp over the window
    rgba = np.ones((n, 4), dtype=np.float32)
    rgba[:, :3] = np.outer(t, base_rgb)
    rgba[:, 3] = t
    return rgba


def _run(queue, fingers, horizon_s, dt_hint):
    """Dashboard process entry point. Imports Qt lazily so the parent never loads it."""
    from collections import deque
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtWidgets, QtCore

    app = pg.mkQApp("hand-control dashboard")
    pg.setConfigOptions(antialias=True)

    win = QtWidgets.QWidget()
    win.setWindowTitle("hand-control — live dashboard")
    grid = QtWidgets.QGridLayout(win)

    # --- header: planning mode (big) + selected target + proximity-active object ---
    mode_lbl = QtWidgets.QLabel("—")
    mode_lbl.setStyleSheet("font-size: 30px; font-weight: bold;")
    target_lbl = QtWidgets.QLabel("target: —")
    target_lbl.setStyleSheet("font-size: 12px; color: #999999;")
    active_obj_lbl = QtWidgets.QLabel("active object: —")
    active_obj_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #55cc88;")
    _SQUEEZE_OFF_STYLE = "font-size: 16px; font-weight: bold; color: #777777;"
    _SQUEEZE_ON_STYLE  = "font-size: 16px; font-weight: bold; color: #ff9944;"
    squeeze_lbl = QtWidgets.QLabel("internal force: off")
    squeeze_lbl.setStyleSheet(_SQUEEZE_OFF_STYLE)
    mode_box = QtWidgets.QVBoxLayout()
    mode_box.addWidget(mode_lbl)
    mode_box.addWidget(target_lbl)
    header = QtWidgets.QHBoxLayout()
    header.addLayout(mode_box)
    header.addStretch(1)
    header.addWidget(squeeze_lbl)
    header.addSpacing(30)
    header.addWidget(active_obj_lbl)
    grid.addLayout(header, 0, 0, 1, 2)

    maxlen = max(10, int(horizon_s / max(dt_hint, 1e-3)))

    def _scroll_plot(title, ylabel, yunit):
        p = pg.PlotWidget(title=title)
        p.addLegend(offset=(-10, 10))
        p.showGrid(x=True, y=True, alpha=0.3)
        p.setLabel('bottom', 'time', 's')
        p.setLabel('left', ylabel, yunit)
        p.addLine(y=0.0, pen=pg.mkPen((150, 150, 150), width=1,
                                      style=QtCore.Qt.DashLine))
        return p

    # --- (1) scrolling fingertip→active-object distance plot ---
    dist_plot = _scroll_plot("Fingertip → active object  (signed geom distance)",
                             'distance', 'm')
    dist_t = deque(maxlen=maxlen)
    dist_curves, dist_buf = {}, {}
    for f in fingers:
        col = FINGER_COLORS.get(f, (200, 200, 200))
        dist_curves[f] = dist_plot.plot(pen=pg.mkPen(col, width=2), name=f)
        dist_buf[f] = deque(maxlen=maxlen)

    # --- (2) per-finger contact normal force (hand ↔ any object) ---
    norm_plot = _scroll_plot("Contact normal force  (per finger, hand ↔ objects)",
                             'force', 'N')
    norm_t = deque(maxlen=maxlen)
    norm_curves, norm_buf = {}, {}
    for f in fingers:
        col = FINGER_COLORS.get(f, (200, 200, 200))
        norm_curves[f] = norm_plot.plot(pen=pg.mkPen(col, width=2), name=f)
        norm_buf[f] = deque(maxlen=maxlen)

    # --- (3)+(4) net hand→object wrench as 3D traces: the (Fx,Fy,Fz) and (τx,τy,τz)
    # vectors plotted as a persistent trajectory in 3D over the SAME horizon_s window
    # as the 2D plots, with a colour gradient (dim/old -> bright/new) showing the
    # recent path of the applied force/moment. ---
    def _wrench_view(title, base_rgb):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        w = gl.GLViewWidget()
        w.setCameraPosition(distance=3.0, elevation=22, azimuth=-60)
        w.addItem(gl.GLGridItem())          # xy reference grid at the origin
        axis = gl.GLAxisItem()
        axis.setSize(1, 1, 1)               # unit x/y/z axis triad (R/G/B by convention)
        w.addItem(axis)
        # Composite wrench-cone hull as a WIREFRAME cage (the feasible force/torque set
        # the solved gamma assumes): the live trace staying inside the cage means the
        # grasp can supply the applied wrench. Wireframe (GLLinePlotItem) rather than a
        # filled GLMeshItem — a translucent solid would occlude the trace, and this
        # pyqtgraph build's GLMeshItem shader path is broken on this GL stack.
        cone = gl.GLLinePlotItem(mode='lines', width=1.5, antialias=True,
                                 color=(base_rgb[0], base_rgb[1], base_rgb[2], 0.5))
        w.addItem(cone)
        # Head marker (current sample) + the fading trajectory line.
        line = gl.GLLinePlotItem(width=2.0, antialias=True, mode='line_strip')
        head = gl.GLScatterPlotItem(size=9.0)
        w.addItem(line)
        w.addItem(head)
        v.addWidget(lbl)
        v.addWidget(w, 1)
        return box, line, head, cone

    force_box,  force_line,  force_head,  force_cone  = _wrench_view(
        "Net hand → object force  (Fx,Fy,Fz world, N)",        (1.0, 0.55, 0.2))
    torque_box, torque_line, torque_head, torque_cone = _wrench_view(
        "Net hand → object torque  (τx,τy,τz about COM, N·m)", (0.4, 0.7, 1.0))
    _FORCE_RGB  = np.array([1.0, 0.55, 0.2], dtype=np.float32)
    _TORQUE_RGB = np.array([0.4, 0.7, 1.0], dtype=np.float32)

    wrench_t   = deque(maxlen=maxlen)
    force_buf  = [deque(maxlen=maxlen) for _ in range(3)]
    torque_buf = [deque(maxlen=maxlen) for _ in range(3)]

    # --- (5) combined planner log: RRT + IK solutions, grouped per solve ---
    plan_log = QtWidgets.QPlainTextEdit()
    plan_log.setReadOnly(True)
    plan_log.setMaximumBlockCount(400)
    plan_log.setStyleSheet("font-family: monospace; font-size: 11px;")

    grid.addWidget(dist_plot,   1, 0)
    grid.addWidget(force_box,   1, 1)
    grid.addWidget(norm_plot,   2, 0)
    grid.addWidget(torque_box,  2, 1)
    grid.addWidget(QtWidgets.QLabel("<b>Planner solutions (RRT + IK)</b>"), 3, 0, 1, 2)
    grid.addWidget(plan_log, 4, 0, 1, 2)
    grid.setRowStretch(1, 2)
    grid.setRowStretch(2, 2)
    grid.setRowStretch(4, 1)
    win.resize(1250, 950)
    win.show()

    def drain():
        got_dist = got_norm = got_wrench = False
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
                dist_t.append(msg['t'])
                for f in fingers:
                    dist_buf[f].append(msg['d'].get(f, float('nan')))
                got_dist = True
            elif mt == 'normals':
                norm_t.append(msg['t'])
                for f in fingers:
                    norm_buf[f].append(msg['n'].get(f, float('nan')))
                got_norm = True
            elif mt == 'wrench':
                wrench_t.append(msg['t'])
                for i in range(3):
                    force_buf[i].append(msg['f'][i])
                    torque_buf[i].append(msg['tau'][i])
                got_wrench = True
            elif mt == 'rrt':
                plan_log.appendPlainText(
                    f"{msg.get('object', '?')} — RRT\n"
                    f"    {msg.get('n_wp', '?')} waypoints   "
                    f"plan time {msg.get('plan_time', float('nan'))*1e3:.0f} ms"
                    + ("   ** LINEAR FALLBACK **" if msg.get('fallback') else ""))
            elif mt == 'ipopt':
                pad    = msg.get('max_pad_deg')
                slack  = msg.get('min_slack_mm')
                dls_ms = msg.get('dls_ms')
                ipo_ms = msg.get('ipopt_ms')
                lines = [
                    f"{msg.get('object', '?')} — IK",
                    f"    {msg.get('status', '?')}   iters={msg.get('iters', '?')}",
                    f"    site err {msg.get('max_site_mm', float('nan')):.1f} mm   "
                    f"pad {'—' if pad is None else f'{pad:.1f}°'}   "
                    f"slack {'—' if slack is None else f'{slack:.1f} mm'}",
                ]
                if dls_ms is not None and ipo_ms is not None:
                    lines.append(f"    DLS {dls_ms:.0f} ms + NLP {ipo_ms:.0f} ms")
                plan_log.appendPlainText("\n".join(lines))
            elif mt == 'mode':
                mode_lbl.setText(msg.get('mode', '—'))
                target_lbl.setText(f"target: {msg.get('target', '—')}")
            elif mt == 'active_obj':
                active_obj_lbl.setText(f"active object: {msg.get('name', '—')}")
            elif mt == 'squeeze':
                if msg.get('on'):
                    squeeze_lbl.setText(
                        f"internal force: ON   γ={msg.get('gamma', float('nan')):.1f}"
                        f"  (~{msg.get('f_contact', float('nan')):.2f} N/contact)")
                    squeeze_lbl.setStyleSheet(_SQUEEZE_ON_STYLE)
                else:
                    squeeze_lbl.setText("internal force: off")
                    squeeze_lbl.setStyleSheet(_SQUEEZE_OFF_STYLE)
            elif mt == 'wrench_cone':
                # Composite grasp wrench cone at the solved gamma, precomputed in the sim
                # process (verts + triangle faces per subspace). Rendered as a wireframe:
                # expand each triangle's 3 edges into a line-segment list. None clears it.
                def _set_cone(item, key):
                    payload = msg.get(key)
                    if payload is None:
                        item.setData(pos=np.zeros((0, 3), np.float32))
                        return
                    verts = np.asarray(payload['verts'], np.float32)
                    faces = np.asarray(payload['faces'], np.int32)
                    if len(faces) == 0:
                        item.setData(pos=np.zeros((0, 3), np.float32))
                        return
                    # unique undirected edges from the triangle faces
                    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
                    e = np.unique(np.sort(e, axis=1), axis=0)
                    seg = verts[e.reshape(-1)]          # (2*n_edges, 3) line-segment pairs
                    item.setData(pos=seg)
                _set_cone(force_cone,  'force')
                _set_cone(torque_cone, 'torque')
        if got_dist:
            x = list(dist_t)
            for f in fingers:
                dist_curves[f].setData(x, list(dist_buf[f]))
        if got_norm:
            x = list(norm_t)
            for f in fingers:
                norm_curves[f].setData(x, list(norm_buf[f]))
        if got_wrench:
            fpts = np.column_stack([np.asarray(force_buf[i], dtype=np.float32)
                                    for i in range(3)])
            tpts = np.column_stack([np.asarray(torque_buf[i], dtype=np.float32)
                                    for i in range(3)])
            n = len(fpts)
            force_line.setData(pos=fpts, color=_time_gradient_rgba(n, _FORCE_RGB))
            torque_line.setData(pos=tpts, color=_time_gradient_rgba(n, _TORQUE_RGB))
            if n:
                force_head.setData(pos=fpts[-1:], color=np.array([[1, 1, 1, 1]], np.float32))
                torque_head.setData(pos=tpts[-1:], color=np.array([[1, 1, 1, 1]], np.float32))

    timer = QtCore.QTimer()
    timer.timeout.connect(drain)
    timer.start(33)  # ~30 Hz refresh
    app.exec_()


class Dashboard:
    """Handle to the dashboard process. push() is non-blocking and drops messages if
    the queue is full, so instrumenting the sim loop can never stall the sim."""

    def __init__(self, fingers, horizon_s=5.0, dt_hint=0.001, maxsize=20000):
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
