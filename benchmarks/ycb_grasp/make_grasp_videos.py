#!/usr/bin/env python3
"""Canonical grasp videos: one per (object, n_contacts), grouped by object.

Replaces the per-debug-session tag sprawl under out/tabletop/ (36 tags, 211
videos, ~1GB at the time this was written) with a single reproducible layout:

    out/tabletop/videos/<object>/n2.mp4        thumb + index
    out/tabletop/videos/<object>/n3.mp4        thumb + index + middle
    out/tabletop/videos/<object>/n2_planned.png / n2_final.png / ...
    out/tabletop/videos/summary.txt            outcome table

Grouping by OBJECT (not by tag) and naming by CONTACT COUNT means the two
arms for one object sit side by side, which is the comparison that matters;
re-running overwrites in place instead of accumulating another tag.
"""
import argparse, os, sys, shutil, traceback
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import benchmarks.ycb_grasp.pick_and_place as PP
from ycb_grasp import out_paths as OP

OBJECTS = ['017_orange', '014_lemon', '056_tennis_ball',
           '036_wood_block', '009_gelatin_box']
ARMS = [('n2', 'thumb,index'), ('n3', 'thumb,index,middle')]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--objects', nargs='*', default=OBJECTS)
    ap.add_argument('--arms', nargs='*', default=[a for a, _ in ARMS],
                    choices=[a for a, _ in ARMS])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--transport', action='store_true',
                    help='carry to the bin as well as lifting (longer clips)')
    ap.add_argument('--no-force-execute', dest='force_execute',
                    action='store_false',
                    help='stop at the pre-squeeze gap check like a normal run. '
                         'By DEFAULT these videos force execution through a failed '
                         'gap check, because a clip that ends at the abort is 4.2s '
                         'of approach and shows nothing about WHY the grasp failed.')
    ap.set_defaults(force_execute=True)
    args = ap.parse_args()

    root = OP.env_dir(OP.TABLETOP, 'videos')
    rows = []
    for obj in args.objects:
        odir = root / obj
        odir.mkdir(parents=True, exist_ok=True)
        for arm, fingers in ARMS:
            if arm not in args.arms:
                continue
            # run_pick_place writes seed<N>.* into out_dir; stage in a temp dir
            # then rename, so one object's two arms cannot collide.
            stage = odir / f'_stage_{arm}'
            stage.mkdir(parents=True, exist_ok=True)
            try:
                _, r = PP.run_pick_place(obj, args.seed, n_seeds=3, n_relin=None,
                                         out_dir=str(stage),
                                         do_transport=args.transport,
                                         fingers=fingers,
                                         force_execute=args.force_execute)
            except Exception as e:
                traceback.print_exc()
                rows.append((obj, arm, f'ERROR {type(e).__name__}', None, None, ''))
                shutil.rmtree(stage, ignore_errors=True)
                continue
            s = args.seed
            for src, dst in ((f'seed{s}.mp4',            f'{arm}.mp4'),
                             (f'seed{s}_planned.png',    f'{arm}_planned.png'),
                             (f'seed{s}.png',            f'{arm}_final.png'),
                             (f'seed{s}_grasp_contacts.pdf', f'{arm}_contacts.pdf'),
                             (f'seed{s}_seeds.pdf',      f'{arm}_seeds.pdf')):
                p = stage / src
                if p.exists():
                    shutil.move(str(p), str(odir / dst))
            shutil.rmtree(stage, ignore_errors=True)
            note = []
            if r.get('gap_check_failed'):
                g = r.get('tip_gaps_mm') or {}
                note.append('gap ' + '/'.join(f'{v:.1f}' for v in g.values()) + 'mm')
            if r.get('wrench_infeasible'):
                note.append('wrench-infeasible')
            rows.append((obj, arm, (r.get('phase_log') or ['?'])[-1],
                         r.get('lift_obj_dz_mm'), r.get('squeeze_forces_N'),
                         '; '.join(note)))
            print(f'[video] {obj:17s} {arm}  {rows[-1][2]:28s} '
                  f'lift={rows[-1][3] if rows[-1][3] is not None else float("nan"):.1f}mm',
                  flush=True)

    lines = [f'{"object":17s} {"arm":4s} {"outcome":26s} {"lift_mm":>9s}  '
             f'{"forces (N)":34s} why',
             '-' * 120]
    for obj, arm, out, lift, f, why in rows:
        fs = ' '.join(f'{k[:3]}={v:.2f}' for k, v in (f or {}).items()) or '-'
        lines.append(f'{obj:17s} {arm:4s} {str(out):26s} '
                     f'{(f"{lift:9.1f}" if lift is not None else "        -")}  '
                     f'{fs:34s} {why}')
    txt = '\n'.join(lines)
    (root / 'summary.txt').write_text(txt + '\n')
    print('\n' + txt)
    print(f'\nvideos under: {root}')


if __name__ == '__main__':
    main()
