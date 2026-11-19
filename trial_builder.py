#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trial_builder.py -- builds the trial lists for Raiden_experiment (PsychoPy).

Put this file in the SAME folder as Raiden_experiment.py / .psyexp.
Everything you might want to change later is in the CONFIG block below.

What it does (per participant, called once at the start of the experiment)
--------------------------------------------------------------------------
* Sequence: odd participant number -> ABBA, even -> BAAB
  (A = target is face-like, B = target is non-face).
* 4 sessions x 72 encoding trials (24 target / 24 distractor / 24 baseline)
  and 4 x 72 recognition trials (4AFC, same-category foils).
* Every object category is used exactly once (288 categories, no repeats).
* Which categories fall into which (session x condition) cell is rotated
  across participant pairs (a Latin-style rotation over 12 cells).
  The studied exemplar (1-4) is rotated across participants as well.
* Central stimuli: 96 face-like (face_reference/) and 96 non-face
  (nonface_centered_0_90_180/). Each session gets 24 of each, balanced over
  shape (and over orientation/side for the non-face set).
* Writes 9 CSV files per participant into OUT_DIR (4 encoding, 4 recognition,
  and one session summary).

Command line test:   python trial_builder.py 7
"""

import csv
import os
import random
import re

# =============================== CONFIG ====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- stimulus folders (relative to BASE_DIR) ---
OBJECTS_DIR = 'objects'                    # objects/<category>/<category>_01.jpg ...
FACE_DIR = 'face_reference'                # circle_01.png ... (face-like central stimuli)
NONFACE_DIR = 'nonface_centered_0_90_180'  # 001_circle_0deg_left.png ... (non-face)
OBJECT_EXTS = ('.jpg', '.jpeg', '.png')
CENTRAL_EXTS = ('.png', '.jpg', '.jpeg')

# --- output ---
OUT_DIR = 'trial_lists'                     # participant subdirectories created inside BASE_DIR
BLANK_IMAGE = 'practice/baseline.png'                # shown at opacity 0 on baseline trials

# --- design ---
N_SESSIONS = 4
N_PER_CONDITION = 24                       # trials per condition per session
N_EXEMPLARS = 4                            # images per category used (1 studied + 3 foils)
CONDITIONS = ('target', 'distractor', 'baseline')
SEQUENCES = {'ABBA': ('A', 'B', 'B', 'A'),
             'BAAB': ('B', 'A', 'A', 'B')}
ODD_SEQUENCE = 'ABBA'                      # odd participant numbers
EVEN_SEQUENCE = 'BAAB'                     # even participant numbers
SETTING_TARGET = {'A': 'face', 'B': 'nonface'}      # which central type is the target
TARGET_LABEL = {'A': 'face-like', 'B': 'non-face'}  # wording used in the instructions
SESSION_TARGET = {'A': 'face', 'B': 'non-face'}     # wording used in the summary CSV
KEY_TARGET = 'space'                       # correct response on target trials
KEY_NONE = 'none'                          # "correct response" = no key press

# --- randomisation ---
BASE_SEED = 2026       # fixes which categories form the 12 blocks (same for everyone)
MAX_RUN = 3            # max. consecutive trials of the same condition
# ===========================================================================

N_CELLS = N_SESSIONS * len(CONDITIONS)
N_CATEGORIES = N_CELLS * N_PER_CONDITION   # 288


# ------------------------------ helpers ------------------------------------
def participant_number(participant):
    """Numeric part of the participant ID ('000012' -> 12)."""
    digits = re.sub(r'\D', '', str(participant))
    if not digits:
        raise ValueError(
            "Participant ID '%s' contains no digits. The sequence (ABBA/BAAB) "
            "is derived from the number, so please use a numeric ID." % participant)
    return int(digits)


def sequence_for(n):
    return ODD_SEQUENCE if n % 2 == 1 else EVEN_SEQUENCE


def _rel(*parts):
    return '/'.join(parts)


def scan_objects():
    """Return [(category, [path1..pathN_EXEMPLARS]), ...] in a fixed, shuffled order."""
    root = os.path.join(BASE_DIR, OBJECTS_DIR)
    if not os.path.isdir(root):
        raise FileNotFoundError("Folder not found: %s" % root)
    cats = []
    for cat in sorted(os.listdir(root)):
        d = os.path.join(root, cat)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(OBJECT_EXTS))
        if len(files) < N_EXEMPLARS:
            print("WARNING: category '%s' has only %d images, skipped." % (cat, len(files)))
            continue
        cats.append((cat, [_rel(OBJECTS_DIR, cat, f) for f in files[:N_EXEMPLARS]]))
    if len(cats) < N_CATEGORIES:
        raise ValueError("Need %d object categories with >= %d images each, found %d."
                         % (N_CATEGORIES, N_EXEMPLARS, len(cats)))
    random.Random(BASE_SEED).shuffle(cats)
    return cats[:N_CATEGORIES]


_NONFACE_RE = re.compile(r'^(?P<num>\d+)_(?P<shape>[A-Za-z]+)_(?P<ori>\d+)deg_(?P<side>left|right)',
                         re.IGNORECASE)
_FACE_RE = re.compile(r'^(?P<shape>[A-Za-z]+)_(?P<num>\d+)')


def scan_central(folder):
    """List central stimuli (dicts) with shape/orientation/side parsed from the file name."""
    root = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(root):
        raise FileNotFoundError("Folder not found: %s" % root)
    items = []
    for f in sorted(os.listdir(root)):
        if not f.lower().endswith(CENTRAL_EXTS):
            continue
        m = _NONFACE_RE.match(f)
        if m:
            shape, ori, side = m.group('shape').lower(), int(m.group('ori')), m.group('side').lower()
        else:
            m = _FACE_RE.match(f)
            shape, ori, side = (m.group('shape').lower() if m else 'na'), -1, ''
        items.append(dict(path=_rel(folder, f), shape=shape, orient=ori, side=side, name=f))
    items.sort(key=lambda d: (d['shape'], d['orient'], d['side'], d['name']))
    return items


def split_subsets(items, what):
    """Deal the sorted stimuli round-robin into N_SESSIONS subsets (balanced over shape/orientation/side)."""
    subsets = [[] for _ in range(N_SESSIONS)]
    for k, it in enumerate(items):
        subsets[k % N_SESSIONS].append(it)
    sizes = [len(s) for s in subsets]
    if any(s != N_PER_CONDITION for s in sizes):
        raise ValueError("%s: found %d images -> subsets of %s; need %d per session (%d in total)."
                         % (what, len(items), sizes, N_PER_CONDITION, N_PER_CONDITION * N_SESSIONS))
    return subsets


def _max_run(rows):
    best = run = 0
    prev = None
    for r in rows:
        run = run + 1 if r['condition'] == prev else 1
        prev = r['condition']
        best = max(best, run)
    return best


def _pseudorandom(rows, rng):
    for _ in range(20000):
        rng.shuffle(rows)
        if _max_run(rows) <= MAX_RUN:
            return rows
    raise RuntimeError("Could not satisfy MAX_RUN=%d" % MAX_RUN)


# ------------------------------ main builder -------------------------------
def build_all(participant, write=True):
    """Build (and optionally write) all trial lists for one participant."""
    n = participant_number(participant)
    seq = sequence_for(n)
    settings = SEQUENCES[seq]
    group = (n - 1) // 2          # participants 1&2 -> 0, 3&4 -> 1, ...

    cats = scan_objects()
    face_sub = split_subsets(scan_central(FACE_DIR), 'Face-like stimuli (%s)' % FACE_DIR)
    non_sub = split_subsets(scan_central(NONFACE_DIR), 'Non-face stimuli (%s)' % NONFACE_DIR)

    pid = re.sub(r'[^A-Za-z0-9]+', '', str(participant)) or str(n)
    out_dir = os.path.join(BASE_DIR, OUT_DIR, pid)
    if write:
        os.makedirs(out_dir, exist_ok=True)

    enc_files, rec_files, all_enc, all_rec = [], [], [], []
    session_summary = [
        dict(session=s + 1, target=SESSION_TARGET[settings[s]], session_type=settings[s])
        for s in range(N_SESSIONS)
    ]
    for s in range(N_SESSIONS):
        setting = settings[s]
        rng = random.Random('%d-%d' % (n, s))
        sub_j = (s + group) % N_SESSIONS
        stims = {'face': list(face_sub[sub_j]), 'nonface': list(non_sub[sub_j])}
        target_type = SETTING_TARGET[setting]
        distractor_type = 'nonface' if target_type == 'face' else 'face'
        pools = {'target': stims[target_type], 'distractor': stims[distractor_type], 'baseline': []}
        for p in pools.values():
            rng.shuffle(p)

        enc_rows = []
        for c, cond in enumerate(CONDITIONS):
            cell = s * len(CONDITIONS) + c
            block = (cell - group) % N_CELLS
            ranks = list(range(block * N_PER_CONDITION, (block + 1) * N_PER_CONDITION))
            rng.shuffle(ranks)
            for i, rank in enumerate(ranks):
                cat, paths = cats[rank]
                studied = (rank + group) % N_EXEMPLARS
                if cond == 'baseline':
                    st, ctype = None, 'none'
                else:
                    st, ctype = pools[cond][i], (target_type if cond == 'target' else distractor_type)
                enc_rows.append(dict(
                    session_n=s + 1, sequence=seq, block_setting=setting, condition=cond,
                    central_type=ctype,
                    central_img=st['path'] if st else BLANK_IMAGE,
                    central_opacity=1 if st else 0,
                    central_shape=st['shape'] if st else '',
                    central_orient=st['orient'] if st else '',
                    central_side=st['side'] if st else '',
                    background_img=paths[studied],
                    obj_category=cat, obj_exemplar=studied + 1,
                    correct_key=KEY_TARGET if cond == 'target' else KEY_NONE,
                    _paths=paths, _studied=studied))
        _pseudorandom(enc_rows, rng)
        for t, r in enumerate(enc_rows, 1):
            r['trial_n'] = t

        # correct-answer position (1-4) balanced within each condition
        slots = {}
        for cond in CONDITIONS:
            sl = [i % N_EXEMPLARS for i in range(N_PER_CONDITION)]
            rng.shuffle(sl)
            slots[cond] = sl
        rec_rows = []
        for r in enc_rows:
            pos = slots[r['condition']].pop()
            others = [j for j in range(N_EXEMPLARS) if j != r['_studied']]
            rng.shuffle(others)
            perm = others[:pos] + [r['_studied']] + others[pos:]   # perm[position] = exemplar index
            opts = [r['_paths'][j] for j in perm]
            rec_rows.append(dict(
                session_n=r['session_n'], sequence=seq, block_setting=setting,
                condition=r['condition'], enc_trial_n=r['trial_n'],
                obj_category=r['obj_category'], studied_exemplar=r['_studied'] + 1,
                studied_position=perm.index(r['_studied']) + 1,
                option1_img=opts[0], option2_img=opts[1], option3_img=opts[2], option4_img=opts[3],
                correct_key=perm.index(r['_studied']) + 1))
        rng.shuffle(rec_rows)
        for t, r in enumerate(rec_rows, 1):
            r['rec_trial_n'] = t

        enc_cols = ['trial_n', 'session_n', 'sequence', 'block_setting', 'condition', 'central_type',
                    'central_img', 'central_opacity', 'central_shape', 'central_orient', 'central_side',
                    'background_img', 'obj_category', 'obj_exemplar', 'correct_key']
        rec_cols = ['rec_trial_n', 'session_n', 'sequence', 'block_setting', 'condition', 'enc_trial_n',
                    'obj_category', 'studied_exemplar', 'studied_position',
                    'option1_img', 'option2_img', 'option3_img', 'option4_img', 'correct_key']
        enc_name = _rel(os.path.join(OUT_DIR, pid), 'P%s_S%d_encoding.csv' % (pid, s + 1))
        rec_name = _rel(os.path.join(OUT_DIR, pid), 'P%s_S%d_recognition.csv' % (pid, s + 1))
        if write:
            for name, cols, rows in ((enc_name, enc_cols, enc_rows), (rec_name, rec_cols, rec_rows)):
                with open(os.path.join(BASE_DIR, name), 'w', newline='', encoding='utf-8') as fh:
                    w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
                    w.writeheader()
                    w.writerows(rows)
        enc_files.append(enc_name)
        rec_files.append(rec_name)
        all_enc.append(enc_rows)
        all_rec.append(rec_rows)

    summary_name = _rel(os.path.join(OUT_DIR, pid), 'P%s_session_summary.csv' % pid)
    if write:
        with open(os.path.join(BASE_DIR, summary_name), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['session', 'target', 'session_type'])
            w.writeheader()
            w.writerows(session_summary)

    _validate(all_enc, all_rec)
    return dict(participant_n=n, sequence=seq, settings=list(settings), group=group,
                enc_files=enc_files, rec_files=rec_files,
                summary_file=summary_name,
                target_labels=[TARGET_LABEL[x] for x in settings],
                _enc=all_enc, _rec=all_rec)


def _validate(all_enc, all_rec):
    cats, faces, nons = set(), set(), set()
    for enc, rec in zip(all_enc, all_rec):
        assert len(enc) == len(rec) == N_PER_CONDITION * len(CONDITIONS)
        for cond in CONDITIONS:
            assert sum(r['condition'] == cond for r in enc) == N_PER_CONDITION, cond
        for r in enc:
            assert r['obj_category'] not in cats, 'category repeated: ' + r['obj_category']
            cats.add(r['obj_category'])
            if r['central_type'] == 'face':
                faces.add(r['central_img'])
            elif r['central_type'] == 'nonface':
                nons.add(r['central_img'])
            for p in (r['background_img'], r['central_img']):
                if p != BLANK_IMAGE:
                    assert os.path.exists(os.path.join(BASE_DIR, p)), 'missing file: ' + p
    assert len(cats) == N_CATEGORIES
    assert len(faces) == len(nons) == N_PER_CONDITION * N_SESSIONS


if __name__ == '__main__':
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else '1'
    info = build_all(pid)
    print('Participant %s -> %s %s' % (pid, info['sequence'], info['settings']))
    for e, r in zip(info['enc_files'], info['rec_files']):
        print('  ', e, '|', r)
    print('  ', info['summary_file'])
