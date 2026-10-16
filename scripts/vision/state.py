# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic progress records and convergence rules (no simulator dependencies)."""
import math
import json
import os
import time
from pathlib import Path


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temp.replace(path)


def converged(results, threshold=.98, required=3, episodes=1024, target='NEWTON'):
    if len(results)<required:
        return False
    tail=results[-required:]
    return (len({r['checkpoint_sha256'] for r in tail})==required
            and all(r['episodes']>=episodes and r['success_rate']>threshold
                    and r.get('domain_strength')==1 and r.get('target')==target for r in tail))


def stalled(results, started, hours=4, now=None):
    now=time.time() if now is None else now
    best=-1.; best_letters=-1.; last=started
    for r in results:
        letters=r.get('mean_completed_letters',0.)
        if r['success_rate']>best+.005 or letters>best_letters+.25:
            best=max(best,r['success_rate']); best_letters=max(best_letters,letters)
            last=max(last,r['finished_at'])
    return now-last>=hours*3600


def adjustment(results, generation, *, current_learning_rate):
    """Bounded changes preserve full-domain evaluation and the success threshold."""
    rate=float(current_learning_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('current_learning_rate must be finite and positive')
    # A floor must never raise a rate already below it.
    reduced=max(min(1e-5, rate), .5*rate)
    failure=results[-1].get('failure_counts',{}) if results else {}
    dominant=max(failure, key=failure.get) if failure else 'insufficient_progress'
    return {'learning_rate':reduced, 'previous_learning_rate':rate,
            'curriculum_iterations':min(4000,1000*(generation+1)),
            'recovery_warmup':min(200,50*generation),
            'reason':f'Four-hour plateau; dominant failure: {dominant}. Set PPO learning rate from {rate:g} to {reduced:g}, add teacher recovery, and extend randomization ramp.'}
