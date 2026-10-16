# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline fixture drawing and three-landmark rigid placement fit (standard library only)."""
import hashlib
import html
import json
import math
from pathlib import Path
from scripts.reinforcement_learning.rsl_rl.public_task import nominal_geometry, rotate_xyzw
from scripts.robot_model_identity import WORKSHOP_USD_SHA256

ROOT = Path(__file__).resolve().parents[1]
LANDMARKS = ('Q', 'P', 'Z')
MAX_RESIDUAL_MM = 1.0  # Measurement consistency only; not policy tolerance.


def load_geometry():
    data = json.loads((ROOT / 'docs/fixtures/model_geometry.json').read_text())
    if data['robot_usd_sha256'] != WORKSHOP_USD_SHA256:
        raise ValueError('fixture geometry uses a different robot USD; regenerate with scripts/build_fixture_geometry.py')
    for name, expected in data['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('fixture drawing geometry is stale; regenerate with scripts/build_fixture_geometry.py')
    return data


def transform(point, geometry, task):
    return nominal_geometry(geometry['reference_position_m'], geometry['reference_rotation_xyzw'],
                            [point], task.get('keyboard', {}))[2][0]


def outline(points):
    """Convex XY projection of the transformed three-dimensional case hull."""
    points = sorted(set(tuple(p[:2]) for p in points))
    def half(seq):
        result = []
        for p in seq:
            while len(result) >= 2:
                a, b = result[-2:]
                if (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) > 0:
                    break
                result.pop()
            result.append(p)
        return result
    return half(points)[:-1]+half(reversed(points))[:-1]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def basis(points):
    def unit(v):
        length = math.hypot(*v)
        if length < 1e-6:
            raise ValueError('Q/P/Z must be distinct, non-collinear landmarks')
        return [x/length for x in v]
    u = unit([p-q for p,q in zip(points['P'], points['Q'])])
    v = [q-z for q,z in zip(points['Q'], points['Z'])]
    projection = dot(u, v)
    v = unit([y-projection*x for x,y in zip(u,v)])
    n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    return u, v, n


def quaternion(columns):
    # Convert an orthonormal rotation matrix without a singularity at 180 degrees.
    m = list(zip(*columns))
    values = [1+m[0][0]-m[1][1]-m[2][2], 1-m[0][0]+m[1][1]-m[2][2],
              1-m[0][0]-m[1][1]+m[2][2], 1+m[0][0]+m[1][1]+m[2][2]]
    i = max(range(4), key=values.__getitem__)
    s = 2*math.sqrt(max(0, values[i]))
    candidates = [(s/4, (m[0][1]+m[1][0])/s, (m[0][2]+m[2][0])/s, (m[2][1]-m[1][2])/s),
                  ((m[0][1]+m[1][0])/s, s/4, (m[1][2]+m[2][1])/s, (m[0][2]-m[2][0])/s),
                  ((m[0][2]+m[2][0])/s, (m[1][2]+m[2][1])/s, s/4, (m[1][0]-m[0][1])/s),
                  ((m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s, s/4)]
    q = candidates[i]
    return [v/math.hypot(*q) for v in q]


def landmarks(geometry, task):
    datum = geometry['datum_m']
    return {key['label']: [(v-datum[i])*1000 for i, v in enumerate(transform(key['rim_center_m'], geometry, task))]
            for key in geometry['keys'] if key['slot'] in ('main_q', 'main_p', 'main_z')}


def fit(geometry, measured):
    if not isinstance(measured, dict) or set(measured) != set(LANDMARKS):
        raise ValueError('measurements must contain exactly Q, P and Z, each [X, Y, Z] in mm from datum D')
    for key, values in measured.items():
        if (not isinstance(values, list) or len(values) != 3 or
                any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 2000 for v in values)):
            raise ValueError(f'{key}: replace all three entries with finite millimetre measurements within +/-2000 mm of datum D')
    reference = landmarks(geometry, {})
    means = [[sum(points[k][i] for k in LANDMARKS)/3 for i in range(3)]
             for points in (reference, measured)]
    a = [[reference[k][i]-means[0][i] for i in range(3)] for k in LANDMARKS]
    b = [[measured[k][i]-means[1][i] for i in range(3)] for k in LANDMARKS]
    source, target = basis(reference), basis(measured)
    if target[2][2] <= 0:
        raise ValueError('Q/P/Z imply an upside-down keyboard; check mirrored axis signs and landmark labels')
    # Fit the two landmark planes, then minimize in-plane squared distance.
    a2 = [[dot(v,axis) for axis in source[:2]] for v in a]
    b2 = [[dot(v,axis) for axis in target[:2]] for v in b]
    angle = math.atan2(sum(u[0]*v[1]-u[1]*v[0] for u,v in zip(a2,b2)),
                       sum(dot(u,v) for u,v in zip(a2,b2)))
    c,s = math.cos(angle),math.sin(angle)
    target = [[c*u+s*v for u,v in zip(target[0],target[1])],
              [-s*u+c*v for u,v in zip(target[0],target[1])], target[2]]
    def turn(vector):
        return [sum(dot(vector,source[j])*target[j][i] for j in range(3)) for i in range(3)]
    rotated = turn(means[0])
    translation = [means[1][i]-rotated[i] for i in range(3)]
    errors = {k: math.dist([v+translation[i] for i, v in enumerate(turn(reference[k]))], measured[k])
              for k in LANDMARKS}
    if max(errors.values()) > MAX_RESIDUAL_MM:
        raise ValueError(f'landmarks do not fit a rigid keyboard pose: max residual {max(errors.values()):.3f} mm '
                         f'exceeds {MAX_RESIDUAL_MM:.1f} mm. Check units, base datum, key centres and keyboard dimensions.')
    # Convert the fit about physical datum D to the task's CAD-origin pivot.
    origin_d = [(v-geometry['datum_m'][i])*1000 for i, v in enumerate(geometry['reference_position_m'])]
    origin = [geometry['datum_m'][i]+(v+translation[i])/1000 for i, v in enumerate(turn(origin_d))]
    rotation = quaternion([turn(rotate_xyzw(geometry['reference_rotation_xyzw'], axis))
                           for axis in ([1,0,0], [0,1,0], [0,0,1])])
    task = {'schema_version': 1, 'keyboard': {'position_m': origin, 'rotation_xyzw': rotation}}
    return task, {'landmark_residual_mm': errors, 'max_residual_mm': max(errors.values()),
                  'measurement_consistency_limit_mm': MAX_RESIDUAL_MM,
                  'meaning': 'Rigid model fit only. Does not validate reachability, calibration, contact or policy performance.'}


def drawing(geometry, task):
    # Robot +X points up the page, +Y left. Equal XY scale throughout.
    case = outline([transform(p, geometry, task) for p in geometry['case_hull_m']])
    points = geometry['base_outline_xy_m'] + case
    if any(not math.isfinite(v) or abs(v) > 2 for p in points for v in p):
        raise ValueError('fixture drawing supports coordinates within +/-2 metres of the robot; check position units')
    xs, ys = [p[0]*1000 for p in points], [p[1]*1000 for p in points]
    xmin, xmax, ymin, ymax = min(xs)-55, max(xs)+35, min(ys)-35, max(ys)+35
    scale = min(1040/(ymax-ymin), 770/(xmax-xmin))
    def xy(p):
        return 80+(ymax-p[1]*1000)*scale, 105+(xmax-p[0]*1000)*scale
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1120" role="img" aria-labelledby="title desc">',
             '<title id="title">SO-101 and MX Keys: simulator layout reference</title>',
             '<desc id="desc">Model projection, not verified physical fixture measurements. X points from the robot toward the keyboard; Y points left. D is a CAD base datum. Q, P and Z mark model keycap rim centres.</desc>',
             '<rect width="1200" height="1120" fill="#f8fafc"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172c42}.label{font-size:17px}.small{font-size:14px}</style>',
             '<text x="50" y="42" font-size="27" font-weight="bold">SO-101 → MX Keys | simulator layout reference</text>',
             '<text x="50" y="73" class="label">Model dimensions in mm • configured scene • physical ruler dimensions are not verified</text>']
    def line(a, b, color='#9aacbd', dash=''):
        ax, ay = xy(a); bx, by = xy(b)
        parts.append(f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" stroke="{color}" stroke-width="1.5" {dash}/>')
    def polygon(vertices, fill, stroke):
        p = ' '.join(f'{x:.2f},{y:.2f}' for x,y in map(xy, vertices))
        parts.append(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
    def label(p, text, color='#172c42', dx=10, dy=-8):
        x,y = xy(p)
        parts.append(f'<text x="{x+dx:.2f}" y="{y+dy:.2f}" class="label" style="fill:{color}">{html.escape(text)}</text>')
    for x in range(math.ceil(xmin/50)*50, math.floor(xmax/50)*50+1, 50):
        line([x/1000,ymin/1000], [x/1000,ymax/1000], '#e0e7ef')
    for y in range(math.ceil(ymin/50)*50, math.floor(ymax/50)*50+1, 50):
        line([xmin/1000,y/1000], [xmax/1000,y/1000], '#e0e7ef')
    polygon(geometry['base_outline_xy_m'], '#c6d9e8', '#456881')
    polygon(case, '#dce4ec', '#456881')
    for key in geometry['keys']:
        special = key['slot'] in ('main_q','main_p','main_z')
        polygon([transform(p, geometry, task) for p in key['outline_m']], '#b7eed9' if special else '#ffffff', '#98a9ba')
        if len(key['label']) == 1 and key['label'].isalpha():
            p = transform(key['rim_center_m'], geometry, task)
            x,y = xy(p)
            parts.append(f'<text x="{x:.2f}" y="{y+4:.2f}" text-anchor="middle" class="small">{html.escape(key["label"])}</text>')
    datum = geometry['datum_m']
    for axis, end in [('+X', [datum[0]+.08,0]), ('+Y', [datum[0],.08])]:
        line(datum, end, '#0c8060'); label(end, axis, '#0c8060')
        x,y=xy(end); parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#0c8060"/>')
    for p, title, color in [(datum,'D: front-base datum','#0c8060'),([0,0],'O: robot CAD origin','#456881'),
                            (transform(geometry['reference_position_m'],geometry,task),'K','#99450d')]:
        x,y=xy(p);parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>');label(p,title,color,dy=24)
    # Dimension legs for Q expose the measurement signs without implying a ruler diagonal.
    q = next(transform(k['rim_center_m'],geometry,task) for k in geometry['keys'] if k['slot']=='main_q')
    elbow = [q[0],0]
    line(datum, elbow, '#0c8060', 'stroke-dasharray="5 4"')
    line(elbow, q, '#0c8060', 'stroke-dasharray="5 4"')
    label([(datum[0]+q[0])/2,0], 'Q: X distance', '#0c8060')
    parts += ['<text x="50" y="925" class="label" font-weight="bold">Model keycap rim centres relative to D (X, Y, Z mm)</text>']
    expected = landmarks(geometry, task)
    for i,k in enumerate(LANDMARKS):
        parts.append(f'<text x="50" y="{955+i*26}" class="label">{k}: ({", ".join(f"{v:.3f}" for v in expected[k])})</text>')
    d=', '.join(f'{v*1000:.3f}' for v in datum)
    parts += [f'<text x="620" y="929" class="small">K = keyboard CAD origin (not a ruler datum)</text><text x="620" y="955" class="small">D in robot CAD frame: ({d}) mm</text>',
              '<text x="620" y="981" class="small">Z = height above the plastic base underside plane.</text>',
              '<text x="620" y="1007" class="small">Clamp/table height is not automatically this zero plane.</text>',
              '<text x="50" y="1060" class="label">Q/P/Z: centre of the key outline at its top rim, keys released; not the recessed bowl.</text>',
              '<text x="50" y="1090" class="label">Fit consistency is not physical validation. Match the base shape and follow FIXTURE_SETUP.md.</text>', '</svg>']
    return '\n'.join(parts)+'\n'


def create(out_dir, measurements=None, task_config=None):
    from scripts.reinforcement_learning.rsl_rl.public_task import load_config
    geometry = load_geometry()
    report = {'mode': 'reference-model', 'hardware_access': False, 'physical_alignment_validated': False,
              'limits': 'Rigid keyboard placement from model geometry. Three landmarks set XYZ and full orientation; '
                        'this does not measure joint offsets or verify physical contacts.'}
    if measurements:
        measured = json.loads(Path(measurements).read_text())
        task, details = fit(geometry, measured)
        report.update(mode='measured-model-fit', measurements_mm=measured, **details)
    elif task_config:
        task = load_config(task_config)
        report['mode'] = 'configured-model'
    else:
        task = load_config(ROOT / 'configs/tasks/physical-reference-20260915.json')
        report['mode'] = 'physical-reference-model'
    report.update(datum_m=geometry['datum_m'], source_sha256=geometry['source_sha256'],
                  robot_usd_sha256=geometry['robot_usd_sha256'], task=task)
    svg = drawing(geometry, task)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, data in [('task.json',task), ('measurement-report.json',report),
                       ('expected-landmarks.json',landmarks(geometry,task)),
                       ('measurement-template.json',{k:[None,None,None] for k in LANDMARKS})]:
        (out_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (out_dir/'birdseye.svg').write_text(svg)
    print(f'Fixture drawing: {out_dir / "birdseye.svg"}')
    print(f'Candidate simulation task: {out_dir / "task.json"}')
    print('Model reference only. Dimensions are not verified physical setup instructions; contact alignment needs measurement.')
    return 0
