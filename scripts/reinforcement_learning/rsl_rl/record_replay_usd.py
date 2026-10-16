# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Record exact Newton body poses for camera rendering without rerunning physics."""
import json
from pathlib import Path

from export_tabletop_usd import export_scene


class ReplayUsdRecorder:
    def __init__(self, env, output: Path):
        from isaaclab_newton.physics import NewtonManager
        self.env = env
        self.output = output
        self.manager = NewtonManager
        self.paths = list(NewtonManager.get_model().body_label)
        self.frames = []
        self.camera_frames = []
        export_scene(env, output, env.cfg.viewer.eye, env.cfg.viewer.lookat)
        self.capture()

    def capture(self):
        self.frames.append(self.manager.get_state().body_q.numpy().copy())
        from pxr import UsdGeom
        prim = self.env.sim.stage.GetPrimAtPath('/OmniverseKit_Persp')
        if prim:
            self.camera_frames.append(UsdGeom.XformCache().GetLocalToWorldTransform(prim))

    def save(self):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdRender
        stage = Usd.Stage.Open(str(self.output))
        fps = 1.0 / self.env.step_dt
        stage.SetTimeCodesPerSecond(fps)
        stage.SetFramesPerSecond(fps)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(len(self.frames) - 1)
        for body, path in enumerate(self.paths):
            prim = stage.GetPrimAtPath(path)
            op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
            scale = Gf.Transform(op.Get()).GetScale()
            for frame, poses in enumerate(self.frames):
                x, y, z, qx, qy, qz, qw = map(float, poses[body])
                transform = Gf.Transform()
                transform.SetScale(scale)
                transform.SetRotation(Gf.Rotation(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))))
                transform.SetTranslation(Gf.Vec3d(x, y, z))
                op.Set(transform.GetMatrix(), Usd.TimeCode(frame))
        if self.camera_frames:
            overview = UsdGeom.Xformable(stage.GetPrimAtPath('/OmniverseKit_Persp'))
            overview.ClearXformOpOrder()
            overview.SetResetXformStack(True)
            op = overview.AddTransformOp()
            for frame, matrix in enumerate(self.camera_frames):
                op.Set(matrix, Usd.TimeCode(frame))
            matrix = self.camera_frames[-1]
            eye = matrix.ExtractTranslation()
            backward = Gf.Vec3d(*matrix[2][:3]).GetNormalized()
            distance = sum((a-b)**2 for a,b in zip(self.env.cfg.viewer.eye, self.env.cfg.viewer.lookat))**0.5
            final_camera = {'eye_m': list(eye), 'lookat_m': list(eye - backward * distance),
                            'source': 'actual final interactive Kit perspective camera'}
            self.output.with_name('overview-camera-final.json').write_text(json.dumps(final_camera, indent=2) + '\n')
        cameras = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)
                   and p.GetName() == 'wowrobo_2MP_camera']
        if len(cameras) != 1:
            raise RuntimeError(f'Expected one wrist camera, found {len(cameras)}')
        product = UsdRender.Product.Define(stage, '/Render/Wrist')
        product.CreateCameraRel().SetTargets([cameras[0].GetPath()])
        product.CreateResolutionAttr(Gf.Vec2i(640, 480))
        product.CreateOrderedVarsRel().SetTargets(['/Render/Vars/LdrColor'])
        prim = product.GetPrim()
        prim.AddAppliedSchema('OmniRtxSettingsCommonAdvancedAPI_1')
        prim.CreateAttribute('omni:rtx:rendermode', Sdf.ValueTypeNames.Token).Set('PathTracing')
        prim.CreateAttribute('omni:rtx:waitForEvents', Sdf.ValueTypeNames.TokenArray).Set(['AllLoadingFinished', 'OnlyOnFirstRequest'])
        prim.CreateAttribute('omni:rtx:background:source:type', Sdf.ValueTypeNames.Token).Set('domeLight')
        stage.Export(str(self.output))
        report = {'scene': self.output.name, 'frames': len(self.frames), 'fps': fps,
                  'duration_seconds': len(self.frames) / fps, 'body_count': len(self.paths),
                  'wrist_camera': str(cameras[0].GetPath()), 'wrist_resolution': [640, 480],
                  'source': 'same evaluation rollout, initial state followed by every control step',
                  'wrist_optics': 'unchanged Workshop asset camera with authored lens distortion'}
        self.output.with_suffix('.replay.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'ANIMATED USD EXPORTED: {self.output} ({len(self.frames)} frames)', flush=True)
        return report
