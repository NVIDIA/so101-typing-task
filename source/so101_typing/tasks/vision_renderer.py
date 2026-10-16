# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Render the authored Workshop camera with scaled OpenCV calibration."""
from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
from pxr import Gf, Usd, UsdGeom


class TypingOVRTXRenderer(OVRTXRenderer):
    def _process_render_frame(self, render_data, frame, output_buffers):
        import warp as wp
        from ovrtx import Device
        # OVRTX 0.5 returns USD paths and a direct DLPack mapping object.
        variables = {str(name).rsplit('/', 1)[-1]: value for name, value in frame.render_vars.items()}
        if 'LdrColor' not in variables or 'rgba' not in output_buffers:
            raise RuntimeError(f'OVRTX RGB output unavailable: {list(variables)}')
        render_data.renderer_info.clear()
        with variables['LdrColor'].map(device=Device.CUDA) as mapping:
            pixels = wp.from_dlpack(mapping)
            self._extract_rgba_tiles(render_data, pixels, output_buffers, 'rgba')

    def prepare_cameras(self, stage, spec):
        super().prepare_cameras(stage, spec)
        for path in spec.camera_prim_paths:
            prim = stage.GetPrimAtPath(path)
            native = prim.GetAttribute('omni:lensdistortion:opencvPinhole:imageSize').Get()
            if native is None:
                raise RuntimeError('Workshop OpenCV wrist camera intrinsics are missing')
            width, height = spec.cfg.width, spec.cfg.height
            for name, ratio in (('fx', width/native[0]), ('cx', width/native[0]),
                                ('fy', height/native[1]), ('cy', height/native[1])):
                attr = prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+name)
                attr.Set(attr.Get()*ratio)
            prim.GetAttribute('omni:lensdistortion:opencvPinhole:imageSize').Set(Gf.Vec2i(width, height))

    def prepare_stage(self, stage, num_envs):
        # Render a separate stage so hiding proxies cannot alter physics import.
        stage = Usd.Stage.Open(stage.Flatten())
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if ('/Keyboard/keys/' in path and path.endswith('/collision')) or path.endswith('/Keyboard/base_link/case_collision'):
                UsdGeom.Imageable(prim).MakeInvisible()
        super().prepare_stage(stage, num_envs)


def register_renderer():
    """Register the project adapter before the pinned factory resolves OVRTX."""
    from isaaclab.renderers.renderer import Renderer
    Renderer.register('ov', TypingOVRTXRenderer)
