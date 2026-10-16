"""Export a frozen Newton scene for the isolated OVRTX renderer."""
from pathlib import Path
import json
import math


def hide_keyboard_collision_visuals(stage):
    """Hide proxy boxes in a render-only stage while retaining scanned visuals."""
    from pxr import UsdGeom

    hidden = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        is_key_proxy = '/Keyboard/keys/' in path and path.endswith('/collision')
        is_case_proxy = path.endswith('/Keyboard/base_link/case_collision')
        if (is_key_proxy or is_case_proxy) and prim.IsA(UsdGeom.Gprim):
            # These proxies intersect the visible scanned shells. Rendering both
            # causes unstable surface selection as the wrist camera moves.
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden.append(path)
    return hidden


def export_scene(env, output: Path, eye, lookat):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdRender
    from isaaclab_newton.physics import NewtonManager

    output.parent.mkdir(parents=True, exist_ok=True)
    # Edit a flattened copy; the simulation's stage and joint state stay intact.
    stage = Usd.Stage.Open(env.sim.stage.Flatten())
    hidden_colliders = hide_keyboard_collision_visuals(stage)
    model = NewtonManager.get_model()
    poses = NewtonManager.get_state().body_q.numpy()
    cache = UsdGeom.XformCache()
    scales = {path: Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).GetScale()
              for path in model.body_label if stage.GetPrimAtPath(path)}
    count = 0
    for path, pose in zip(model.body_label, poses):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f'Newton body missing from USD: {path}')
        x, y, z, qx, qy, qz, qw = map(float, pose)
        transform = Gf.Transform()
        transform.SetScale(scales[path])
        transform.SetRotation(Gf.Rotation(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))))
        transform.SetTranslation(Gf.Vec3d(x, y, z))
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(transform.GetMatrix())
        xform.SetResetXformStack(True)
        count += 1
    camera = UsdGeom.Camera.Define(stage, '/RenderCamera')
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*lookat), Gf.Vec3d(0, 0, 1))
    camera.AddTransformOp().Set(view.GetInverse())
    # Match the Newton viewer's vertical 45 degree field of view.
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateVerticalApertureAttr(48.0 * math.tan(math.radians(22.5)))
    camera.CreateHorizontalApertureAttr(camera.GetVerticalApertureAttr().Get() * 16 / 9)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    product = UsdRender.Product.Define(stage, '/Render/Tabletop')
    product.CreateCameraRel().SetTargets([camera.GetPath()])
    product.CreateResolutionAttr(Gf.Vec2i(1920, 1080))
    color = UsdRender.Var.Define(stage, '/Render/Vars/LdrColor')
    color.CreateSourceNameAttr('LdrColor')
    product.CreateOrderedVarsRel().SetTargets([color.GetPath()])
    p = product.GetPrim()
    p.AddAppliedSchema('OmniRtxSettingsCommonAdvancedAPI_1')
    p.CreateAttribute('omni:rtx:rendermode', Sdf.ValueTypeNames.Token).Set('PathTracing')
    p.CreateAttribute('omni:rtx:waitForEvents', Sdf.ValueTypeNames.TokenArray).Set(['AllLoadingFinished', 'OnlyOnFirstRequest'])
    p.CreateAttribute('omni:rtx:background:source:type', Sdf.ValueTypeNames.Token).Set('domeLight')
    stage.Export(str(output))
    report = {'scene': output.name, 'hidden_keyboard_collision_visuals': hidden_colliders, 'newton_bodies_exported': count, 'quaternion_order': 'xyzw',
              'eye_m': list(eye), 'lookat_m': list(lookat), 'resolution': [1920, 1080],
              'joint_positions_rad': {name: env.scene[name].data.joint_pos.torch.cpu().tolist()
                                      for name in ('robot', 'keyboard')},
              'physics_steps_during_export': 0}
    output.with_suffix('.snapshot.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'POSED USD EXPORTED: {output} ({count} Newton bodies)', flush=True)
