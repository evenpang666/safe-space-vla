#!/usr/bin/env python3
"""Export the public PiKA Gripper STEP assembly into collision-mesh STLs.

Requires OpenCascade Python bindings (``cadquery-ocp``):
``PYTHONPATH=/path/to/ocp python real_scripts/export_pika_step_collision_meshes.py ...``.
The STEP source uses millimetres.  Consumers must use ``scale=0.001`` in URDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_ocp():
    try:
        from OCP.BRep import BRep_Builder
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.Bnd import Bnd_Box
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.StlAPI import StlAPI_Writer
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS_Compound
    except ImportError as error:
        raise RuntimeError(
            "STEP export needs cadquery-ocp/OpenCascade. Install it in an isolated "
            "environment, then prepend that environment to PYTHONPATH."
        ) from error
    return {
        "BRep_Builder": BRep_Builder,
        "BRepBndLib": BRepBndLib,
        "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
        "Bnd_Box": Bnd_Box,
        "IFSelect_RetDone": IFSelect_RetDone,
        "STEPControl_Reader": STEPControl_Reader,
        "StlAPI_Writer": StlAPI_Writer,
        "TopAbs_SOLID": TopAbs_SOLID,
        "TopExp_Explorer": TopExp_Explorer,
        "TopoDS_Compound": TopoDS_Compound,
    }


def _make_compound(solids, ocp):
    compound = ocp["TopoDS_Compound"]()
    builder = ocp["BRep_Builder"]()
    builder.MakeCompound(compound)
    for solid in solids:
        builder.Add(compound, solid)
    return compound


def _write_stl(shape, path: Path, linear_deflection_mm: float, ocp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ocp["BRepMesh_IncrementalMesh"](shape, linear_deflection_mm, False, 0.35, True)
    writer = ocp["StlAPI_Writer"]()
    writer.Write(shape, str(path))


def _bounds_mm(shape, ocp) -> list[float]:
    box = ocp["Bnd_Box"]()
    ocp["BRepBndLib"].Add_s(shape, box)
    return [round(float(value), 4) for value in box.Get()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_step", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--linear-deflection-mm",
        type=float,
        default=1.5,
        help="surface tessellation tolerance; 1.5 mm matches D435i-scale collision masking",
    )
    args = parser.parse_args()
    if args.linear_deflection_mm <= 0.0:
        raise ValueError("--linear-deflection-mm must be positive")

    ocp = _require_ocp()
    reader = ocp["STEPControl_Reader"]()
    status = reader.ReadFile(str(args.input_step))
    if status != ocp["IFSelect_RetDone"]:
        raise RuntimeError(f"cannot read STEP file: {args.input_step}")
    if reader.TransferRoots() == 0:
        raise RuntimeError("STEP file has no transferable roots")
    assembly = reader.OneShape()
    explorer = ocp["TopExp_Explorer"](assembly, ocp["TopAbs_SOLID"])
    solids = []
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    if not solids:
        raise RuntimeError("no solid components found in STEP assembly")

    # The official PiKA STEP's last two solids are the two symmetric long finger
    # members.  Preserve that evidence as candidates rather than silently
    # claiming a kinematic convention that has not yet been verified on hardware.
    finger_candidate_indices = [len(solids) - 2, len(solids) - 1]
    body_solids = [solid for index, solid in enumerate(solids) if index not in finger_candidate_indices]
    output_dir = args.output_dir
    _write_stl(assembly, output_dir / "pika_gripper_full_collision.stl", args.linear_deflection_mm, ocp)
    _write_stl(_make_compound(body_solids, ocp), output_dir / "pika_gripper_body_collision.stl", args.linear_deflection_mm, ocp)
    for index, solid in enumerate(solids):
        _write_stl(solid, output_dir / f"component_{index + 1:02d}.stl", args.linear_deflection_mm, ocp)
    _write_stl(solids[finger_candidate_indices[0]], output_dir / "finger_a_candidate.stl", args.linear_deflection_mm, ocp)
    _write_stl(solids[finger_candidate_indices[1]], output_dir / "finger_b_candidate.stl", args.linear_deflection_mm, ocp)
    manifest = {
        "source_step": str(args.input_step),
        "source_unit": "mm",
        "urdf_mesh_scale": [0.001, 0.001, 0.001],
        "linear_deflection_mm": args.linear_deflection_mm,
        "component_count": len(solids),
        "finger_candidate_indices_zero_based": finger_candidate_indices,
        "components": [
            {"index_zero_based": index, "bounds_mm": _bounds_mm(solid, ocp)}
            for index, solid in enumerate(solids)
        ],
        "warning": (
            "Finger candidate identities and joint axes must be verified against the "
            "physical PiKA mechanism before enabling moving-finger masking."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"exported {len(solids)} PiKA STEP solids to {output_dir}")


if __name__ == "__main__":
    main()
