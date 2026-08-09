#!/usr/bin/env python3
"""Convert saved HGFD particle/fluid fields to a ParaView VTI/PVD series."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="hgfd_fluidized_bed")
    parser.add_argument("--t-final", type=float, required=True)
    return parser.parse_args()


def indexed_files(directory, prefix, suffix):
    return {int(path.stem.rsplit("_", 1)[1]): path for path in directory.glob(f"{prefix}_[0-9]*{suffix}")}


def add_cell_array(image, name, values, components=1):
    values = np.asarray(values, dtype=np.float32)
    if components == 1:
        flattened = values.ravel(order="F")
    else:
        flattened = np.stack(
            [values[:, :, component].ravel(order="F") for component in range(components)],
            axis=1,
        )
    vtk_array = numpy_to_vtk(flattened, deep=True)
    vtk_array.SetName(name)
    vtk_array.SetNumberOfComponents(components)
    image.GetCellData().AddArray(vtk_array)


def write_pvd(path, entries):
    vtk_file = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(vtk_file, "Collection")
    for time_value, filename in entries:
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{time_value:.8g}",
            group="",
            part="0",
            file=filename,
        )
    tree = ET.ElementTree(vtk_file)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {
        "nu": indexed_files(args.input, "nu", ".npy"),
        "s_bar": indexed_files(args.input, "s_bar", ".npy"),
        "u": indexed_files(args.input, "u", ".npy"),
        "v": indexed_files(args.input, "v", ".npy"),
        "fluid": indexed_files(args.input, "fluid", ".npz"),
    }
    frame_ids = sorted(set.intersection(*(set(files) for files in sources.values())))
    if not frame_ids:
        raise RuntimeError(f"No complete HGFD frames found in {args.input}")

    x = np.atleast_1d(np.loadtxt(args.input / "x.csv", delimiter=","))
    y = np.atleast_1d(np.loadtxt(args.input / "y.csv", delimiter=","))
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    dy = float(np.median(np.diff(y))) if y.size > 1 else 1.0
    origin = (float(x[0] - dx / 2), float(y[0] - dy / 2), 0.0)

    entries = []
    final_step = frame_ids[-1]
    for output_index, frame_id in enumerate(frame_ids):
        solid_fraction = np.load(sources["nu"][frame_id])
        grain_size = np.load(sources["s_bar"][frame_id]) * 1_000.0
        particle_u = np.load(sources["u"][frame_id])
        particle_v = np.load(sources["v"][frame_id])
        fluid = np.load(sources["fluid"][frame_id])
        fluid_u = fluid["u"]
        fluid_v = fluid["v"]
        occupied = solid_fraction > 0.0

        particle_speed = np.where(occupied, np.hypot(particle_u, particle_v), np.nan)
        particle_velocity = np.stack((particle_u, particle_v, np.zeros_like(particle_u)), axis=-1)
        fluid_speed = np.hypot(fluid_u, fluid_v)
        fluid_velocity = np.stack((fluid_u, fluid_v, np.zeros_like(fluid_u)), axis=-1)

        nx, ny = solid_fraction.shape
        image = vtk.vtkImageData()
        image.SetDimensions(nx + 1, ny + 1, 2)
        image.SetOrigin(*origin)
        image.SetSpacing(dx, dy, min(dx, dy) * 0.08)
        add_cell_array(image, "Solid fraction", solid_fraction)
        add_cell_array(image, "Mean grain size (mm)", np.where(occupied, grain_size, np.nan))
        add_cell_array(image, "Particle speed (m/s)", particle_speed)
        add_cell_array(image, "Particle velocity (m/s)", particle_velocity, components=3)
        add_cell_array(image, "Fluid speed (m/s)", fluid_speed)
        add_cell_array(image, "Fluid velocity (m/s)", fluid_velocity, components=3)
        add_cell_array(image, "Fluid vertical velocity (m/s)", fluid_v)
        add_cell_array(image, "Fluid pressure (Pa)", fluid["pressure"])
        add_cell_array(image, "Momentum exchange beta", fluid["beta"])
        add_cell_array(image, "Fluidized", fluid["fluidized"].astype(np.float32))
        image.GetCellData().SetActiveScalars("Solid fraction")
        image.GetCellData().SetActiveVectors("Fluid velocity (m/s)")

        filename = f"{args.name}_{output_index:04d}.vti"
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(str(args.output / filename))
        writer.SetInputData(image)
        writer.SetDataModeToBinary()
        writer.SetCompressorTypeToZLib()
        if writer.Write() != 1:
            raise RuntimeError(f"VTK failed to write {filename}")

        time_value = args.t_final * frame_id / final_step if final_step else 0.0
        entries.append((time_value, filename))

    pvd_path = args.output / f"{args.name}.pvd"
    write_pvd(pvd_path, entries)
    print(f"Wrote {len(entries)} frames to {pvd_path}")


if __name__ == "__main__":
    main()
