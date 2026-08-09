#!/usr/bin/env python3
"""Render the official-gallery four-panel HGFD GIF/MP4 from a PVD series."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import imageio_ffmpeg
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

PANELS = (
    (
        "Solid fraction",
        "SOLID FRACTION",
        ((0.03, 0.05, 0.09), (0.45, 0.08, 0.15), (1.0, 0.44, 0.05), (1.0, 0.90, 0.25)),
    ),
    (
        "Particle speed (m/s)",
        "PARTICLE SPEED",
        ((0.02, 0.05, 0.10), (0.02, 0.35, 0.48), (0.05, 0.80, 0.70), (0.95, 0.88, 0.30)),
    ),
    (
        "Fluid pressure (Pa)",
        "FLUID PRESSURE",
        ((0.02, 0.04, 0.08), (0.34, 0.06, 0.20), (0.92, 0.22, 0.08), (1.0, 0.82, 0.20)),
    ),
    (
        "Fluid speed (m/s)",
        "FLUID SPEED",
        ((0.02, 0.05, 0.10), (0.02, 0.35, 0.48), (0.05, 0.80, 0.70), (0.95, 0.88, 0.30)),
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pvd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="HGFD · FLUIDIZED BED")
    parser.add_argument("--ffmpeg", help="ffmpeg executable; defaults to imageio-ffmpeg's bundled binary")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def pvd_entries(path):
    tree = ET.parse(path)
    return [
        (float(node.attrib["timestep"]), path.parent / node.attrib["file"])
        for node in tree.findall(".//DataSet")
    ]


def read_image(path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = vtk.vtkImageData()
    image.ShallowCopy(reader.GetOutput())
    return image


def field_ranges(entries):
    samples = {field: [] for field, _, _ in PANELS}
    stride = max(1, len(entries) // 12)
    for _, path in entries[::stride]:
        image = read_image(path)
        for field in samples:
            values = vtk_to_numpy(image.GetCellData().GetArray(field))
            samples[field].append(values[np.isfinite(values)])

    ranges = {}
    for field, chunks in samples.items():
        values = np.concatenate([chunk for chunk in chunks if chunk.size])
        if field == "Solid fraction":
            ranges[field] = (0.0, max(0.6, float(np.nanmax(values))))
        elif field == "Fluid pressure (Pa)":
            low, high = np.nanpercentile(values, (1.0, 99.0))
            ranges[field] = (float(low), float(high if high > low else low + 1.0))
        else:
            high = float(np.nanpercentile(values, 99.0))
            ranges[field] = (0.0, high if high > 0 else 1.0)
    return ranges


def make_lookup_table(colors, value_range):
    table = vtk.vtkColorTransferFunction()
    table.SetColorSpaceToRGB()
    low, high = value_range
    for index, color in enumerate(colors):
        value = low + (high - low) * index / (len(colors) - 1)
        table.AddRGBPoint(value, *color)
    table.SetNanColor(0.02, 0.035, 0.06)
    return table


def add_text(renderer, value, x, y, size, color=(0.92, 0.95, 1.0)):
    actor = vtk.vtkTextActor()
    actor.SetInput(value)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(x, y)
    actor.GetTextProperty().SetFontFamilyToArial()
    actor.GetTextProperty().SetFontSize(size)
    actor.GetTextProperty().SetBold(True)
    actor.GetTextProperty().SetColor(*color)
    renderer.AddActor2D(actor)
    return actor


def main():
    args = parse_args()
    if args.ffmpeg is None:
        args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = pvd_entries(args.pvd)
    if not entries:
        raise RuntimeError(f"No time steps found in {args.pvd}")
    initial = read_image(entries[0][1])
    ranges = field_ranges(entries)
    bounds = initial.GetBounds()
    width = bounds[1] - bounds[0]
    height = bounds[3] - bounds[2]

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(True)
    window.SetSize(1920, 1080)
    window.SetMultiSamples(8)
    window.SetWindowName(args.title)
    viewports = (
        (0.0, 0.5, 0.5, 1.0),
        (0.5, 0.5, 1.0, 1.0),
        (0.0, 0.0, 0.5, 0.5),
        (0.5, 0.0, 1.0, 0.5),
    )
    interpolators = []
    time_labels = []

    for (field, label, colors), viewport in zip(PANELS, viewports):
        renderer = vtk.vtkRenderer()
        renderer.SetViewport(*viewport)
        renderer.SetBackground(0.018, 0.03, 0.052)
        renderer.SetBackground2(0.055, 0.075, 0.11)
        renderer.GradientBackgroundOn()
        window.AddRenderer(renderer)

        interpolator = vtk.vtkCellDataToPointData()
        interpolator.SetInputData(initial)
        interpolator.PassCellDataOff()
        interpolators.append(interpolator)
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(interpolator.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(field)
        mapper.SetLookupTable(make_lookup_table(colors, ranges[field]))
        mapper.SetScalarRange(*ranges[field])
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        renderer.AddActor(actor)
        camera = renderer.GetActiveCamera()
        center_x = 0.5 * (bounds[0] + bounds[1])
        center_y = 0.5 * (bounds[2] + bounds[3])
        camera.SetFocalPoint(center_x, center_y, 0.0)
        camera.SetPosition(center_x, center_y, max(width, height) * 4.0)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(max(height * 0.57, width * 0.62))

        add_text(renderer, label, 0.035, 0.91, 21, (1.0, 0.58, 0.12))
        time_labels.append(add_text(renderer, "t = 0.000 s", 0.035, 0.055, 15, (0.74, 0.81, 0.90)))
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(mapper.GetLookupTable())
        scalar_bar.SetNumberOfLabels(4)
        scalar_bar.SetOrientationToHorizontal()
        scalar_bar.SetPosition(0.55, 0.055)
        scalar_bar.SetWidth(0.40)
        scalar_bar.SetHeight(0.055)
        scalar_bar.GetLabelTextProperty().SetColor(0.78, 0.84, 0.92)
        scalar_bar.GetLabelTextProperty().SetFontSize(11)
        scalar_bar.SetLabelFormat("%.3g")
        renderer.AddActor2D(scalar_bar)

    first_renderer = window.GetRenderers().GetFirstRenderer()
    add_text(first_renderer, args.title, 0.035, 0.835, 17, (0.78, 0.84, 0.92))
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()

    with tempfile.TemporaryDirectory(prefix="hgfd-render-") as temporary:
        frame_directory = Path(temporary)
        poster_index = round(0.65 * (len(entries) - 1))
        for frame_index, (time_value, path) in enumerate(entries):
            image = read_image(path)
            for interpolator, time_label in zip(interpolators, time_labels):
                interpolator.SetInputData(image)
                time_label.SetInput(f"t = {time_value:.3f} s")
            window.Render()
            capture.Modified()
            capture.Update()
            frame_path = frame_directory / f"frame_{frame_index:04d}.png"
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(frame_path))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
            if frame_index == poster_index:
                (args.output / "poster.png").write_bytes(frame_path.read_bytes())

        subprocess.run(
            [
                args.ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(frame_directory / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(args.output / "demo.mp4"),
            ],
            check=True,
        )
        subprocess.run(
            [
                args.ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(frame_directory / "frame_%04d.png"),
                "-filter_complex",
                "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];"
                "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                str(args.output / "preview.gif"),
            ],
            check=True,
        )
    print(f"Rendered {len(entries)} frames to {args.output}")


if __name__ == "__main__":
    main()
