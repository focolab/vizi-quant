# vizi-quant

Region quantification and plotting for [vizi](https://vizi.science), as a
plugin.  Press **Measure** and every region is measured over every frame in
the background, nearest the frame on screen first, into a cache beside the
dataset that later sessions reuse.  The panel shows a table of the regions,
plots their traces (raw brightness F or ΔF/F) and exports two CSV tables,
each with a provenance file.  The first has one row per frame (the frame
index and its time) and, for every region, columns for its voxel count, its
centre and size in X, Y and Z, and the mean, 85–95th and 5–15th percentile
band means and standard deviation of each channel -- the layout of vizi's
built-in Measure export.  The second is tidy: one row per region, frame,
channel and statistic, ΔF/F included.  Nothing is measured until Measure is
pressed.

Needs **vizi 1.0.0 or later** (plugin protocol 1.0).  Requirements are locked
for Linux x86_64, Windows x86_64 and macOS on Apple Silicon, with Python 3.11
or 3.13; it is tested on Linux x86_64.

## Install

In vizi, **Tools → Plugin Manager…** installs vizi-quant: vizi provides a
Python of the version `.python-version` names and runs `scripts/bootstrap.py`
in the plugin's folder.  Then **Tools → Quantify** opens the panel.

By hand, from a release:

1. Download `org.focolab.vizi_quant-<version>.zip` from this repository's
   [Releases](https://github.com/focolab/vizi-quant/releases).
2. In vizi: **Tools → Plugin Manager… → Open plugins folder**, and unzip the
   release there.  You get a folder `org.focolab.vizi_quant/`.
3. In a terminal, in that folder, build the plugin's own environment and
   check it (use `python3.11` if that is what you have):

       python3.13 scripts/bootstrap.py
       .venv/bin/python scripts/doctor.py

   On Windows the second command is `.venv\Scripts\python.exe scripts\doctor.py`.
   The doctor ends with `doctor: ready`.  Bootstrap installs exactly the
   pinned, hash-checked packages in `requirements/locks/` for your system and
   Python; it needs internet access to PyPI and downloads no Python of its
   own.
4. In vizi's Plugin Manager press **Rescan**.  *vizi-quant* is listed, and
   **Tools → Quantify** opens the panel.

To update, delete the folder and repeat with the new release.

## Use

Open a dataset with regions, open Quantify, press **Measure** (it becomes
**Stop** while it runs).  The table has a row per region with the chosen
statistic on the current frame, as raw brightness F and as ΔF/F; click a
header to sort.  **Show** chooses the statistic, the channel and raw
brightness or ΔF/F for the plot.  Select rows (Ctrl or Shift for several) to
draw those regions bold, and click the plot to go to that frame; vizi also
selects the region you picked and frames it once *Control view* is ticked
for vizi-quant in **Tools → Plugin Manager…**.  With ΔF/F shown, the baseline
F0 is set under the plot.  **Export table…** writes one row per frame and
**Export tidy table…** one row per value with ΔF/F, each where you choose,
with a `.provenance.json` beside it.

Measurements are always of the file's own voxels, also while a filter is
applied in vizi.  Centres and sizes are in the file's unit on each axis whose
voxel size the file records (the header says `x (um)`); an axis it does not
record stays in voxels (the header says `z`).  The plot follows vizi's theme.

## Develop

    python3.13 -m venv .venv && .venv/bin/python -m pip install -e '.[test]'
    .venv/bin/python -m pytest                       # unit, parity, emulator
    .venv/bin/python vizi_quant_plugin.py --install  # into vizi's plugins folder
    .venv/bin/python tools/write_manifest.py         # after changing a declaration
    .venv/bin/python tools/make_release.py           # dist/org.focolab.vizi_quant-<version>.zip

The plugin is built on [vizi-plugin-sdk](https://github.com/focolab/vizi-plugin-sdk);
the emulator tests run against the SDK's host emulator, vendored in
`tests/_vendor/`.  `tools/lock.py` regenerates the locks in
`requirements/locks/` from `requirements/cpu.in`.

## License

Apache License 2.0; see `LICENSE` and `NOTICE`.
