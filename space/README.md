---
title: floorplan-parser
emoji: 📐
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: "Read lines from a drawing PDF: vector paths vs pixels"
---

# floorplan-parser showcase

Do you need computer vision to read construction drawings? floorplan-parser pulls line geometry out of a drawing PDF two ways: by reading the vector paths the PDF already stores (PyMuPDF), and by rendering the page to pixels and finding the lines with classic computer vision (OpenCV threshold and Hough lines). Both are scored against the drawing's own geometry.

The pictures are synthetic drawings made for this project; pick a drawing and a method to see what was found (blue), missed (red) and added (orange). The table also lists measured results on ten held-out FloorPlanCAD drawings, as numbers only, including `--chain`, which joins the pieces of each curve. A last panel shows the lines → walls → rooms step on a synthetic plan, with its scores on synthetic plans and on real CubiCasa5K layouts. Everything was computed ahead of time; the page runs no model and no server.

Code (MIT): [github.com/dzakybd/floorplan-parser](https://github.com/dzakybd/floorplan-parser). Write-up: [dzaky.ai.id](https://dzaky.ai.id/en/blog/floorplan-parser-vector-vs-raster).

FloorPlanCAD: Fan et al., "FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting", ICCV 2021, https://floorplancad.github.io/ (annotations CC BY-NC 4.0). No FloorPlanCAD drawing is included in this Space.
