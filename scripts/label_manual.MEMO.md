# label_manual.py — Memo

## Overview

Interactive OpenCV GUI for drawing/correcting YOLO bounding boxes. Supports Stage 1 (dial bbox) and Stage 2 (digit bboxes in ROI crop).

## Running on Remote Linux from Windows (SSH)

The script requires a display. When connecting from Windows via SSH, use **PuTTY + VcXsrv**:

1. **Install VcXsrv** — https://sourceforge.net/projects/vcxsrv/
2. **Start VcXsrv** (XLaunch) with display `0` and "Disable access control" enabled.
3. **Install PuTTY** — https://www.chiark.greenend.org.uk/~sgtatham/putty/latest
4. **Configure PuTTY**:
   - Connection → SSH → X11: enable "X11 forwarding", set X display location to `localhost:0`
5. Connect via PuTTY and run the script; the GUI appears in VcXsrv.

## Qt Font Warnings

You may see:

```
QFontDatabase: Cannot find font directory .../cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
```

These are harmless; the labeling tool works normally. To suppress them, install DejaVu fonts or configure fontconfig on the server.
