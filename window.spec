# -*- mode: python ; coding: utf-8 -*-
import site
import os

python_lib_path = site.getsitepackages()[0]
deep_sort_path = os.path.join(python_lib_path, 'lib/site-packages/deep_sort_realtime')

a = Analysis(
    ['mainwindow.py'],
    pathex=[],
    binaries=[],
    datas=[('coco.names', '.'), ('yolov8n.pt', '.'), ('settings.json', '.'), (deep_sort_path, 'deep_sort_realtime')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RTSP Stream Viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
