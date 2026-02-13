# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

tzdata_datas = collect_data_files('tzdata')
pytz_datas = []
metadata_datas = []
hiddenimports = []

try:
    import pytz  # noqa: F401
except Exception:
    print("WARNING: 'pytz' is not installed in the build environment.")
    print("         Install it with: python -m pip install pytz")
else:
    pytz_datas = collect_data_files('pytz')
    metadata_datas += copy_metadata('pytz')
    hiddenimports.append('pytz')

try:
    metadata_datas += copy_metadata('tzdata')
except Exception:
    pass

a = Analysis(
    ['main.pyw'],
    pathex=[],
    binaries=[],
    datas=[('icon.ico', '.')] + pytz_datas + tzdata_datas + metadata_datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
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
    [],
    exclude_binaries=True,
    name='OpusSpectrumVisualizator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OpusSpectrumVisualizator',
)
