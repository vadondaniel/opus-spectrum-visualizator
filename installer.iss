; Inno Setup Script for Opus Spectrum Visualizator

[Setup]
AppName=Opus Spectrum Visualizator
AppVersion=1.0
DefaultDirName={pf}\Opus Spectrum Visualizator
DefaultGroupName=Opus Spectrum Visualizator
OutputBaseFilename=OpusSpectrumVisualizatorInstaller
SetupIconFile=icon.ico
Compression=lzma
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\main\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name: "{group}\Opus Spectrum Visualizator"; Filename: "{app}\main.exe"
; Desktop shortcut
Name: "{autodesktop}\Opus Spectrum Visualizator"; Filename: "{app}\main.exe"

[Run]
Filename: "{app}\main.exe"; Description: "Launch Opus Spectrum Visualizator"; Flags: nowait postinstall skipifsilent
