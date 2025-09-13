; Inno Setup Script for Opus Spectrum Visualizator

[Setup]
AppName=Opus Spectrum Visualizator
AppVersion=1.1
AppId={{14DBC78D-7245-4D78-B431-1E662A905EFA}}
DefaultDirName={pf}\Opus Spectrum Visualizator
DefaultGroupName=Opus Spectrum Visualizator
OutputBaseFilename=OpusSpectrumVisualizatorInstaller
SetupIconFile=icon.ico
Compression=lzma
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\OpusSpectrumVisualizator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "icon.ico"; DestDir: "{app}"

[Icons]
; Start Menu shortcut
Name: "{group}\Opus Spectrum Visualizator"; Filename: "{app}\OpusSpectrumVisualizator.exe"; IconFilename: "{app}\icon.ico"
; Desktop shortcut
Name: "{autodesktop}\Opus Spectrum Visualizator"; Filename: "{app}\OpusSpectrumVisualizator.exe"; IconFilename: "{app}\icon.ico"

[Run]
Filename: "{app}\OpusSpectrumVisualizator.exe"; Description: "Launch Opus Spectrum Visualizator"; Flags: nowait postinstall skipifsilent
