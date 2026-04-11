; Instalador Windows (Inno Setup 6+)
; Compilar: ISCC.exe installer\ClipMaster.iss
; Ou: & "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" installer\ClipMaster.iss

#define MyAppName "ClipMaster"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "ClipMaster"
#define MyAppExeName "ClipMaster.exe"

[Setup]
AppId={{E4C8B2A1-9F3D-4E6C-8B7A-1D2E3F4A5B6C}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Instalação por utilizador (sem pedir administrador)
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist_installer
OutputBaseFilename=ClipMaster_Setup_{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
SetupLogging=yes
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "pt"; MessagesFile: "compiler:Languages\Portuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho no Ambiente de trabalho"; GroupDescription: "Atalhos:"; Flags: unchecked

[Files]
Source: "..\dist\ClipMaster\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Executar {#MyAppName}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; Evita ficheiros antigos de builds anteriores misturados com a nova pasta _internal
Type: filesandordirs; Name: "{app}\_internal"
