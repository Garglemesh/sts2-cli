# Game Version Compatibility

This project runs the **real Slay the Spire 2 engine** (`sts2.dll`) headless by calling
into the game's own internal APIs (`RunState.CreateForTest`, `RunManager`,
`CombatManager`, `ICardSelector`, …). Those APIs are **not a stable public interface** —
MegaCrit changes them freely between patches. So the thing most likely to break this fork
is not your OS; it's a **game update**.

## Known-good version

| | |
|---|---|
| **Verified game version** | `v0.107.1` (commit `59260271`, 2026-06-18) |
| **Platform verified** | Linux (`data_sts2_linuxbsd_x86_64`), .NET 9 SDK |
| **Regression** | 5/5 full runs complete for all 5 characters |

`setup.sh` reads `release_info.json` from your Steam install and warns if your installed
version differs from this. The warning is **non-blocking** — a newer version may work
unchanged, or may need a few source fixes (below).

### Pinning the version (optional)

Steam auto-updates by default, which can silently break the build. To stay on a known-good
build, in Steam: right-click **Slay the Spire 2 → Properties → Updates → "Only update this
game when I launch it"**, and/or use a Steam beta branch if MegaCrit exposes one.

## Recognizing a version-skew break

When the installed game has drifted from what the source expects, you'll see one of two
failure shapes:

1. **Build errors** — `error CS1061: 'X' does not contain a definition for 'Y'`,
   `error CS1501: No overload for method 'Z' takes N arguments`, `CS0738` (interface
   return-type mismatch). The C# wrapper is calling a game member that was renamed,
   removed, or re-signatured.

2. **Runtime errors** — the build succeeds but `start_run` fails, often with a gate the GUI
   normally satisfies. Example from `v0.107.1`:
   `InvalidOperationException: ModManager is not finished initializing!`

Both mean the same thing: **reconcile the wrapper against the installed DLL.**

## The fix workflow (decompile-and-reconcile)

The wrapper code lives in `src/Sts2Headless/` (mainly `RunSimulator.cs`). To find what a
broken symbol became, inspect the actual installed `sts2.dll` with
[Mono.Cecil](https://github.com/jbevain/cecil) (no game source needed). Minimal recipe:

```bash
mkdir /tmp/dumpapi && cd /tmp/dumpapi
cat > dump.csproj <<'EOF'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net9.0</TargetFramework><Nullable>disable</Nullable></PropertyGroup>
  <ItemGroup><PackageReference Include="Mono.Cecil" Version="0.11.6" /></ItemGroup>
</Project>
EOF
cat > Program.cs <<'EOF'
using System; using System.Linq; using Mono.Cecil;
var m = ModuleDefinition.ReadModule(args[0], new ReaderParameters{ ReadingMode = ReadingMode.Deferred });
var want = args.Skip(1).ToHashSet();
foreach (var t in m.GetTypes().Where(t => want.Contains(t.Name))) {
    Console.WriteLine($"\n== {t.FullName} (base={t.BaseType?.Name}) ==");
    foreach (var me in t.Methods.Where(x => !x.IsConstructor).OrderBy(x => x.Name))
        Console.WriteLine($"  M {me.Name}({string.Join(", ", me.Parameters.Select(p => p.ParameterType.Name+" "+p.Name))}) : {me.ReturnType.Name}");
    foreach (var p in t.Properties) Console.WriteLine($"  P {p.Name} : {p.PropertyType.Name}");
    foreach (var f in t.Fields) Console.WriteLine($"  F {(f.IsStatic?\"static \":\"\")}{(f.IsPublic?\"pub \":\"priv \")}{f.Name} : {f.FieldType.Name}");
}
EOF
# Dump the type(s) named in the build error, e.g. MerchantRoom Reward CardRewardSelection
dotnet run -- "/path/to/Steam/.../data_sts2_linuxbsd_x86_64/sts2.dll" MerchantRoom Reward
```

Then update `RunSimulator.cs` to match the new names/signatures. The pattern is always:
read the error → dump the type → rename/re-sign the call → rebuild.

### History of fixes (examples of the churn)

- `GetSelectedCardReward` return type flip-flopped: `CardModel` → `CardRewardSelection`
  struct → `CardModel` → `CardRewardSelection` (current). This is why the wrapper has a
  comment block documenting which build used which.
- `SetUpSavedSinglePlayer` → `SetUpSavedSingleplayer` (casing).
- `v0.107.1` added the `ModManager` init gate before `ReflectionHelper.ModTypes`; the
  wrapper now satisfies it reflectively in `EnsureModManagerInitialized()`.

## Verifying after a fix

Always run the full regression (the bar is **completion**, not winning — the batch tool
drives a random agent):

```bash
for char in Ironclad Silent Defect Regent Necrobinder; do
    python3 python/play_full_run.py 5 "$char" 2>&1 | grep -E "Wins|Completed"
done
# Expect: Completed: 5/5 for every character.
```

A single character at `4/5` usually points at one card/relic/event that misbehaves in
headless — check the failing run's `logs/*.jsonl` (the last line shows where it died).
