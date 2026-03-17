# Releases and packaging

NuGet package versioning is generated with [GitVersion](https://gitversion.net/) through the repository-local `dotnet-tools.json` manifest.

## Local version calculation

From the repository root:

```bash
dotnet tool restore
dotnet tool run dotnet-gitversion /output json /showvariable SemVer /nonormalize
```

That command returns the semantic version used for `BitNetSharp.Core` and the `BitNetSharp.App` .NET tool package.

## Release publishing

The build workflow continues to restore, build, test, and pack on `main` and pull requests.

When you push a tag that matches `v*`, the same workflow also:

- Packs both NuGet artifacts with the GitVersion-generated semantic version
- Uploads the generated `.nupkg` files into the matching GitHub release
- Pushes the packages to the repository GitHub Packages feed at `https://nuget.pkg.github.com/<owner>/index.json`, where `<owner>` is the GitHub repository owner (user or organization)

This keeps prerelease CI artifacts available for inspection while reserving release publication for explicit version tags.
