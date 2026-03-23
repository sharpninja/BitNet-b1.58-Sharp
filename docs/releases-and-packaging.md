# Releases and packaging

NuGet package versioning is generated with [GitVersion](https://gitversion.net/) through the standard repository-local tool manifest at `.config/dotnet-tools.json`.

## Local version calculation

From the repository root:

```bash
dotnet tool restore
dotnet tool run dotnet-gitversion /output json /showvariable SemVer /nonormalize
```

That command returns the semantic version used for `BitNetSharp.Core` and the `BitNetSharp.App` .NET tool package.

## Release publishing

The Azure DevOps main pipeline at [`/azure-pipelines.yml`](../azure-pipelines.yml) restores, builds, tests, and packs on `main` and pull requests.

When you push a tag that matches `v*` or `V*`, the same pipeline also:

- Packs both NuGet artifacts with the GitVersion-generated semantic version
- Publishes the generated `.nupkg` files as Azure Pipeline artifacts
- Pushes the packages to the NuGet feed configured by the `NuGetFeedUrl` pipeline variable

If the target feed requires an API key, store it as the secret pipeline variable `NuGetApiKey`. For same-organization Azure Artifacts feeds, the pipeline can authenticate with `NuGetAuthenticate@1` and the build identity without an extra key.

This keeps prerelease CI artifacts available for inspection while reserving feed publication for explicit version tags. See [Azure DevOps pipelines](azure-devops-pipelines.md) for the exact variables and cutover notes.
