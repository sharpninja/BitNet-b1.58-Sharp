# Azure DevOps pipelines

The repository now includes Azure DevOps YAML pipelines that mirror the existing GitHub Actions flows while you cut over CI/CD:

- [`/azure-pipelines.yml`](../azure-pipelines.yml) for the main build, test, pack, and tagged package publication flow
- [`/azure-pipelines-benchmark-report.yml`](../azure-pipelines-benchmark-report.yml) for the slow-lane benchmark report flow and optional static-site deployment

## Main CI pipeline

`azure-pipelines.yml` is intended to replace the GitHub `build.yml` workflow behavior:

- Triggers on pushes to `main`
- Triggers on tags that start with `v` or `V`
- Triggers on pull requests
- Restores the repository-local .NET tools and calculates the package version with GitVersion
- Restores, builds, and tests the solution
- Packs `BitNetSharp.Core` and the `BitNetSharp.App` .NET tool
- Publishes both `.nupkg` outputs as Azure Pipeline artifacts

### Tagged release publication

The `release` stage only runs for refs that match `refs/tags/v*` or `refs/tags/V*`.

To publish packages from tagged runs, configure these Azure Pipeline variables:

- `NuGetFeedUrl`: the target feed URL, for example an Azure Artifacts NuGet v3 feed URL or another NuGet-compatible source
- `NuGetApiKey`: optional secret variable for feeds that require a real API key such as NuGet.org

If you leave `NuGetFeedUrl` empty, tagged builds still produce the `.nupkg` pipeline artifacts but skip the push step.

The pipeline uses `NuGetAuthenticate@1` and falls back to `--api-key AzureArtifacts` when `NuGetApiKey` is not set. That covers same-organization Azure Artifacts feeds without storing an extra secret. If you need to publish to a feed in another Azure DevOps organization, add the appropriate NuGet service connection to the `NuGetAuthenticate@1` task before enabling the release stage.

## Benchmark report pipeline

`azure-pipelines-benchmark-report.yml` is intended to replace the GitHub `benchmark-report.yml` workflow behavior:

- Triggers on pushes to `main` when benchmark-relevant source paths change
- Restores, builds, and runs the `Category=SlowLane` test slice
- Runs `BitNetSharp.App benchmark-report`
- Publishes the generated report directory as a pipeline artifact named `benchmark-report`

### Optional static-site deployment

The `deploy` stage runs only on `main` and only when the `AzureStaticWebAppsApiToken` pipeline variable is present.

To enable deployment:

1. Create an Azure Static Web App for the benchmark report site.
2. Add its deployment token to the pipeline as a secret variable named `AzureStaticWebAppsApiToken`.
3. Point the Azure DevOps pipeline at [`/azure-pipelines-benchmark-report.yml`](../azure-pipelines-benchmark-report.yml).

The deployment step uploads the generated benchmark report artifact directly with `skip_app_build: true`, so the static site serves the already-built `index.html` and companion report assets.

## Cutover note

The legacy GitHub workflow files are still present in `.github/workflows/` so you can validate the Azure DevOps pipelines before disabling GitHub Actions. Once the Azure pipelines are live and validated, remove or disable the GitHub workflows to avoid duplicate CI runs and duplicate package/report publication.
