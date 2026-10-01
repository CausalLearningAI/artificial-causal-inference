# ECI concept atlas (published page snapshots)

The published page lives at https://claude.ai/artifact/JfJnP4CqjYWoprEq9dM8Qa.
After each publish this folder gets the page that was published (`index.html`),
the asset-store map of its clip packs (`asset_map.json`) and a `manifest.json`
(artifact version, build command, code commit, version file list).
Data files and clips are not tracked: `scripts/eci/build_explorer.sh` rebuilds them from `results/`.
