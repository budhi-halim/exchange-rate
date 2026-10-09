# Static publishing

Editable application sources remain in their existing locations. `site-build.json` explicitly lists the files and directories allowed in the public site. The builder produces a separate, ignored `dist` directory; it does not modify application source, stored browser data or scheduled source snapshots.

Run `npm ci --ignore-scripts` in this tool directory, then run `node build.mjs --root <repository>`. Build dependencies are pinned in the lockfile. HTML whitespace/comments, CSS whitespace and JavaScript formatting are minified. JavaScript names and logic, vendor libraries, Python and binary assets are preserved. JSON is serialized without indentation.

Extension ZIPs contain one named folder with the manifest directly inside it. Only listed runtime files are packaged. Diagnostics, review evidence, source exports, tests, developer notes and obsolete prototypes are excluded. Existing unpacked folders stay at their original paths.

`build-info.json` records checksums, byte counts, package versions and the source commit. Generated output is replaced only when it contains this builder's marker. An interrupted build leaves the previous output intact.

The Pages workflow builds from the current default branch on a push or manual dispatch. Repositories with scheduled data fetches also publish after a successful data workflow, using the latest branch contents. This handles bot commits that do not trigger another push workflow. Workflow permissions separate read-only building from Pages deployment. Publication never requires a force push or a generated-output branch.
