import { readFile, writeFile, readdir, mkdir, lstat, realpath, mkdtemp, rm, rename } from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { minify as minifyHtml } from 'html-minifier-terser';
import { minify as minifyJs } from 'terser';
import CleanCSS from 'clean-css';
import { zipSync } from 'fflate';

const argument = process.argv.indexOf('--root');
const root = await realpath(argument >= 0 ? process.argv[argument + 1] : process.cwd());
const config = JSON.parse(await readFile(path.join(root, 'site-build.json'), 'utf8'));
const output = path.join(root, 'dist');
const staged = await mkdtemp(path.join(root, '.site-build-'));
const ignored = new Set(['family-manifest.json', 'managed-assets.json']);
const assets = new Map();
const hash = buffer => createHash('sha256').update(buffer).digest('hex');
let sourceBytes = 0;

function safePath(base, relative) {
  if (typeof relative !== 'string' || !relative || relative.split(/[\\/]/).some(part => !part || part.startsWith('.')) || path.isAbsolute(relative)) throw new Error(`Invalid publishing path: ${relative}`);
  const resolved = path.resolve(base, relative);
  if (!resolved.startsWith(`${base}${path.sep}`)) throw new Error('Publishing path escapes its source folder');
  return resolved;
}

async function collect(base, relative, files) {
  const filename = safePath(base, relative);
  const info = await lstat(filename);
  if (info.isSymbolicLink()) throw new Error(`Symlinks are not published: ${relative}`);
  if (info.isDirectory()) {
    for (const name of (await readdir(filename)).sort()) await collect(base, `${relative}/${name}`, files);
  } else {
    if (ignored.has(path.basename(relative))) return;
    if (!/\.(?:html|css|js|json|txt|svg|png|jpg|jpeg|ico|woff2?|py)$/i.test(relative)) throw new Error(`Review unsupported public file: ${relative}`);
    files.set(relative, await readFile(filename));
  }
}

async function optimize(name, content) {
  sourceBytes += content.length;
  const source = content.toString('utf8');
  if (name.endsWith('.html')) {
    return Buffer.from(await minifyHtml(source, { collapseWhitespace: true, conservativeCollapse: true, removeComments: true, minifyCSS: false, minifyJS: false, keepClosingSlash: true }));
  }
  if (name.endsWith('.js') && !name.endsWith('.min.js')) {
    const options = { compress: false, mangle: false, ecma: 2022, format: { comments: /^!|@license|@preserve/i } };
    let result;
    try {
      result = await minifyJs({ [name]: source }, options);
    } catch (error) {
      if (error.name !== 'SyntaxError') throw error;
      // Module workers can use top-level await; classic scripts retain their parsing mode.
      result = await minifyJs({ [name]: source }, { ...options, module: true });
    }
    if (!result.code) throw new Error(`JavaScript produced no output: ${name}`);
    return Buffer.from(result.code);
  }
  if (name.endsWith('.css')) {
    const result = new CleanCSS({ level: 0, rebase: false, inline: ['none'], format: { breaks: { afterAtRule: false } } }).minify(source);
    if (result.errors.length) throw new Error(`${name}: ${result.errors.join('; ')}`);
    return Buffer.from(result.styles);
  }
  if (name.endsWith('.json')) return Buffer.from(JSON.stringify(JSON.parse(source)));
  return content;
}

async function removeGenerated(directory) {
  const resolved = path.resolve(directory);
  if (path.dirname(resolved) !== root || (resolved !== output && !path.basename(resolved).startsWith('.site-build-'))) throw new Error('Refusing to remove a non-build directory');
  const info = await lstat(resolved).catch(error => { if (error.code !== 'ENOENT') throw error; });
  if (!info) return;
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error('Build output must be an ordinary directory');
  if (resolved === output) {
    const previous = JSON.parse(await readFile(path.join(resolved, 'build-info.json'), 'utf8'));
    if (previous.generator !== 'islandsun-site-build') throw new Error('Refusing to replace unrecognized output');
  }
  await rm(resolved, { recursive: true });
}

try {
  if (config.schema !== 1 || !Array.isArray(config.include) || !config.include.includes('index.html')) throw new Error('Invalid site build configuration');
  const files = new Map();
  for (const relative of config.include) await collect(root, relative, files);
  for (const [name, content] of files) assets.set(name, await optimize(name, content));
  const packages = [];
  for (const extension of config.extensions || []) {
    const directory = safePath(root, extension.directory);
    const extensionFiles = new Map();
    for (const relative of extension.include) await collect(directory, relative, extensionFiles);
    const packaged = {};
    for (const [name, content] of extensionFiles) {
      packaged[`${extension.directory}/${name}`] = [await optimize(name, content), { mtime: new Date(2000, 0, 1) }];
    }
    const manifest = JSON.parse(extensionFiles.get('manifest.json').toString('utf8'));
    const name = `downloads/${extension.directory}.zip`;
    assets.set(name, Buffer.from(zipSync(packaged, { level: 9 })));
    packages.push({ file: name, name: manifest.name, version: manifest.version, files: extensionFiles.size });
  }
  assets.set('.nojekyll', Buffer.alloc(0));
  let commit = null;
  try { commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); }
  catch { /* Initial local verification may precede repository creation. */ }
  const publicBytes = [...assets.values()].reduce((sum, value) => sum + value.length, 0);
  const report = { generator: 'islandsun-site-build', schema: 1, commit, sourceBytes, publicBytes, packages, files: Object.fromEntries([...assets].map(([name, value]) => [name, { bytes: value.length, sha256: hash(value) }])) };
  assets.set('build-info.json', Buffer.from(JSON.stringify(report)));
  for (const [name, content] of assets) {
    const filename = path.join(staged, name);
    await mkdir(path.dirname(filename), { recursive: true });
    await writeFile(filename, content);
  }
  await removeGenerated(output);
  await rename(staged, output);
  console.log(`${path.basename(root)}: ${assets.size} public files, ${packages.length} extension packages, ${sourceBytes} source bytes → ${publicBytes} published bytes`);
} catch (error) {
  await removeGenerated(staged);
  throw error;
}
