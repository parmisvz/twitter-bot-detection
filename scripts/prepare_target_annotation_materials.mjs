// Usage: bundled-node scripts/prepare_target_annotation_materials.mjs <bundled-node_modules>
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire(path.resolve(process.argv[2], '__resolver.cjs'));
const { Workbook } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const source = 'results/versions/target_audit_v1/seed20260928_n600';
const output = 'results/versions/target_audit_annotation_materials_v1';
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const manifestBytes = await fs.readFile(`${source}/manifest.json`);
const sourceManifest = JSON.parse(manifestBytes);
async function verifySource() {
  if (sha(await fs.readFile(`${source}/manifest.json`)) !== sha(manifestBytes)) throw Error('Source manifest changed');
  for (const [file, hash] of Object.entries(sourceManifest.artifacts)) {
    if (sha(await fs.readFile(`${source}/${file}`)) !== hash) throw Error(`Source hash mismatch: ${file}`);
  }
}
await verifySource();
await fs.mkdir(output); // Fail rather than overwrite a previous materials package.
const hashes = {};
const counts = {};
async function save(file, content) {
  await fs.mkdir(path.dirname(`${output}/${file}`), { recursive: true });
  await fs.writeFile(`${output}/${file}`, content, { flag: 'wx' });
  hashes[file] = sha(await fs.readFile(`${output}/${file}`));
}
const readForms = async role => (await fs.readFile(`${source}/${role}/annotation_forms.jsonl`, 'utf8')).trim().split(/\r?\n/).map(JSON.parse);
const reference = await readForms('annotator_A');
const ids = reference.map(r => r.annotation_id);
if (ids.length !== 600 || new Set(ids).size !== 600) throw Error('Invalid source IDs');
const allowed = Object.keys(reference[0]);
async function writeCSV(file, rows, pilot = false) {
  const columns = Object.keys(rows[0]);
  if (columns.some(k => !allowed.includes(k) && k !== 'adjudication_rationale')) throw Error('Unexpected field');
  if (!pilot && JSON.stringify(rows.map(r => r.annotation_id)) !== JSON.stringify(ids)) throw Error('ID/order mismatch');
  for (const row of rows) {
    if (JSON.stringify(Object.keys(row)) !== JSON.stringify(columns)) throw Error('Schema mismatch');
    if (Object.entries(row).some(([k,v]) => !['annotation_id','rubric_version'].includes(k) && v !== '')) throw Error('Nonblank form');
  }
  const wb = Workbook.create();
  const sheet = wb.worksheets.add('Annotations');
  const matrix = [columns, ...rows.map(r => columns.map(k => r[k]))];
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, columns.length);
  range.values = matrix;
  wb.recalculate();
  const values = range.values.map(row => row.map(v => v ?? ''));
  if (JSON.stringify(values) !== JSON.stringify(matrix)) throw Error('Workbook round-trip changed values');
  // CSV has no formatting or formulas. Quote every field and include a UTF-8 BOM for Excel/Persian.
  const csv = '\ufeff' + values.map(row => row.map(v => '"' + String(v).replaceAll('"', '""') + '"').join(',')).join('\r\n') + '\r\n';
  await save(file, csv);
  counts[file] = rows.length;
}
for (const role of ['annotator_A', 'annotator_B', 'adjudicator']) {
  await writeCSV(`${role}/annotation_forms.csv`, await readForms(role));
}
const pilot = Array.from({length: 10}, (_, i) => ({...reference[0], annotation_id: `PILOT${String(i+1).padStart(3,'0')}`}));
if (pilot.some(r => ids.includes(r.annotation_id))) throw Error('Pilot ID overlap');
await writeCSV('pilot/non_audit_pilot.csv', pilot, true);
for (const role of ['annotator_A','annotator_B','pilot']) {
  await save(`${role}/ANNOTATOR_GUIDE.md`, await fs.readFile('reports/annotation_materials/ANNOTATOR_GUIDE.md'));
}
await save('adjudicator/ANNOTATOR_GUIDE.md', await fs.readFile('reports/annotation_materials/ANNOTATOR_GUIDE.md'));
await save('adjudicator/ADJUDICATOR_GUIDE.md', await fs.readFile('reports/annotation_materials/ADJUDICATOR_GUIDE.md'));
await save('COORDINATOR_README.md', await fs.readFile('reports/annotation_materials/COORDINATOR_README.md'));
await verifySource();
await save('manifest.json', JSON.stringify({
  version: 'target_audit_annotation_materials_v1',
  source_manifest_sha256: sha(manifestBytes),
  row_counts: counts,
  encoding: 'UTF-8 BOM; quoted RFC4180 fields; CRLF',
  checks: { source_hashes_unchanged: true, exact_audit_ids_and_order: true, annotation_fields_blank: true,
    source_schema_preserved: true, workbook_value_roundtrip: true, pilot_accounts_assigned: 0 },
  artifacts: {...hashes},
}, null, 2) + '\n');
console.log(JSON.stringify({output, counts, frozen_source_unchanged: true}));
