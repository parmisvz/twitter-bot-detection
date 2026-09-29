// Run after python -m src.target_annotation_pilot; pass bundled node_modules as argv[2].
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require = createRequire(path.resolve(process.argv[2], '__resolver.cjs'));
const {Workbook} = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const out = 'results/versions/target_annotation_pilot_v1';
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
for (const role of ['annotator_A','annotator_B','adjudicator']) {
  const rows = (await fs.readFile(`${out}/${role}/annotation_forms.jsonl`,'utf8')).trim().split(/\r?\n/).map(JSON.parse);
  const columns = Object.keys(rows[0]);
  if (rows.length !== 18 || new Set(rows.map(r=>r.annotation_id)).size !== 18) throw Error('Invalid pilot count');
  if (columns.some(c=>['user_key','stratum','sampling_weight','prior_model_exposure','model_score','pseudo_label','predicted_label'].includes(c))) throw Error('Restricted column');
  const matrix = [columns,...rows.map(r=>columns.map(c=>r[c]))];
  const wb = Workbook.create();
  const sheet = wb.worksheets.add('Pilot');
  const range = sheet.getRangeByIndexes(0,0,matrix.length,columns.length);
  range.values = matrix;
  wb.recalculate();
  const values = range.values.map(row=>row.map(v=>v??''));
  if (JSON.stringify(values)!==JSON.stringify(matrix)) throw Error('Value round-trip changed');
  const csv = '\ufeff'+values.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n')+'\r\n';
  await fs.writeFile(`${out}/${role}/annotation_forms.csv`,csv,{flag:'wx'});
}
const artifacts = {};
async function collect(dir) {
  for (const entry of await fs.readdir(dir,{withFileTypes:true})) {
    const p=path.join(dir,entry.name);
    if (entry.isDirectory()) await collect(p);
    else artifacts[path.relative(out,p).replaceAll('\\','/')] = sha(await fs.readFile(p));
  }
}
await collect(out);
await fs.writeFile(`${out}/manifest.json`,JSON.stringify({version:'target_annotation_pilot_v1',pilot_only:true,selected_n:18,artifacts},null,2)+'\n',{flag:'wx'});
console.log('Created three 18-row UTF-8 CSV forms and package hash manifest.');
