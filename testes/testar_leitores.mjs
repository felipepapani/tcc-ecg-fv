// Testa os leitores de arquivo da página (docs/index.html) contra os casos de gerar_casos.py.
// Uso:  python testes/gerar_casos.py && node testes/testar_leitores.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const aqui = dirname(fileURLToPath(import.meta.url));
const saida = join(aqui, 'saida');
const html = readFileSync(join(aqui, '..', 'docs', 'index.html'), 'utf8');

// Extrai a seção de leitura de arquivos do script da página, sem copiar código.
const ini = html.indexOf('// ---------- leitura de arquivos ----------');
const fim = html.indexOf('// ---------- exemplo sintético ----------');
if (ini < 0 || fim < 0) throw new Error('Marcadores da seção de leitura não encontrados em docs/index.html');
const mediana = (a) => { const b = a.filter(Number.isFinite).sort((x, y) => x - y); const m = b.length >> 1; return b.length ? (b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2) : NaN; };
const L = new Function('mediana', html.slice(ini, fim) + '\nreturn { lerHea, lerDat, lerCsv, lerEdf, metadadosCsv };')(mediana);

const FATOR = { mV: 1, uV: 0.001, V: 1000 };
const buf = (nome) => { const b = readFileSync(join(saida, nome)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const txt = (nome) => readFileSync(join(saida, nome), 'utf8');

function ler(caso) {
  const arq = caso.arquivos;
  if (arq.some((a) => /\.hea$/i.test(a))) {
    const cab = L.lerHea(txt(arq.find((a) => /\.hea$/i.test(a))));
    const bufs = {}; arq.filter((a) => /\.dat$/i.test(a)).forEach((a) => { bufs[a.toLowerCase()] = buf(a); });
    return L.lerDat(cab, bufs);
  }
  if (arq.some((a) => /\.(edf|bdf)$/i.test(a))) return L.lerEdf(buf(arq[0]), arq[0]);
  const t = txt(arq[0]), o = caso.opcoes;
  let fs = o.fs, unid = o.unid;
  if (unid === 'auto') { const md = L.metadadosCsv(t); if (md.fs) fs = md.fs; unid = md.unid || 'mV'; }
  return L.lerCsv(t, fs, o.tempo, FATOR[unid]);
}

const casos = JSON.parse(txt('casos.json'));
let falhas = 0;
const linha = (ok, nome, msg) => { if (!ok) falhas++; console.log(`${ok ? 'ok   ' : 'FALHA'}  ${nome.padEnd(32)} ${msg}`); };
for (const caso of casos) {
  let r;
  try { r = ler(caso); } catch (e) {
    if (caso.erro) linha(e.message.includes(caso.erro), caso.nome, `recusado: "${e.message}"`);
    else linha(false, caso.nome, `erro inesperado: ${e.message}`);
    continue;
  }
  if (caso.erro) { linha(false, caso.nome, 'deveria ter sido recusado'); continue; }
  const prob = [];
  if (Math.abs(r.fs - caso.fs) > 1e-9) prob.push(`fs ${r.fs} ≠ ${caso.fs}`);
  if (r.chs.length !== caso.sinal.length) prob.push(`${r.chs.length} canais ≠ ${caso.sinal.length}`);
  if (r.n !== caso.n) prob.push(`n ${r.n} ≠ ${caso.n}`);
  if (JSON.stringify(r.nomes) !== JSON.stringify(caso.nomes)) prob.push(`nomes ${JSON.stringify(r.nomes)} ≠ ${JSON.stringify(caso.nomes)}`);
  let erroMax = 0;
  caso.sinal.forEach((esp, k) => { const c = r.chs[k]; if (!c) return; for (let i = 0; i < Math.min(esp.length, c.length); i++) erroMax = Math.max(erroMax, Math.abs(c[i] - esp[i])); });
  if (!(erroMax < 1e-6)) prob.push(`erro máximo ${erroMax.toExponential(2)} mV`);
  linha(!prob.length, caso.nome, prob.length ? prob.join('; ') : `${r.chs.length} canais × ${r.n} amostras a ${r.fs} Hz, erro máximo ${erroMax.toExponential(1)} mV`);
}
console.log(falhas ? `\n${falhas} falha(s) em ${casos.length} casos` : `\nTodos os ${casos.length} casos passaram`);
process.exit(falhas ? 1 : 0);
