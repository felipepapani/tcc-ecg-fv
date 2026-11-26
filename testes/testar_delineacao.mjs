// Confere a delineação da página (docs/index.html) com a do TCC (tcc_lib.py), versões 1 e 2, ponto a ponto.
// Uso:  python testes/gerar_delineacao.py && node testes/testar_delineacao.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const aqui = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(aqui, '..', 'docs', 'index.html'), 'utf8');
const marca = '<script id="alg-src">', i0 = html.indexOf(marca) + marca.length;
const ALG = new Function(html.slice(i0, html.indexOf('</script>', i0)) + '\nreturn ALG;')();
const casos = JSON.parse(readFileSync(join(aqui, 'saida', 'delineacao.json'), 'utf8'));

const PONTOS = ['R', 'QRSon', 'QRSfim', 'Ton', 'Tpico', 'Tfim', 'Ton_faixa', 'Tfim_faixa'];   // em amostras: têm de ser iguais
const VALORES = ['amp_QRS', 'amp_T', 'T_prom'];                                                // contínuos: tolerância numérica
let falhas = 0;
for (const c of casos) {
  const chs = c.sinal.map((a) => Float64Array.from(a));
  const prep = ALG.prepararParaDelinear(chs, c.fs);
  let errPrep = 0; c.prep.forEach((a, k) => a.forEach((v, i) => { errPrep = Math.max(errPrep, Math.abs(prep[k][i] - v)); }));
  for (const v of [1, 2]) {
    const ref = c['v' + v], js = ALG.delinear(prep, c.fs, c.picos, 0.20, 0.05, v);
    const prob = [];
    if (js.length !== ref.R.length) prob.push(`${js.length} batimentos ≠ ${ref.R.length}`);
    let difPontos = 0, errVal = 0, difConf = 0;
    js.forEach((b, k) => {
      PONTOS.forEach((p) => { const r = ref[p][k], j = Number.isFinite(b[p]) ? b[p] : null; if (r !== j) difPontos++; });
      VALORES.forEach((p) => { if (!ref[p].length) return; const r = ref[p][k], j = Number.isFinite(b[p]) ? b[p] : null; if ((r === null) !== (j === null)) difPontos++; else if (r !== null) errVal = Math.max(errVal, Math.abs(r - j)); });
      if (v === 2 && ref.T_confiavel[k] !== b.T_confiavel) difConf++;
    });
    if (difPontos) prob.push(`${difPontos} pontos diferentes`);
    if (difConf) prob.push(`${difConf} marcas de confiança diferentes`);
    if (!(errVal < 1e-6)) prob.push(`erro em amplitude ${errVal.toExponential(1)}`);
    if (!(errPrep < 1e-6)) prob.push(`sinal preparado difere em ${errPrep.toExponential(1)} mV`);
    const conf = v === 2 ? `, ${js.filter((b) => b.T_confiavel).length} com T confiável` : '';
    if (prob.length) falhas++;
    console.log(`${prob.length ? 'FALHA' : 'ok   '}  ${c.nome.padEnd(22)} versão ${v}: ${prob.length ? prob.join('; ') : `${js.length} batimentos idênticos ao Python${conf}`}`);
  }
}
console.log(falhas ? `\n${falhas} falha(s)` : '\nDelineação idêntica à do TCC em todos os casos');
process.exit(falhas ? 1 : 0);
