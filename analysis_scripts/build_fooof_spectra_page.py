#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble reports/attn_explore_14Sep26_fooof_broadband.html, the broadband
successor of the "Spectral Features by Gaze State" companion page.

Sources:
    {MOVIE_DATA}/attn_explore_14Sep26/fooof_spectra/spectra_by_state.json
        average spectral components per state   (attn_explore_14Sep26_fooof_spectra.py)
    {MOVIE_DATA}/attn_explore_14Sep26/fooof_states_broadband/{profile,twostate,trend6,contrasts3}.json
    {MOVIE_DATA}/attn_explore_14Sep26/fooof_states_broadband/presence/{...}.json
        scalar-feature tests                     (attn_explore_14Sep26_fooof_states_broadband.py)
    OLD_PAGE  the light-fit page; its CSS and the bar/heatmap JavaScript are
              reused verbatim, only the embedded data and the prose change.

The page is regenerated, not hand-edited. Run after both producers.
"""
import os, re, json, sys
import numpy as np, pandas as pd
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here); sys.path.insert(0, os.path.join(os.path.dirname(_here), 'src'))
from paths import MOVIE_DATA  # noqa

OLD_PAGE = os.path.join(os.path.dirname(_here), 'reports', 'attn_explore_14Sep26_fooof.html')
OUT_PAGE = os.path.join(os.path.dirname(_here), 'reports', 'attn_explore_14Sep26_fooof_broadband.html')
ST = f'{MOVIE_DATA}/attn_explore_14Sep26/fooof_states_broadband'
SP = f'{MOVIE_DATA}/attn_explore_14Sep26/fooof_spectra/spectra_by_state.json'
REG = ['far external', 'external', 'middle', 'ambiguous', 'internal', 'far internal']


def load_bundle(d):
    prof = json.load(open(f'{d}/profile.json')); tw = json.load(open(f'{d}/twostate.json'))
    t6 = json.load(open(f'{d}/trend6.json')); c3 = json.load(open(f'{d}/contrasts3.json'))
    S = pd.read_csv(f'{d}/profile_by_region.csv')
    # overall = mean of the network means (as the light page did), sem across networks
    prof['overall'] = {}
    for atlas, sa in S.groupby('atlas'):
        prof['overall'][atlas] = {}
        for (vid, band), g in sa.groupby(['video', 'band']):
            gg = g.groupby('region')['mean'].agg(['mean', 'sem']).reindex(REG)
            prof['overall'][atlas][f'{vid}|{band}'] = {'mean': [None if np.isnan(x) else round(float(x), 5) for x in gg['mean']],
                                                       'sem': [None if np.isnan(x) else round(float(x), 5) for x in gg['sem']]}
    # ylim per feature: 90th percentile of |mean|+sem over every network cell, rounded up
    prof['ylim'] = {}
    for band, g in S.groupby('band'):
        v = (g['mean'].abs() + g['sem'].fillna(0)).quantile(0.9)
        prof['ylim'][band] = float(f'{v:.3g}') if band not in ('Has_theta', 'Has_alpha') else 100
    return prof, tw, t6, c3


def js(o):
    return json.dumps(o, separators=(',', ':'))


html = open(OLD_PAGE).read()
head = html[:html.index('<div class="wrap">')]
head = head.replace('<title>Spectral Features by Gaze State</title>', '<title>Spectral Components by Gaze State</title>')
script = html[html.index('<script>'):html.index('</script>') + len('</script>')]
blocks = script.split('(function(){')
assert len(blocks) == 3, len(blocks)
pre, b1, b2 = blocks


FL1 = {'Aperiodic_Exponent': 'exponent', 'Aperiodic_Offset': 'offset', 'Knee_Freq_Hz': 'knee (Hz)', 'Periodic_theta': 'periodic θ',
       'Periodic_alpha': 'periodic α', 'Periodic_beta': 'periodic β', 'Periodic_gamma': 'periodic γ', 'Periodic_hfa': 'periodic HFA'}
FL2 = {'Has_theta': 'theta peak present (%)', 'Has_alpha': 'alpha peak present (%)'}


def swap_data(block, prof, tw, t6, c3, fl):
    """The old page defines PF, TW, T6, C3 and FL in one const statement on one line."""
    i = block.index('const PF=')
    j = block.index('\n', i)
    return block[:i] + f'const PF={js(prof)},TW={js(tw)},T6={js(t6)},C3={js(c3)},FL={js(fl)};' + block[j:]


P1, TW1, T61, C31 = load_bundle(ST)
P2, TW2, T62, C32 = load_bundle(f'{ST}/presence')
b1 = swap_data(b1, P1, TW1, T61, C31, FL1)
b2 = swap_data(b2, P2, TW2, T62, C32, FL2)
# add HFA to the feature lists of block 1
b1 = b1.replace("'Periodic_gamma']]", "'Periodic_gamma','Periodic_hfa']]")
b1 = b1.replace("Periodic_gamma:'Periodic gamma'}", "Periodic_gamma:'Periodic gamma',Periodic_hfa:'Periodic HFA'}")
assert b1.count('Periodic_hfa') >= 3

# ---------------------------------------------------------------- sections
sec1_start = html.index('<section class="entry" id="entry-2026-09-15-fooof">')
sec2_start = html.index('<section class="entry" id="entry-presence">')
sec_end = html.index('<div class="tip" id="tip" hidden>')
sec1 = html[sec1_start:sec2_start]; sec2 = html[sec2_start:sec_end]

n_rec = {'despicable_me_english': 16, 'despicable_me_hungarian': 13, 'inscapes': 13}
intro1 = '''<div class="prose">
    <p>The same states and tests as the <a href="https://claude.ai/code/artifact/729927dd-8efb-4a17-bcfe-88d49e1028ad">light-fit page</a>, recomputed from the broadband fits: knee-mode FOOOF over 1 to 150 Hz on the same 10 s window grid, with the notch-filter dips interpolated out of each window's spectrum before the fit (<code>extract_fooof_broadband_interp.py</code>, peak cap 8). Periodic HFA (57 to 150 Hz) is added to the periodic features. All 42 recordings of the neural inclusion list now have fits (16 English, 13 Hungarian, 13 Inscapes; NS205 English was missing before). Values are contact-relative as before.</p>
    <p>Two things to keep in mind when comparing with the light page. The broadband exponent is not the light exponent: across the joined channel-windows they correlate at 0.64, and the broadband one is lower by 0.38 on average, because a single knee model over 1 to 150 Hz measures a compromise slope where the light fit measured the slope below 57 Hz. And periodic gamma is now measured mid-range rather than at the top edge of the fit, so it is a different quantity too (r 0.25 with the light one). Theta, alpha and beta periodic power carry over (r about 0.8).</p>
  </div>'''
sec1 = re.sub(r'<div class="prose">.*?</div>', intro1, sec1, count=1, flags=re.S)
sec1 = sec1.replace('<h2>Aperiodic and periodic spectral features by gaze state</h2>', '<h2>Scalar FOOOF features by gaze state, broadband fits</h2>')
sec1 = re.sub(r'<figcaption>.*?</figcaption>', '<figcaption>Network-averaged region means of the contact-relative features; y scale set per feature from the spread of the network cells.</figcaption>', sec1, count=1, flags=re.S)
note1 = '''<p class="note" style="margin-top:18px">Reading, broadband fits. Periodic alpha rises toward the internal pole wherever its trend is significant (Y17: 13 rising, 0 falling; Y7: 7 and 0) and periodic beta does the same (6 and 0), as on the light page. Periodic gamma now falls toward the internal pole (Y17: 0 rising, 12 falling; Y7: 0 and 6); on the light page gamma was near zero, but that gamma was measured at the edge of the fit. The aperiodic picture has changed with the wider fit: the exponent trend is now mostly negative (1 rising, 4 falling on Y17) where the light fit had it rising in 8 networks, and the knee frequency falls toward the internal pole in every significant network (0 and 5). The two-state contrast follows the trends for alpha and gamma (alpha: 13 significant Y17 cells, gamma: 12) and is sparse for the aperiodic parameters. The curves in the first section show what these summaries are averaging over.</p>'''
sec1 = re.sub(r'<p class="note" style="margin-top:18px">.*?</p>', note1, sec1, count=1, flags=re.S)
sec1 = sec1.replace('id="entry-2026-09-15-fooof"', 'id="entry-scalar"')

sec2 = sec2.replace('15 English, 13 Hungarian, 13 Inscapes', '16 English, 13 Hungarian, 13 Inscapes')
sec2 = re.sub(r'<figcaption>.*?</figcaption>', '<figcaption>Peak presence stays almost flat across the axis on the broadband fits too: alpha peaks in about 91 to 92% of windows and theta in 70 to 73%, in every state. As before, the states differ in how much periodic power a peak carries, not in whether one is fitted.</figcaption>', sec2, count=1, flags=re.S)
sec2 = sec2.replace('<h2>Theta and alpha peak presence, percent of windows</h2>', '<h2>Theta and alpha peak presence, percent of windows, broadband fits</h2>')

# ---------------------------------------------------------------- new section: average components
SPJ = json.load(open(SP))
sec0 = '''<section class="entry" id="entry-components">
  <div class="entry-head">
    <span class="entry-date">2026-09-15</span>
    <h2>Average spectral components by gaze state</h2>
  </div>
  <div class="prose">
    <p>Instead of a scalar per feature, these figures show the curves the broadband FOOOF fit decomposes each window into, averaged over the windows of a state. For every kept 10 s window of every contact: the <strong>total</strong> spectrum is the log10 Welch power with the notch dips interpolated, exactly as fitted; the <strong>aperiodic</strong> component is rebuilt from that window's stored offset, knee and exponent (offset minus log10 of knee plus f to the exponent); the <strong>periodic</strong> component is their difference, the flattened spectrum whose band means are the Periodic features below. Each curve is averaged over a state's windows within a contact (at least 3 windows), then across contacts. Curves are shown absolute, and relative to the contact's mean over all its kept windows, which is the convention the scalar features use.</p>
    <p>States are those of the log: six PC1 regions with the deviation gate, the three collapsed groups, and the two-state conj 0.6 label. Bands are the standard error across contacts; no test is computed on the curves themselves, the tests are on the scalar features in the next section. Frequency axis is logarithmic, 1 to 150 Hz, and so are the bins: 20 per decade, each contact's curve binned before averaging, so a bin at 100 Hz averages about ten times as many 0.5 Hz Welch bins as one at 10 Hz. The Welch estimation noise is the same size at every frequency (about 0.002 log10 after averaging over windows and contacts) while the state effect above 30 Hz is only about 0.005, so on a linear 1 Hz grid the high band looked like noise.</p>
  </div>
  <div class="ctl" style="display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:8px 0 4px;font-family:var(--sans);font-size:14px">
    <label>Atlas <select id="sp-atlas"></select></label>
    <label>Network <select id="sp-net"></select></label>
    <span id="sp-n" style="color:var(--ink-3);font-family:var(--mono);font-size:12px"></span>
  </div>
  <figure id="fig-sp-abs">
    <p class="fig-title">Absolute components, six states</p>
    <p class="fig-sub">Rows: total, aperiodic, periodic. Columns: films. Six curves per panel, far external to far internal; on this scale the states nearly coincide, which is the point of the relative view below.</p>
    <div class="legend" id="sp-legend6"></div>
    <div id="sp-abs"></div>
  </figure>
  <figure id="fig-sp-rel">
    <p class="fig-title">State minus contact mean, six states</p>
    <p class="fig-sub">Same panels, each contact's all-window mean curve subtracted before averaging. Shaded band: SEM across contacts.</p>
    <div id="sp-rel"></div>
  </figure>
  <figure id="fig-sp-three">
    <p class="fig-title">State minus contact mean, three states</p>
    <div class="legend" id="sp-legend3"></div>
    <div id="sp-three"></div>
  </figure>
  <figure id="fig-sp-two">
    <p class="fig-title">Two-state label (conj 0.6): Internal and External, hypothesis networks</p>
    <p class="fig-sub">Rows: all contacts, then the hypothesis networks of the selected atlas. Columns: films. Component: <select id="sp-comp"><option value="2">periodic</option><option value="1">aperiodic</option><option value="0">total</option></select></p>
    <div class="legend" id="sp-legend2"></div>
    <div id="sp-two"></div>
  </figure>
  <figcaption id="sp-caption"></figcaption>
</section>
'''
sec0_js = r'''
(function(){
  const J=__SPJ__;
  const VN={despicable_me_english:'Despicable Me, English',despicable_me_hungarian:'Despicable Me, Hungarian',inscapes:'Inscapes'};
  const HYP={Y7:['Dorsal Attention Network (DAN)','Default Mode Network (DMN)','Visual Network (VN)'],Y17:['Dorsal Attention A','Default A','Visual Central (Visual A)','Visual Peripheral (Visual B)']};
  const dark=()=>document.documentElement.getAttribute('data-theme')==='dark'||(!document.documentElement.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);
  const RC=['#9a6d00','#d5a531','#b8bdb9','#8e9491','#6f5bcc','#3f2f96'], RCd=['#c98f10','#dbb455','#5a615e','#7f8886','#7c6ad4','#5a4bbd'];
  const cols6=()=>dark()?RCd:RC; const C3c=()=>dark()?['#c98f10','#7f8886','#7c6ad4']:['#9a6d00','#8e9491','#5546b8']; const C2c=()=>dark()?['#7c6ad4','#c98f10']:['#5546b8','#9a6d00'];
  const CN=['total (log10 power)','aperiodic (log10)','periodic (log10, flattened)'];
  const short=r=>r.replace(' Network','').replace('Visual Central (Visual A)','Vis Central').replace('Visual Peripheral (Visual B)','Vis Periph').replace('Dorsal Attention','DorsAttn').replace(/\s*\(.*?\)/g,'');
  const F=J.freqs, lx=F.map(Math.log10), x0=lx[0], x1=lx[lx.length-1];
  function chart(series,opts){
    const W=340,L=46,R=8,T=14,B=26,pw=W-L-R,ph=opts.h||120,H=T+ph+B;
    let lo=Infinity,hi=-Infinity; series.forEach(s=>{s.y.forEach((v,i)=>{const e=s.sem?s.sem[i]:0; if(v-e<lo)lo=v-e; if(v+e>hi)hi=v+e;});});
    if(opts.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);} const pad=(hi-lo)*0.08||0.01; lo-=pad;hi+=pad;
    const sx=v=>L+(v-x0)/(x1-x0)*pw, sy=v=>T+(hi-v)/(hi-lo)*ph;
    let s=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;overflow:visible">`;
    if(opts.tag) s+=`<text class="grp" x="${W-R}" y="9" text-anchor="end">${opts.tag}</text>`;
    [1,2,5,10,20,50,100].forEach(f=>{const x=sx(Math.log10(f)); s+=`<line class="grid" x1="${x}" x2="${x}" y1="${T}" y2="${T+ph}"/><text class="tick" x="${x}" y="${T+ph+12}" text-anchor="middle">${f}</text>`;});
    const nt=4; for(let k=0;k<=nt;k++){const v=lo+pad+(hi-lo-2*pad)*k/nt; const y=sy(v); s+=`<line class="grid" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><text class="tick" x="${L-4}" y="${y+3.5}" text-anchor="end">${v.toFixed(Math.abs(hi-lo)<0.1?3:2)}</text>`;}
    if(opts.zero) s+=`<line class="zero" x1="${L}" x2="${W-R}" y1="${sy(0)}" y2="${sy(0)}"/>`;
    series.forEach(sr=>{ if(sr.sem){const up=sr.y.map((v,i)=>`${sx(lx[i]).toFixed(1)},${sy(v+sr.sem[i]).toFixed(1)}`); const dn=sr.y.map((v,i)=>`${sx(lx[i]).toFixed(1)},${sy(v-sr.sem[i]).toFixed(1)}`).reverse(); s+=`<polygon points="${up.concat(dn).join(' ')}" fill="${sr.c}" fill-opacity="0.13" stroke="none"/>`;} });
    series.forEach(sr=>{s+=`<polyline points="${sr.y.map((v,i)=>`${sx(lx[i]).toFixed(1)},${sy(v).toFixed(1)}`).join(' ')}" fill="none" stroke="${sr.c}" stroke-width="${sr.w||1.6}" stroke-linejoin="round"><title>${sr.label}</title></polyline>`;});
    s+=`<text class="tick" x="${L+pw/2}" y="${T+ph+24}" text-anchor="middle">Hz</text></svg>`; return s;
  }
  const row3=inner=>`<div class="row3" style="display:grid;grid-template-columns:repeat(3,minmax(0,340px));gap:8px 14px">${inner}</div>`;
  const selA=document.getElementById('sp-atlas'), selN=document.getElementById('sp-net'), selC=document.getElementById('sp-comp');
  Object.keys(J.atlases).forEach(a=>{const o=document.createElement('option');o.value=a;o.textContent=a;selA.appendChild(o);});
  function fillNets(){selN.innerHTML='';J.atlases[selA.value].networks.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n==='all'?'all contacts':n;selN.appendChild(o);});}
  document.getElementById('sp-legend6').innerHTML=J.regions.map((r,i)=>`<span><i style="background:${cols6()[i]}"></i>${r}</span>`).join('');
  document.getElementById('sp-legend3').innerHTML=['External','Middle','Internal'].map((g,i)=>`<span><i style="background:${C3c()[i]}"></i>${g}</span>`).join('');
  document.getElementById('sp-legend2').innerHTML=['Internal','External'].map((g,i)=>`<span><i style="background:${C2c()[i]}"></i>${g}</span>`).join('')+'<span>band = SEM across contacts</span>';
  function get(a,vid,net,st){return J.atlases[a].data[`${vid}|${net}|${st}`];}
  function block(kind,states,colors,host,zero){
    const a=selA.value,net=selN.value; let h='';
    for(let c=0;c<3;c++){ h+=`<p class="fgroup">${CN[c]}</p>`+row3(J.videos.map(vid=>{const ser=[];states.forEach((st,i)=>{const d=get(a,vid,net,st); if(!d) return; ser.push({y:d[kind][c],sem:kind==='rel'?d.rel_sem[c]:null,c:colors[i],label:`${st}, n=${d.n} contacts, ${d.np} people`});}); return `<div class="panel dev">${ser.length?chart(ser,{zero,tag:VN[vid].replace('Despicable Me, ','DM ')}):'<p class="meta">no data</p>'}</div>`;}).join('')); }
    document.getElementById(host).innerHTML=h;
  }
  function two(){
    const a=selA.value,c=+selC.value; const nets=['all'].concat(HYP[a].filter(n=>J.atlases[a].networks.includes(n))); let h='';
    nets.forEach(net=>{h+=`<div style="margin-top:8px"><p class="meta" style="margin:0 0 2px;font-weight:500;color:var(--ink)">${net==='all'?'all contacts':net}</p>`+row3(J.videos.map(vid=>{const ser=[];[['Internal2','Internal'],['External2','External']].forEach(([st,lab],i)=>{const d=get(a,vid,net,st); if(!d) return; ser.push({y:d.rel[c],sem:d.rel_sem[c],c:C2c()[i],label:`${lab}, n=${d.n}`});}); return `<div class="panel dev">${ser.length?chart(ser,{zero:true,tag:VN[vid].replace('Despicable Me, ','DM ')}):'<p class="meta">no data</p>'}</div>`;}).join(''))+'</div>';});
    document.getElementById('sp-two').innerHTML=h;
  }
  function render(){
    const a=selA.value,net=selN.value; const d=get(a,J.videos[0],net,'all');
    document.getElementById('sp-n').textContent=J.videos.map(v=>{const q=get(a,v,net,'all');return q?`${VN[v].replace('Despicable Me, ','DM ')}: ${q.n} contacts / ${q.np} people`:'';}).join('   ');
    block('abs',J.regions,cols6(),'sp-abs',false); block('rel',J.regions,cols6(),'sp-rel',true); block('rel',['External','Middle','Internal'],C3c(),'sp-three',true); two();
  }
  selA.addEventListener('change',()=>{fillNets();render();}); selN.addEventListener('change',render); selC.addEventListener('change',two);
  fillNets(); render();
  if(matchMedia) matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
})();
'''.replace('__SPJ__', js(SPJ))

page = (head + '<div class="wrap">' + html[html.index('<div class="wrap">') + len('<div class="wrap">'):sec1_start]
        .replace('<h1>Spectral Features by Gaze State</h1>', '<h1>Spectral Components by Gaze State</h1>')
        .replace('attn_explore_14Sep26 · companion page', 'attn_explore_14Sep26 · companion page · broadband refit')
        .replace('<p>FOOOF aperiodic and periodic features across the gaze states defined in the',
                 '<p>The FOOOF decomposition by gaze state, recomputed from the broadband (1 to 150 Hz, notch-interpolated) fits, and shown as the average spectral components themselves wherever a curve exists behind a number. Gaze states are those defined in the')
        + sec0 + sec1 + sec2
        + html[sec_end:html.index('<script>')]
        + pre + '(function(){' + b1 + '(function(){' + b2.replace('</script>', '') + '<' + '/script><script>' + sec0_js + '</script>'
        + html[html.index('</script>') + len('</script>'):])
open(OUT_PAGE, 'w').write(page)
print(OUT_PAGE, f'{len(page)/1e6:.1f} MB')
