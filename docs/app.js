/* ============================================================
   2026 World Cup Predictor — front-end rendering
   Loads results/snapshots JSON and renders the page.
   ============================================================ */

// Team name -> ISO code for flagcdn.com (supports gb-eng / gb-sct)
const FLAG = {
  "Mexico":"mx","South Africa":"za","South Korea":"kr","Czech Republic":"cz",
  "Canada":"ca","Qatar":"qa","Switzerland":"ch","Bosnia and Herzegovina":"ba",
  "Brazil":"br","Morocco":"ma","Scotland":"gb-sct","Haiti":"ht",
  "United States":"us","Paraguay":"py","Australia":"au","Turkey":"tr",
  "Germany":"de","Ivory Coast":"ci","Ecuador":"ec","Curaçao":"cw",
  "Netherlands":"nl","Sweden":"se","Japan":"jp","Tunisia":"tn",
  "Belgium":"be","Iran":"ir","Egypt":"eg","New Zealand":"nz",
  "Spain":"es","Saudi Arabia":"sa","Uruguay":"uy","Cape Verde":"cv",
  "France":"fr","Senegal":"sn","Iraq":"iq","Norway":"no",
  "Argentina":"ar","Algeria":"dz","Austria":"at","Jordan":"jo",
  "Portugal":"pt","Uzbekistan":"uz","Colombia":"co","DR Congo":"cd",
  "England":"gb-eng","Ghana":"gh","Panama":"pa","Croatia":"hr"
};
const flag = (team, w = 40) => {
  const code = FLAG[team];
  return code
    ? `<img class="flag-img" data-team="${team}" src="https://flagcdn.com/w${w}/${code}.png" alt="${team}" loading="lazy"/>`
    : "";
};
// Clean number formatting: kill float noise, max 2 decimals, drop trailing zeros.
const fmt = v => parseFloat(Number(v).toFixed(2));
const ROUND_LABEL = {
  round_of_16: "Round of 16", quarterfinal: "Quarter-final",
  semifinal: "Semi-final", final: "Final", champion: "Champion"
};

let DATA = null;
let TITLE_MAX = 0;
let RESULTS = [];
let SCHEDULE = [];

init();

async function init() {
  try {
    const res = await fetch("data/snapshot.json");
    DATA = await res.json();
  } catch (e) {
    document.body.innerHTML =
      `<div style="max-width:620px;margin:18vh auto;text-align:center;font-family:sans-serif;color:#e8f5ee;padding:1.5rem">
         <h1 style="color:#2bff88">Run a local server</h1>
         <p>The page loads its data with <code>fetch()</code>, which browsers block on <code>file://</code>.</p>
         <p>From the <b>web/</b> folder run:</p>
         <pre style="background:#11271c;padding:1rem;border-radius:8px;display:inline-block">python3 -m http.server 8000</pre>
         <p>then open <a style="color:#2bff88" href="http://localhost:8000">http://localhost:8000</a></p>
       </div>`;
    return;
  }
  // These two are optional — the page still works if they're missing/empty.
  try { RESULTS = await (await fetch("data/results.json")).json(); } catch (e) { RESULTS = []; }
  try { SCHEDULE = await (await fetch("data/schedule.json")).json(); } catch (e) { SCHEDULE = []; }

  renderTracker();
  renderChampion();
  renderTitleRace();
  renderReachTable();
  renderBracket();
  renderGroups();
  renderFooter();
  setupReveal();
  setupScrollBall();
  setupModal();
}

/* ---------- Tracker: schedule + results + our picks, all in one ---------- */
function renderTracker() {
  const el = document.getElementById("tracker");
  if (!SCHEDULE.length) { el.innerHTML = ""; return; }
  const now = new Date();
  const LA = "America/Los_Angeles";

  // index results + predictions by "home|away"
  const resultOf = {};
  RESULTS.forEach(r => { resultOf[`${r.home}|${r.away}`] = r; });
  const predOf = {};
  DATA.group_predictions.forEach(p => { predOf[`${p.home}|${p.away}`] = p; });

  const pickFor = (m) => {
    const p = predOf[`${m.home}|${m.away}`];
    if (!p) return null;
    const opts = [["home", p.home_win, m.home], ["draw", p.draw, "Draw"], ["away", p.away_win, m.away]];
    opts.sort((a, b) => b[1] - a[1]);
    return { side: opts[0][0], pct: opts[0][1], label: opts[0][2] };
  };

  // running accuracy (only over played matches we have a prediction for)
  let correct = 0, played = 0;
  RESULTS.forEach(r => {
    const pick = pickFor(r);
    if (!pick) return;
    played++;
    const actual = r.hs > r.as ? "home" : r.hs < r.as ? "away" : "draw";
    if (pick.side === actual) correct++;
  });
  const pct = played ? Math.round(correct / played * 100) : 0;
  const summary = played
    ? `<div class="sb-summary"><div class="sb-stat">${correct} / ${played}</div>
         <div class="sb-stat-label">correct so far · ${pct}% hit rate</div></div>`
    : `<div class="sb-summary"><div class="sb-stat-label">no matches played yet — check back soon ⚽</div></div>`;

  // group every match by its Pacific calendar date
  const groups = {};
  SCHEDULE.forEach(m => {
    const dt = new Date(m.kickoff);
    const day = dt.toLocaleDateString("en-CA", { timeZone: LA });
    (groups[day] ??= []).push({ ...m, dt });
  });

  const days = Object.keys(groups).sort().map(day => {
    const label = new Date(day + "T12:00:00").toLocaleDateString("en-US",
      { weekday: "long", month: "short", day: "numeric" });
    const rows = groups[day].sort((a, b) => a.dt - b.dt).map(m => {
      const time = m.dt.toLocaleTimeString("en-US", { timeZone: LA, hour: "numeric", minute: "2-digit" });
      const res = resultOf[`${m.home}|${m.away}`];
      const pick = pickFor(m);

      if (res) {   // played: show score + whether our pick was right
        const actual = res.hs > res.as ? "home" : res.hs < res.as ? "away" : "draw";
        const ok = pick && pick.side === actual;
        return `
          <div class="trk-row ${ok ? "ok" : "miss"}">
            <span class="trk-time">${time}</span>
            <span class="trk-match">${flag(m.home, 40)} <span>${m.home}</span>
              <span class="sc">${res.hs}–${res.as}</span>
              <span>${m.away}</span> ${flag(m.away, 40)}</span>
            <span class="trk-mark ${ok ? "ok" : "miss"}" title="picked ${pick ? pick.label : "?"}">${ok ? "✓" : "✗"}</span>
          </div>`;
      }
      // upcoming (or kicked-off, result not entered yet): show time + our pick
      const live = m.dt < now;
      return `
        <div class="trk-row upcoming ${live ? "live" : ""}">
          <span class="trk-time">${live ? "TBD" : time}</span>
          <span class="trk-match">${flag(m.home, 40)} <span>${m.home}</span>
            <span class="trk-v">v</span>
            <span>${m.away}</span> ${flag(m.away, 40)}</span>
          <span class="trk-pick">${pick ? `pick: ${pick.label} ${fmt(pick.pct)}%` : ""}</span>
        </div>`;
    }).join("");
    return `<div class="sch-day"><div class="sch-daylabel">${label}</div>${rows}</div>`;
  }).join("");

  el.innerHTML = summary + `<div class="sch-scroll">${days}</div>`;
}

/* Click any flag anywhere -> team stats popup */
function setupModal() {
  document.addEventListener("click", (e) => {
    const img = e.target.closest(".flag-img");
    if (img && img.dataset.team) openTeamModal(img.dataset.team);
  });
  document.getElementById("modalClose").addEventListener("click", closeTeamModal);
  document.getElementById("teamModal").addEventListener("click", (e) => {
    if (e.target.id === "teamModal") closeTeamModal();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeTeamModal(); });
}

function openTeamModal(name) {
  const t = DATA.teams.find(x => x.team === name);
  if (!t) return;
  const rank = DATA.teams.findIndex(x => x.team === name) + 1;   // teams sorted by title %
  const rankLabel = rank === 1 ? "#1 favorite" : `#${rank} to win it all`;

  const bar = (label, val, suffix, gold = false) => `
    <div class="stat-row">
      <span class="stat-label">${label}</span>
      <div class="stat-track"><div class="stat-fill ${gold ? "gold" : ""}" style="width:${Math.min(100, val)}%"></div></div>
      <span class="stat-val">${fmt(val)}${suffix}</span>
    </div>`;

  document.getElementById("modalBody").innerHTML = `
    <div class="modal-head">
      ${flag(name, 160)}
      <div>
        <div class="modal-team">${name}</div>
        <div class="modal-sub">ELO ${t.elo} · ${rankLabel}</div>
      </div>
    </div>
    <div class="modal-section-title">Squad Strength</div>
    ${bar("Attack", t.stats.attack, "")}
    ${bar("Defense", t.stats.defense, "")}
    ${bar("Keeper", t.stats.gk, "")}
    ${bar("Depth", t.stats.depth, "")}
    <div class="modal-section-title">Road to the Final</div>
    ${bar("Round of 16", t.reach.round_of_16, "%")}
    ${bar("Quarterfinal", t.reach.quarterfinal, "%")}
    ${bar("Semifinal", t.reach.semifinal, "%")}
    ${bar("Final", t.reach.final, "%")}
    ${bar("Champion 🏆", t.reach.champion, "%", true)}`;

  const modal = document.getElementById("teamModal");
  modal.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeTeamModal() {
  document.getElementById("teamModal").hidden = true;
  document.body.style.overflow = "";
}

/* Soccer ball that rolls across the bottom and spins with scroll progress */
function setupScrollBall() {
  const ball = document.getElementById("scrollBall");
  if (!ball) return;
  const onScroll = () => {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    const p = max > 0 ? Math.min(1, h.scrollTop / max) : 0;
    ball.style.left = (p * (window.innerWidth - 64)) + "px";
    ball.style.transform = `rotate(${p * 1440}deg)`;
    ball.style.opacity = p > 0.01 ? "1" : "0";
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  onScroll();
}

function renderChampion() {
  const champ = DATA.predicted_bracket.champion;
  const row = DATA.teams.find(t => t.team === champ);
  const finalRound = DATA.predicted_bracket.rounds.find(r => r.round === "final");
  const finalPct = finalRound ? finalRound.matches[0].win_pct : null;
  document.getElementById("championCard").innerHTML = `
    <p class="cr-kicker">Erik's model's pick to win it all</p>
    <div class="cr-flag">${flag(champ, 320)}<span class="cr-trophy">🏆</span></div>
    <div class="cr-team">${champ}</div>
    <div class="cr-prob"><b>${fmt(row.reach.champion)}%</b> title probability ·
      wins the final <b>${finalPct != null ? fmt(finalPct) : "—"}%</b> of the time</div>`;
}

function renderTitleRace() {
  const top = DATA.teams.slice(0, 14);
  TITLE_MAX = top[0].reach.champion;
  const bars = top.map((t, i) => `
      <div class="bar-row">
        <span class="rank">${i + 1}</span>
        ${flag(t.team, 40)}
        <div class="bar-track">
          <div class="bar-fill ${i === 0 ? "gold" : ""}" data-w="${Math.max(6, (t.reach.champion / TITLE_MAX) * 100)}">
            <span class="bar-team">${t.team}</span>
          </div>
        </div>
        <span class="pct">${fmt(t.reach.champion)}%</span>
      </div>`).join("");

  // Row 15: pick ANY country from a dropdown to see its odds
  const others = [...DATA.teams].sort((a, b) => a.team.localeCompare(b.team));
  const options = others.map(t => `<option value="${t.team}">${t.team}</option>`).join("");
  const dropRow = `
      <div class="bar-row drop-row">
        <span class="rank">15</span>
        <span class="drop-flag" id="dropFlag"></span>
        <div class="bar-track">
          <div class="bar-fill custom" id="customFill"><span class="bar-team" id="customName"></span></div>
        </div>
        <span class="pct" id="customPct">—</span>
      </div>
      <div class="drop-wrap">
        <label for="teamSelect">Don't see your team? Pick any country:</label>
        <select id="teamSelect">
          <option value="">Choose a country…</option>
          ${options}
        </select>
      </div>`;

  document.getElementById("titleBars").innerHTML = bars + dropRow;
  document.getElementById("teamSelect").addEventListener("change", onTeamSelect);
}

function onTeamSelect(e) {
  const name = e.target.value;
  const fill = document.getElementById("customFill");
  const pctEl = document.getElementById("customPct");
  const flagEl = document.getElementById("dropFlag");
  const nameEl = document.getElementById("customName");
  if (!name) {
    fill.style.width = "0"; pctEl.textContent = "—"; flagEl.innerHTML = ""; nameEl.textContent = "";
    return;
  }
  const t = DATA.teams.find(x => x.team === name);
  flagEl.innerHTML = flag(name, 40);
  nameEl.textContent = name;
  fill.style.width = Math.max(4, (t.reach.champion / TITLE_MAX) * 100) + "%";
  pctEl.textContent = fmt(t.reach.champion) + "%";
}

function heatStyle(v) {
  // green heat scale; gold for very high
  if (v >= 50) return `background:rgba(255,210,74,${0.25 + v/200});color:#06140d`;
  const a = 0.06 + (v / 100) * 0.8;
  return `background:rgba(43,255,136,${a});color:#06140d`;
}

function renderReachTable() {
  const cols = ["round_of_16", "quarterfinal", "semifinal", "final", "champion"];
  const head = `<thead><tr>
      <th class="team-col">Team</th>
      ${cols.map(c => `<th>${ROUND_LABEL[c]}</th>`).join("")}
    </tr></thead>`;
  const rows = DATA.teams.slice(0, 16).map(t => `
    <tr>
      <td class="team-col">${flag(t.team, 40)} ${t.team}</td>
      ${cols.map(c => {
        const v = t.reach[c];
        return `<td><span class="heat" style="${heatStyle(v)};padding:.25rem .5rem;display:inline-block;min-width:46px">${fmt(v)}%</span></td>`;
      }).join("")}
    </tr>`).join("");
  document.getElementById("reachTable").innerHTML = head + `<tbody>${rows}</tbody>`;
}

function matchCard(m) {
  const row = (team) => {
    const win = team === m.winner;
    const pct = win ? m.win_pct : 100 - m.win_pct;   // no draws in knockout
    return `<div class="bk-team ${win ? "win" : "lose"}">
        ${flag(team, 40)} <span class="slot-team">${team}</span>
        <span class="slot-pct">${fmt(pct)}%</span>
      </div>`;
  };
  return `<div class="match-card">${row(m.home)}${row(m.away)}</div>`;
}

function renderBracket() {
  // Show Round of 16 onward as real matchups (skip the 16-match R32 for clarity).
  const rounds = DATA.predicted_bracket.rounds.filter(r => r.round !== "round_of_32");
  const cols = rounds.map(r => `
    <div class="bk-round">
      <div class="round-title">${ROUND_LABEL[r.round]}</div>
      <div class="bk-matches">${r.matches.map(matchCard).join("")}</div>
    </div>`).join("");

  const champ = DATA.predicted_bracket.champion;
  const champCol = `
    <div class="bk-round champ-col">
      <div class="round-title">Champion</div>
      <div class="bk-matches">
        <div class="slot champ">${flag(champ, 40)} <span class="slot-team">${champ}</span>
          <span class="trophy">🏆</span></div>
      </div>
    </div>`;
  document.getElementById("bracket").innerHTML = cols + champCol;
}

function renderGroups() {
  const byGroup = {};
  DATA.group_predictions.forEach(m => (byGroup[m.group] ??= []).push(m));
  const html = Object.keys(byGroup).sort().map(g => {
    const matches = byGroup[g].map(m => `
      <div class="match">
        <div class="match-teams">
          <span class="t">${flag(m.home, 40)} ${m.home}</span>
          <span class="t">${m.away} ${flag(m.away, 40)}</span>
        </div>
        <div class="wdl">
          <span class="w" style="width:${m.home_win}%"></span>
          <span class="d" style="width:${m.draw}%"></span>
          <span class="l" style="width:${m.away_win}%"></span>
        </div>
        <div class="wdl-legend"><span>${fmt(m.home_win)}%</span><span>Draw ${fmt(m.draw)}%</span><span>${fmt(m.away_win)}%</span></div>
      </div>`).join("");
    return `<div class="group-card"><h3>Group ${g}</h3>${matches}</div>`;
  }).join("");
  document.getElementById("groups").innerHTML = html;
}

function renderFooter() {
  document.getElementById("footerMeta").innerHTML =
    `Model ${DATA.model_version} · ${DATA.n_simulations.toLocaleString()} simulations · ` +
    `temperature ${DATA.temperature} · snapshot ${DATA.date}`;
}

/* Scroll-reveal + animate the title bars when they enter the viewport */
function setupReveal() {
  document.querySelectorAll(".section-head, .reach-table-wrap, .champion-card")
    .forEach(el => el.classList.add("reveal"));

  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      e.target.classList.add("visible");
      if (e.target.id === "titleRace") {
        e.target.querySelectorAll(".bar-fill").forEach(f => { f.style.width = f.dataset.w + "%"; });
      }
      obs.unobserve(e.target);
    });
  }, { threshold: 0.15 });

  document.querySelectorAll(".reveal, #titleRace").forEach(el => obs.observe(el));
}
