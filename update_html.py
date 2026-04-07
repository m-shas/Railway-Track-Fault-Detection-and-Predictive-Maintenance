import os
import sys

FILE_PATH = r"d:\COLLEGE\projects\minor-proj\minor1\src\pipeline.py"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

start_idx = content.find("def _get_dashboard_html(data_json: str) -> str:")
end_idx = content.find("# ── ENTRY POINT", start_idx)

if start_idx == -1 or end_idx == -1:
    print("Could not find the bounds!")
    sys.exit(1)

new_func = """def _get_dashboard_html(data_json: str) -> str:
    \"\"\"Return the full HTML dashboard string.\"\"\"
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Railway Track Fault Detection Dashboard</title>
<meta name="description" content="AI-powered Railway Track Fault Detection and Predictive Maintenance Dashboard">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg-primary: #0f1923;
  --bg-secondary: #1a2736;
  --bg-card: #213243;
  --text-primary: #e8edf2;
  --text-secondary: #8899aa;
  --accent-blue: #3b82f6;
  --accent-green: #10b981;
  --accent-yellow: #f59e0b;
  --accent-red: #ef4444;
  --accent-purple: #8b5cf6;
  --border: #2d3f52;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg-primary); color:var(--text-primary); display:flex; height:100vh; overflow:hidden; }}

.sidebar {{ width:260px; background:var(--bg-secondary); border-right:1px solid var(--border); display:flex; flex-direction:column; padding:24px 16px; overflow-y:auto; }}
.sidebar h2 {{ font-size:1.2rem; font-weight:700; margin-bottom:24px; padding-left:8px; }}
.sidebar h2 span {{ color:var(--accent-blue); }}
.sidebar .badge {{ background:var(--accent-green); color:#fff; padding:4px 12px; border-radius:12px; font-size:0.75rem; font-weight:600; display:inline-block; margin-left:8px; margin-bottom:24px; }}

.tabs {{ display:flex; flex-direction:column; gap:4px; }}
.tab {{ padding:10px 12px; cursor:pointer; color:var(--text-secondary); border-radius:6px; font-weight:500; transition:all 0.2s; text-align:left; border:none; background:transparent; font-size:1rem; font-family:inherit; }}
.tab:hover {{ color:var(--text-primary); background:rgba(59,130,246,0.05); }}
.tab.active {{ color:var(--text-primary); background:rgba(59,130,246,0.15); font-weight:600; border-left:4px solid var(--accent-blue); padding-left:8px; }}

.main-content {{ flex:1; overflow-y:auto; padding:32px 48px; background:var(--bg-primary); }}
.header {{ margin-bottom:24px; border-bottom:1px solid var(--border); padding-bottom:16px; }}
.header h1 {{ font-size:2rem; font-weight:700; }}

.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}

.grid {{ display:grid; gap:20px; }}
.grid-6 {{ grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); }}
.grid-2 {{ grid-template-columns:repeat(auto-fill,minmax(400px,1fr)); }}
.card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
.card h3 {{ color:var(--text-secondary); font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px; }}
.card .value {{ font-size:1.8rem; font-weight:700; }}
.card .sub {{ color:var(--text-secondary); font-size:0.8rem; margin-top:4px; }}
.chart-card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
.chart-card h3 {{ color:var(--text-primary); font-size:1rem; margin-bottom:16px; font-weight:600; }}
.health-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); gap:8px; margin-top:16px; }}
.health-block {{ padding:12px; border-radius:8px; text-align:center; font-size:0.8rem; font-weight:600; }}
.health-block .score {{ font-size:1.2rem; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
th {{ text-align:left; padding:10px 12px; background:var(--bg-secondary); color:var(--text-secondary); font-weight:600; border-bottom:1px solid var(--border); }}
td {{ padding:10px 12px; border-bottom:1px solid var(--border); }}
tr:hover {{ background:rgba(59,130,246,0.05); }}
.badge-critical {{ background:rgba(239,68,68,0.15); color:var(--accent-red); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; }}
.badge-warning {{ background:rgba(245,158,11,0.15); color:var(--accent-yellow); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; }}
.badge-healthy {{ background:rgba(16,185,129,0.15); color:var(--accent-green); padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem; }}
.health-bar {{ height:6px; background:var(--border); border-radius:3px; overflow:hidden; }}
.health-bar-fill {{ height:100%; border-radius:3px; transition:width 0.5s; }}
.filter-bar {{ display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap; }}
.filter-bar select {{ padding:8px 12px; background:var(--bg-secondary); color:var(--text-primary); border:1px solid var(--border); border-radius:6px; font-size:0.85rem; }}
.maint-cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; margin-top:16px; }}
.maint-card {{ background:var(--bg-secondary); border-left:3px solid var(--accent-blue); border-radius:8px; padding:16px; }}
.maint-card.critical {{ border-left-color:var(--accent-red); }}
.maint-card.warning {{ border-left-color:var(--accent-yellow); }}
.maint-card h4 {{ font-size:0.9rem; margin-bottom:6px; }}
.maint-card p {{ font-size:0.8rem; color:var(--text-secondary); }}
canvas {{ max-height: 350px; }}
</style>
</head>
<body>

<div class="sidebar">
  <h2>🚂 <span>Railway</span> AI Dashboard</h2>
  <span class="badge">Static Version</span>
  
  <div class="tabs" id="tabs">
    <div style="font-size:0.75rem; color:var(--text-secondary); margin-bottom:8px; padding-left:12px; text-transform:uppercase; letter-spacing:0.05em;">Navigate</div>
    <div class="tab active" data-tab="overview">Overview</div>
    <div class="tab" data-tab="live">Live Monitoring</div>
    <div class="tab" data-tab="vibration">Vibration Analysis</div>
    <div class="tab" data-tab="blocks">Track Blocks</div>
    <div class="tab" data-tab="models">AI Models</div>
    <div class="tab" data-tab="xai">XAI Explainability</div>
    <div class="tab" data-tab="alerts">Alerts</div>
  </div>
</div>

<div class="main-content">
  <!-- TAB 1: OVERVIEW -->
  <div class="tab-content active" id="tab-overview">
    <div class="header"><h1>Overview</h1></div>
    <div class="grid grid-6" id="kpi-cards"></div>
    <div class="grid grid-2" style="margin-top:20px;">
      <div class="chart-card"><h3>Alert Distribution</h3><canvas id="alertDonut"></canvas></div>
      <div class="chart-card"><h3>Failure Type Distribution</h3><canvas id="failureBar"></canvas></div>
    </div>
    <div class="grid grid-2" style="margin-top:20px;">
      <div class="chart-card"><h3>Location Health Overview</h3><canvas id="locationHealth"></canvas></div>
      <div class="chart-card"><h3>Maintenance Actions</h3><canvas id="maintDonut"></canvas></div>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Track Block Health Grid</h3>
      <div class="health-grid" id="healthGrid"></div>
    </div>
  </div>

  <!-- TAB 2: LIVE MONITORING -->
  <div class="tab-content" id="tab-live">
    <div class="header"><h1>🔴 Live Sensor Monitoring</h1></div>
    <div class="chart-card" style="text-align: center; padding: 60px 20px;">
      <h2 style="margin-bottom: 16px;">This feature is only available in Streamlit</h2>
      <p style="color: var(--text-secondary); max-width: 600px; margin: 0 auto; line-height: 1.6;">
        Live streaming telemetry and dynamic background processes require a backend server.<br><br>
        To access the real-time simulation engine, run the following in your terminal:<br><br>
        <code style="background: var(--bg-primary); padding: 8px 16px; border-radius: 6px; color: var(--accent-blue); font-size: 1.1rem; display: inline-block; margin-top: 10px;">streamlit run app.py</code>
      </p>
    </div>
  </div>

  <!-- TAB 3: VIBRATION -->
  <div class="tab-content" id="tab-vibration">
    <div class="header"><h1>Vibration Analysis</h1></div>
    <div class="grid grid-2">
      <div class="chart-card"><h3>Acceleration Magnitude (Time Series)</h3><canvas id="accelTS"></canvas></div>
      <div class="chart-card"><h3>Temperature vs Humidity</h3><canvas id="tempHumScatter"></canvas></div>
    </div>
    <div class="grid grid-2" style="margin-top:20px;">
      <div class="chart-card"><h3>PLC Vibration & Track Resistance</h3><canvas id="vibResLine"></canvas></div>
      <div class="chart-card"><h3>Vibration Histogram</h3><canvas id="vibHist"></canvas></div>
    </div>
  </div>

  <!-- TAB 4: TRACK BLOCKS -->
  <div class="tab-content" id="tab-blocks">
    <div class="header"><h1>Track Blocks</h1></div>
    <div class="grid grid-2">
      <div class="chart-card"><h3>RUL by Block</h3><canvas id="rulBar"></canvas></div>
      <div class="chart-card"><h3>Vibration by Block</h3><canvas id="vibBar"></canvas></div>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Block Detail Table</h3>
      <table id="blockTable">
        <thead><tr><th>Block ID</th><th>Avg Health</th><th>Avg RUL</th><th>Vibration</th><th>Critical</th><th>Warning</th><th>Anomalies</th><th>Health Bar</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- TAB 5: AI MODELS -->
  <div class="tab-content" id="tab-models">
    <div class="header"><h1>AI Model Performance</h1></div>
    <div class="grid grid-2">
      <div class="chart-card"><h3>RUL: Actual vs Predicted</h3><canvas id="rulScatter"></canvas></div>
      <div class="chart-card"><h3>Feature Importance</h3><canvas id="featImpBar"></canvas></div>
    </div>
    <div class="grid grid-2" style="margin-top:20px;">
      <div class="chart-card"><h3>Confusion Matrix (Heatmap)</h3><canvas id="confMatrix"></canvas></div>
      <div class="chart-card">
        <h3>Model Performance Summary</h3>
        <table id="modelTable">
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody></tbody>
        </table>
        <div style="margin-top:20px;">
          <h3 style="font-size:1rem; margin-bottom:12px;">Pipeline Architecture</h3>
          <div style="background:var(--bg-secondary);padding:16px;border-radius:8px;font-family:monospace;font-size:0.8rem;color:var(--accent-blue);">
            CSV + XLSX → <span style="color:var(--accent-green)">Preprocess</span> → <span style="color:var(--accent-yellow)">Isolation Forest</span> → <span style="color:var(--accent-purple)">GradientBoosting (RUL)</span> + <span style="color:var(--accent-red)">RandomForest (Fault)</span> → Alert Engine → Dashboard
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 6: XAI EXPLAINABILITY -->
  <div class="tab-content" id="tab-xai">
    <div class="header"><h1>🧠 XAI — Explainable AI Dashboard</h1></div>
    <div class="chart-card" style="text-align: center; padding: 60px 20px;">
      <h2 style="margin-bottom: 16px;">This feature is only available in Streamlit</h2>
      <p style="color: var(--text-secondary); max-width: 600px; margin: 0 auto; line-height: 1.6;">
        Generating Shapley Additive exPlanations (SHAP) requires compiling the model in Python.<br><br>
        To access interactive SHAP waterfall models and beeswarm plots, run the following:<br><br>
        <code style="background: var(--bg-primary); padding: 8px 16px; border-radius: 6px; color: var(--accent-blue); font-size: 1.1rem; display: inline-block; margin-top: 10px;">streamlit run app.py</code>
      </p>
    </div>
  </div>

  <!-- TAB 7: ALERTS -->
  <div class="tab-content" id="tab-alerts">
    <div class="header"><h1>Alerts & Maintenance</h1></div>
    <div class="filter-bar">
      <select id="alertFilter"><option value="ALL">All Levels</option><option value="CRITICAL">Critical Only</option><option value="WARNING">Warning Only</option></select>
    </div>
    <div class="chart-card">
      <h3>Alert Table</h3>
      <table id="alertTable">
        <thead><tr><th>Block</th><th>Location</th><th>Level</th><th>Score</th><th>Fault</th><th>Action</th><th>Prob</th><th>RUL</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Maintenance Recommendations</h3>
      <div class="maint-cards" id="maintCards"></div>
    </div>
    <div class="chart-card" style="margin-top:20px;">
      <h3>Alert Timeline</h3>
      <canvas id="alertTimeline"></canvas>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};

// ── Tab Switching ──────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  }});
}});

// ── Helpers ────────────────────────────────────────────────────────────
const COLORS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#f97316','#14b8a6','#6366f1'];
function healthColor(s) {{ return s >= 70 ? '#10b981' : s >= 40 ? '#f59e0b' : '#ef4444'; }}

// ── TAB 1: OVERVIEW ───────────────────────────────────────────────────
(function() {{
  const bs = DATA.block_summary;
  const avgHealth = (bs.reduce((a,b)=>a+b.avg_health,0)/bs.length).toFixed(1);
  const totalCrit = bs.reduce((a,b)=>a+b.n_critical,0);
  const totalWarn = bs.reduce((a,b)=>a+b.n_warning,0);
  const totalAnom = bs.reduce((a,b)=>a+b.n_anomalies,0);
  const avgRul = (bs.reduce((a,b)=>a+b.avg_rul,0)/bs.length).toFixed(0);
  const kpis = [
    {{label:'Track Blocks',value:bs.length,color:'var(--accent-blue)',sub:'monitored'}},
    {{label:'Avg Health',value:avgHealth,color:healthColor(avgHealth),sub:'/ 100'}},
    {{label:'Critical Alerts',value:totalCrit,color:'var(--accent-red)',sub:'immediate'}},
    {{label:'Warnings',value:totalWarn,color:'var(--accent-yellow)',sub:'watch list'}},
    {{label:'Anomalies',value:totalAnom,color:'var(--accent-purple)',sub:'IF detected'}},
    {{label:'Avg RUL',value:avgRul+' d',color:'var(--accent-green)',sub:'days remaining'}}
  ];
  const kpiHtml = kpis.map(k => `<div class="card"><h3>${{k.label}}</h3><div class="value" style="color:${{k.color}}">${{k.value}}</div><div class="sub">${{k.sub}}</div></div>`).join('');
  document.getElementById('kpi-cards').innerHTML = kpiHtml;

  // Alert donut
  const ad = DATA.alert_distribution;
  new Chart(document.getElementById('alertDonut'), {{type:'doughnut',data:{{labels:Object.keys(ad),datasets:[{{data:Object.values(ad),backgroundColor:['#ef4444','#f59e0b','#10b981']}}]}},options:{{plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Failure bar
  const fd = DATA.failure_distribution;
  new Chart(document.getElementById('failureBar'), {{type:'bar',data:{{labels:Object.keys(fd),datasets:[{{label:'Count',data:Object.values(fd),backgroundColor:COLORS}}]}},options:{{plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa'}}}}}}}}}});

  // Location health
  const ls = DATA.location_summary;
  new Chart(document.getElementById('locationHealth'), {{type:'bar',data:{{labels:ls.map(l=>l.location_id),datasets:[{{label:'Avg Health',data:ls.map(l=>l.avg_health),backgroundColor:ls.map(l=>healthColor(l.avg_health))}},{{label:'Alerts',data:ls.map(l=>l.n_alerts),backgroundColor:'rgba(239,68,68,0.5)',type:'line',borderColor:'#ef4444',fill:false,yAxisID:'y1'}}]}},options:{{scales:{{y:{{ticks:{{color:'#8899aa'}}}},y1:{{position:'right',ticks:{{color:'#8899aa'}},grid:{{display:false}}}},x:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Maintenance donut
  const md = DATA.maintenance_distribution;
  new Chart(document.getElementById('maintDonut'), {{type:'doughnut',data:{{labels:Object.keys(md),datasets:[{{data:Object.values(md),backgroundColor:COLORS}}]}},options:{{plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Health grid
  const gridHtml = bs.map(b => {{
    const c = healthColor(b.avg_health);
    return `<div class="health-block" style="background:${{c}}22;border:1px solid ${{c}}44;color:${{c}}"><div>${{b.block_id}}</div><div class="score">${{b.avg_health}}</div></div>`;
  }}).join('');
  document.getElementById('healthGrid').innerHTML = gridHtml;
}})();

// ── TAB 3: VIBRATION ──────────────────────────────────────────────────
(function() {{
  const va = DATA.vibration_accel;
  const labels = Array.from({{length:va.accel_magnitude.length}},(_,i)=>i);

  // Accel time series with anomaly dots
  const normalData = va.accel_magnitude.map((v,i) => va.vibr_anomaly[i]===0 ? v : null);
  const anomalyData = va.accel_magnitude.map((v,i) => va.vibr_anomaly[i]===1 ? v : null);
  new Chart(document.getElementById('accelTS'), {{type:'line',data:{{labels,datasets:[
    {{label:'Normal',data:normalData,borderColor:'#3b82f6',pointRadius:0,borderWidth:1}},
    {{label:'Anomaly',data:anomalyData,borderColor:'#ef4444',backgroundColor:'#ef4444',pointRadius:3,showLine:false}}
  ]}},options:{{scales:{{x:{{display:false}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Temp vs Hum scatter
  if(va.temp && va.hum) {{
    new Chart(document.getElementById('tempHumScatter'), {{type:'scatter',data:{{datasets:[{{label:'Temp/Hum',data:va.temp.map((t,i)=>({{x:t,y:va.hum[i]}})),backgroundColor:'rgba(139,92,246,0.4)',pointRadius:2}}]}},options:{{scales:{{x:{{title:{{display:true,text:'Temperature',color:'#8899aa'}},ticks:{{color:'#8899aa'}}}},y:{{title:{{display:true,text:'Humidity',color:'#8899aa'}},ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});
  }}

  // PLC Vibration vs Resistance
  const ts = DATA.time_series;
  const tsLabels = ts.timestamps.map(t=>t.substr(11,5));
  new Chart(document.getElementById('vibResLine'), {{type:'line',data:{{labels:tsLabels.slice(0,200),datasets:[
    {{label:'Vibration',data:ts.vibration.slice(0,200),borderColor:'#3b82f6',borderWidth:1,pointRadius:0,yAxisID:'y'}},
    {{label:'FailureProb',data:ts.failure_prob.slice(0,200),borderColor:'#ef4444',borderWidth:1,pointRadius:0,yAxisID:'y1'}}
  ]}},options:{{scales:{{y:{{ticks:{{color:'#8899aa'}}}},y1:{{position:'right',ticks:{{color:'#8899aa'}},grid:{{display:false}}}},x:{{ticks:{{color:'#8899aa',maxTicksLimit:10}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Vibration histogram
  const bins = Array.from({{length:20}},(_,i)=>i);
  const hist = new Array(20).fill(0);
  const magMin = Math.min(...va.accel_magnitude), magMax = Math.max(...va.accel_magnitude);
  const step = (magMax-magMin)/20;
  va.accel_magnitude.forEach(v => {{ const b = Math.min(19,Math.floor((v-magMin)/step)); hist[b]++; }});
  new Chart(document.getElementById('vibHist'), {{type:'bar',data:{{labels:bins.map(i=>(magMin+i*step).toFixed(2)),datasets:[{{label:'Count',data:hist,backgroundColor:'#10b981'}}]}},options:{{scales:{{x:{{ticks:{{color:'#8899aa',maxTicksLimit:10}}}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}}}}}}}});
}})();

// ── TAB 4: TRACK BLOCKS ──────────────────────────────────────────────
(function() {{
  const bs = DATA.block_summary.sort((a,b) => a.block_id.localeCompare(b.block_id, undefined, {{numeric:true}}));

  new Chart(document.getElementById('rulBar'), {{type:'bar',data:{{labels:bs.map(b=>b.block_id),datasets:[{{label:'Avg RUL (days)',data:bs.map(b=>b.avg_rul),backgroundColor:bs.map(b=>b.avg_rul<60?'#ef4444':b.avg_rul<120?'#f59e0b':'#10b981')}}]}},options:{{indexAxis:'y',scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}}}}}}}});

  new Chart(document.getElementById('vibBar'), {{type:'bar',data:{{labels:bs.map(b=>b.block_id),datasets:[{{label:'Avg Vibration',data:bs.map(b=>b.avg_vibration),backgroundColor:'#8b5cf6'}}]}},options:{{indexAxis:'y',scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}}}}}}}});

  // Table
  const tbody = document.querySelector('#blockTable tbody');
  tbody.innerHTML = bs.map(b => {{
    const hc = healthColor(b.avg_health);
    return `<tr><td>${{b.block_id}}</td><td style="color:${{hc}}">${{b.avg_health}}</td><td>${{b.avg_rul}} d</td><td>${{b.avg_vibration}}</td><td style="color:var(--accent-red)">${{b.n_critical}}</td><td style="color:var(--accent-yellow)">${{b.n_warning}}</td><td>${{b.n_anomalies}}</td><td><div class="health-bar"><div class="health-bar-fill" style="width:${{b.avg_health}}%;background:${{hc}}"></div></div></td></tr>`;
  }}).join('');
}})();

// ── TAB 5: AI MODELS ──────────────────────────────────────────────────
(function() {{
  const rs = DATA.rul_scatter;
  new Chart(document.getElementById('rulScatter'), {{type:'scatter',data:{{datasets:[
    {{label:'Predictions',data:rs.actual.map((a,i)=>({{x:a,y:rs.predicted[i]}})),backgroundColor:'rgba(59,130,246,0.4)',pointRadius:2}},
    {{label:'Perfect',data:[{{x:0,y:0}},{{x:Math.max(...rs.actual),y:Math.max(...rs.actual)}}],borderColor:'#ef4444',borderDash:[5,5],showLine:true,pointRadius:0,borderWidth:1}}
  ]}},options:{{scales:{{x:{{title:{{display:true,text:'Actual RUL',color:'#8899aa'}},ticks:{{color:'#8899aa'}}}},y:{{title:{{display:true,text:'Predicted RUL',color:'#8899aa'}},ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{labels:{{color:'#e8edf2'}}}}}}}}}});

  // Feature importance
  const fi = DATA.feature_importance;
  const fiSorted = Object.entries(fi).sort((a,b) => b[1]-a[1]);
  new Chart(document.getElementById('featImpBar'), {{type:'bar',data:{{labels:fiSorted.map(f=>f[0]),datasets:[{{label:'Importance',data:fiSorted.map(f=>f[1]),backgroundColor:COLORS}}]}},options:{{indexAxis:'y',scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa',font:{{size:10}}}}}}}},plugins:{{legend:{{display:false}}}}}}}});

  // Confusion matrix
  const cm = DATA.model_metrics.confusion_matrix;
  if(cm && cm.length > 0) {{
    const cmLabels = cm.map((_,i)=>'C'+(i+1));
    const cmData = [];
    cm.forEach((row,i) => row.forEach((val,j) => cmData.push({{x:j,y:i,v:val}})));
    const maxVal = Math.max(...cmData.map(d=>d.v));
    new Chart(document.getElementById('confMatrix'), {{type:'scatter',data:{{datasets:[{{
      data:cmData.map(d=>({{x:d.x,y:d.y}})),
      backgroundColor:cmData.map(d=>`rgba(59,130,246,${{Math.max(0.1,d.v/maxVal)}})`),
      pointRadius:cmData.map(d=>5+15*(d.v/maxVal)),
      pointStyle:'rect'
    }}]}},options:{{scales:{{x:{{min:-0.5,max:cm[0].length-0.5,ticks:{{callback:v=>cmLabels[v]||'',color:'#8899aa'}}}},y:{{min:-0.5,max:cm.length-0.5,reverse:true,ticks:{{callback:v=>cmLabels[v]||'',color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>{{const d=cmData[ctx.dataIndex];return `${{cmLabels[d.y]}}→${{cmLabels[d.x]}}: ${{d.v}}`}}}}}}}}}}}});
  }}

  // Model metrics table
  const mm = DATA.model_metrics;
  const mtBody = document.querySelector('#modelTable tbody');
  mtBody.innerHTML = `
    <tr><td>RUL MAE</td><td>${{mm.rul.mae.toFixed(2)}} days</td></tr>
    <tr><td>RUL RMSE</td><td>${{mm.rul.rmse.toFixed(2)}} days</td></tr>
    <tr><td>RUL R²</td><td>${{mm.rul.r2.toFixed(4)}}</td></tr>
    <tr><td>Classifier Accuracy</td><td>${{(mm.classifier_accuracy*100).toFixed(1)}}%</td></tr>
    <tr><td>Fault Classes</td><td>${{cm?cm.length:10}}</td></tr>
  `;
}})();

// ── TAB 7: ALERTS ─────────────────────────────────────────────────────
(function() {{
  const alerts = DATA.alerts;
  function renderAlerts(filter) {{
    const filtered = filter === 'ALL' ? alerts : alerts.filter(a=>a.level===filter);
    const tbody = document.querySelector('#alertTable tbody');
    tbody.innerHTML = filtered.map(a => {{
      const bc = a.level==='CRITICAL'?'badge-critical':'badge-warning';
      return `<tr><td>${{a.block_id}}</td><td>${{a.location}}</td><td><span class="${{bc}}">${{a.level}}</span></td><td>${{a.score}}</td><td>${{a.fault}}</td><td style="max-width:200px;font-size:0.75rem">${{a.action}}</td><td>${{a.failure_prob}}</td><td>${{a.rul}}</td></tr>`;
    }}).join('');

    // Maint cards
    const mc = document.getElementById('maintCards');
    mc.innerHTML = filtered.slice(0,8).map(a => {{
      const cls = a.level==='CRITICAL'?'critical':'warning';
      return `<div class="maint-card ${{cls}}"><h4>${{a.block_id}} — ${{a.fault}}</h4><p>${{a.action}}</p><p style="margin-top:6px;color:var(--accent-blue)">Score: ${{a.score}} | RUL: ${{a.rul}}d</p></div>`;
    }}).join('');
  }}
  renderAlerts('ALL');
  document.getElementById('alertFilter').addEventListener('change', e => renderAlerts(e.target.value));

  // Alert timeline
  const levelCounts = {{}};
  alerts.forEach(a => {{ levelCounts[a.level] = (levelCounts[a.level]||0)+1; }});
  new Chart(document.getElementById('alertTimeline'), {{type:'bar',data:{{labels:Object.keys(levelCounts),datasets:[{{label:'Count',data:Object.values(levelCounts),backgroundColor:Object.keys(levelCounts).map(l=>l==='CRITICAL'?'#ef4444':'#f59e0b')}}]}},options:{{scales:{{x:{{ticks:{{color:'#8899aa'}}}},y:{{ticks:{{color:'#8899aa'}}}}}},plugins:{{legend:{{display:false}}}}}}}});
}})();
</script>
</body>
</html>'''
"""

new_content = content[:start_idx] + new_func + "\n\n" + content[end_idx:]

with open(FILE_PATH, "w", encoding="utf-8") as f:
    f.write(new_content)
    
print("Successfully patched pipeline.py")
