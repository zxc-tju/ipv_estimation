const appEl=document.getElementById('app');
const progressWrap=document.getElementById('progress-wrap');
const progressBar=document.getElementById('progress-bar');
const progressText=document.getElementById('progress-text');
const progressNumber=document.getElementById('progress-number');
let config=null, token=localStorage.getItem('sociality_session_token'), currentTrial=null;
let trialStartedAt=0, replayCount=0;

async function api(path,options={}){
  const res=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});
  if(!res.ok){const b=await res.json().catch(()=>({detail:res.statusText}));throw new Error(b.detail||'请求失败')}
  return res.json();
}
function esc(v){return String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function setProgress(done=0,total=0,label=''){
  if(!total){progressWrap.classList.add('hidden');return}
  progressWrap.classList.remove('hidden');progressBar.style.width=`${Math.round(100*done/total)}%`;
  progressText.textContent=label||'实验进度';progressNumber.textContent=`${done} / ${total}`;
}
function errorBox(msg){return `<div class="notice warning"><strong>暂时无法继续</strong><div>${esc(msg)}</div></div>`}

function showWelcome(){
  setProgress();
  appEl.innerHTML=`<h1>${esc(config.title)}</h1><p class="lead">${esc(config.subtitle)}</p><div class="notice">${esc(config.consent_text)}</div><div class="actions">${token?'<button class="btn secondary" id="resume">继续上次实验</button>':''}<button class="btn" id="begin">开始阅读知情说明</button></div>`;
  document.getElementById('begin').onclick=showProfile;
  if(token)document.getElementById('resume').onclick=resume;
}
function showProfile(){
  appEl.innerHTML=`<h2>知情同意与基本信息</h2><p class="muted">只收集分档后的驾驶背景，不收集姓名、身份证号或驾驶证号码。</p>
  <form id="profile" class="grid">
    <label class="checkbox"><input type="checkbox" name="consent" required><span>我已阅读研究说明，自愿参加，并理解可以随时退出。</span></label>
    <div class="grid two">
      ${selectField('age_band','年龄段',['18–24','25–34','35–44','45–54','55–64','65+'],true)}
      ${selectField('gender_optional','性别（可不回答）',['女','男','其他'],false,'不回答')}
      ${selectField('valid_license','目前是否持有有效机动车驾驶证',['是','否'],true)}
      ${selectField('years_licensed','取得驾驶证已有多少年',['<1年','1–3年','4–7年','8–15年','>15年'],true)}
      ${selectField('driving_frequency','过去一年驾驶频率',['几乎每天','每周数次','每周一次','每月数次','很少'],true)}
      ${selectField('annual_mileage_band','过去一年大致驾驶里程（可选）',['<1,000 km','1,000–5,000 km','5,001–10,000 km','10,001–20,000 km','>20,000 km'],false,'不确定/不回答')}
      ${selectField('urban_driving_frequency','城市道路驾驶频率',['从不','很少','有时','经常','几乎总是'],true)}
      ${selectField('professional_driver','是否从事职业驾驶（可选）',['是','否'],false,'不回答')}
    </div><div id="profile-error" class="error-text"></div><div class="actions"><button type="button" class="btn secondary" id="back">返回</button><button class="btn" type="submit">创建匿名实验会话</button></div>
  </form>`;
  document.getElementById('back').onclick=showWelcome;
  document.getElementById('profile').onsubmit=async e=>{
    e.preventDefault();const f=new FormData(e.currentTarget),p=Object.fromEntries(f.entries());
    p.consent=!!f.get('consent');p.valid_license=p.valid_license==='是';
    p.professional_driver=p.professional_driver? p.professional_driver==='是':null;
    p.user_agent=navigator.userAgent;p.device_type=/Mobi|Android/i.test(navigator.userAgent)?'mobile':'desktop';
    try{const out=await api('/api/session/start',{method:'POST',body:JSON.stringify(p)});token=out.token;localStorage.setItem('sociality_session_token',token);showInstructions()}
    catch(err){document.getElementById('profile-error').textContent=err.message}
  };
}
function selectField(name,label,options,required,blank='请选择'){
  return `<div class="field"><label>${label}</label><select name="${name}" ${required?'required':''}><option value="">${blank}</option>${options.map(x=>`<option>${x}</option>`).join('')}</select></div>`;
}
function showInstructions(){
  setProgress();appEl.innerHTML=`<h2>任务说明</h2><div class="notice"><strong>最重要的规则</strong><p>${esc(config.participant_instruction)}</p></div>
  <h3>第一部分：成对选择</h3><p>同一场景会依次播放视频 A 和 B。两段都结束后，请选择您更愿意与哪辆车互动，也可以选择“无明显偏好”。</p>
  <h3>第二部分：单片段评价</h3><p>观看一段视频后，对目标车辆作出多项评分。</p>
  <div class="practice-question"><strong>理解检查 1</strong><p>您需要代入哪一方？</p><label><input type="radio" name="q1" value="target"> 被标记的目标车辆</label><br><label><input type="radio" name="q1" value="counterpart"> 与目标车辆互动的另一辆车</label></div>
  <div class="practice-question"><strong>理解检查 2</strong><p>实验会告诉您视频来自人类还是自动驾驶系统吗？</p><label><input type="radio" name="q2" value="yes"> 会</label><br><label><input type="radio" name="q2" value="no"> 不会</label></div>
  <div id="instruction-error" class="error-text"></div><div class="actions"><button class="btn" id="start">我已理解，开始正式实验</button></div>`;
  document.getElementById('start').onclick=()=>{
    if(document.querySelector('input[name=q1]:checked')?.value!=='counterpart'||document.querySelector('input[name=q2]:checked')?.value!=='no'){
      document.getElementById('instruction-error').textContent='请重新阅读：您需要代入另一辆车；视频来源不会告知。';return;
    }loadState();
  };
}
async function resume(){try{await loadState()}catch(err){token=null;localStorage.removeItem('sociality_session_token');appEl.innerHTML=errorBox(err.message)+`<div class="actions"><button class="btn" onclick="location.reload()">重新开始</button></div>`}}
async function loadState(){
  const state=await api(`/api/session/${encodeURIComponent(token)}/state`);
  if(state.status==='completed')return showComplete();if(state.status==='post')return showPost();
  currentTrial=state.trial;replayCount=0;trialStartedAt=performance.now();
  setProgress(state.completed,state.total,currentTrial.type==='pairwise'?'成对选择':'单片段评价');
  currentTrial.type==='pairwise'?renderPair(currentTrial,true):renderSingle(currentTrial,true);
}
function videoHtml(url,label,key){
  if(url)return `<div class="video-frame"><div class="video-header"><strong>${label}</strong><span>请完整播放</span></div><video id="video-${key}" src="${esc(url)}" controls playsinline preload="metadata" controlsList="nodownload"></video></div>`;
  const sim=config.allow_placeholder_trials?`<button class="btn secondary small" data-sim="${key}">模拟完成播放（开发测试）</button>`:'';
  return `<div class="video-frame"><div class="video-header"><strong>${label}</strong><span>视频待添加</span></div><div class="video-placeholder"><div><strong>Video placeholder</strong><p>研究人员尚未上传该视频。</p>${sim}</div></div></div>`;
}
function bindVideo(key,onDone){
  const v=document.getElementById(`video-${key}`);if(v){v.addEventListener('play',()=>logEvent('video_play',{key}));v.addEventListener('ended',()=>{logEvent('video_ended',{key});onDone()})}
  document.querySelector(`[data-sim="${key}"]`)?.addEventListener('click',onDone);
}
function renderPair(t,initial=false){
  let played={a:false,b:false},selected=null;if(initial)replayCount=0;
  appEl.innerHTML=`<div class="muted">${esc(t.scenario_label)} · 任务 ${t.order_index}</div><h2>依次观看视频 A 和视频 B</h2><p>${esc(t.instruction)}</p>
  <div class="sequence-strip"><span id="sa" class="sequence-step active">1 视频 A</span><span>→</span><span id="sb" class="sequence-step">2 视频 B</span><span>→</span><span id="sr" class="sequence-step">3 作答</span></div>
  <div id="va" class="video-stage">${videoHtml(t.video_a_url,'视频 A','a')}</div><div id="vb" class="video-stage hidden">${videoHtml(t.video_b_url,'视频 B','b')}</div>
  <section id="pair-response" class="hidden"><h3>如果您是另一辆车的驾驶员，您更愿意与哪辆车这样互动？</h3>
  <div class="choice-grid">${[['A','Vehicle A'],['B','Vehicle B'],['NO_PREFERENCE','无明显偏好']].map(([v,l])=>`<button class="choice-card" data-choice="${v}">${l}</button>`).join('')}</div>
  <div class="field"><label>您对选择有多大把握？</label><select id="pair-confidence"><option value="">请选择</option><option value="1">1 完全没把握</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5 非常有把握</option></select></div>
  <div id="pair-error" class="error-text"></div><div class="actions"><button class="btn secondary" id="replay">完整重播 A 和 B</button><button class="btn" id="submit" disabled>提交并继续</button></div></section>`;
  bindVideo('a',()=>{played.a=true;document.getElementById('va').classList.add('hidden');document.getElementById('vb').classList.remove('hidden');document.getElementById('sa').classList.remove('active');document.getElementById('sb').classList.add('active')});
  bindVideo('b',()=>{played.b=true;document.getElementById('vb').classList.add('hidden');document.getElementById('pair-response').classList.remove('hidden');document.getElementById('sb').classList.remove('active');document.getElementById('sr').classList.add('active')});
  const validate=()=>document.getElementById('submit').disabled=!(selected&&document.getElementById('pair-confidence').value);
  document.querySelectorAll('[data-choice]').forEach(btn=>btn.onclick=e=>{e.preventDefault();selected=btn.dataset.choice;document.querySelectorAll('[data-choice]').forEach(x=>x.classList.toggle('selected',x===btn));validate()});
  document.getElementById('pair-confidence').onchange=validate;
  document.getElementById('replay').onclick=()=>{if(replayCount>=config.max_replays){document.getElementById('pair-error').textContent='已达到允许重播次数。';return}replayCount++;logEvent('pair_replay',{replay_count:replayCount});renderPair(t,false)};
  document.getElementById('submit').onclick=async()=>{try{await api(`/api/session/${encodeURIComponent(token)}/pairwise`,{method:'POST',body:JSON.stringify({trial_id:t.id,preference_raw:selected,choice_confidence:Number(document.getElementById('pair-confidence').value),response_time_ms:Math.round(performance.now()-trialStartedAt),playback_a_complete:played.a,playback_b_complete:played.b,replay_count:replayCount})});loadState()}catch(err){document.getElementById('pair-error').textContent=err.message}};
}
const items=[
 ['acceptability','在当前路况下，这辆车的处理方式是可以接受的。','完全不可接受','完全可以接受'],
 ['predictability','这辆车接下来的行为容易预判。','完全无法预判','非常容易预判'],
 ['comfort','如果我是另一辆车的驾驶员，与这辆车互动会让我感到舒适。','非常不舒适','非常舒适'],
 ['interaction_burden','这辆车让另一辆车作出了不必要的额外减速、等待或调整。','完全没有','非常明显'],
 ['perceived_unsafe','这辆车的行为让我感到不安全。','完全没有','非常明显'],
 ['assertiveness','这辆车表现得过于激进或强势。','完全没有','非常明显'],
 ['hesitation','这辆车表现得过于谨慎、迟疑或让行过度。','完全没有','非常明显']
];
function rating([name,title,left,right]){return `<div class="rating-row"><div class="rating-title">${title}</div><div class="scale">${[1,2,3,4,5,6,7].map(n=>`<label><input type="radio" name="${name}" value="${n}" required>${n}</label>`).join('')}</div><div class="scale-anchors"><span>${left}</span><span>${right}</span></div></div>`}
function renderSingle(t,initial=false){
  let played=false;if(initial)replayCount=0;
  appEl.innerHTML=`<div class="muted">${esc(t.scenario_label)} · 任务 ${t.order_index}</div><h2>观看视频并评价目标车辆</h2><p>${esc(t.instruction)}</p><div id="sv" class="video-stage">${videoHtml(t.video_url,'目标车辆互动视频','single')}</div>
  <form id="single-form" class="hidden"><div class="rating-list">${items.map(rating).join('')}</div><div class="field"><label>评分置信度（可选）</label><select name="rating_confidence"><option value="">不回答</option>${[1,2,3,4,5].map(n=>`<option value="${n}">${n}${n===1?' 完全没把握':n===5?' 非常有把握':''}</option>`).join('')}</select></div><div class="field"><label>主要判断依据（可选）</label><textarea name="free_text_reason" maxlength="2000"></textarea></div><div id="single-error" class="error-text"></div><div class="actions"><button type="button" class="btn secondary" id="replay-single">重播一次</button><button class="btn" type="submit">提交并继续</button></div></form>`;
  bindVideo('single',()=>{played=true;document.getElementById('sv').classList.add('hidden');document.getElementById('single-form').classList.remove('hidden')});
  document.getElementById('replay-single').onclick=()=>{if(replayCount>=config.max_replays){document.getElementById('single-error').textContent='已达到允许重播次数。';return}replayCount++;logEvent('single_replay',{replay_count:replayCount});renderSingle(t,false)};
  document.getElementById('single-form').onsubmit=async e=>{e.preventDefault();const f=new FormData(e.currentTarget),p={trial_id:t.id,playback_complete:played,replay_count:replayCount,response_time_ms:Math.round(performance.now()-trialStartedAt),courtesy:null};items.forEach(([n])=>p[n]=Number(f.get(n)));p.rating_confidence=f.get('rating_confidence')?Number(f.get('rating_confidence')):null;p.free_text_reason=f.get('free_text_reason')||null;try{await api(`/api/session/${encodeURIComponent(token)}/single`,{method:'POST',body:JSON.stringify(p)});loadState()}catch(err){document.getElementById('single-error').textContent=err.message}};
}
function showPost(){
  setProgress(1,1,'正式任务已完成');appEl.innerHTML=`<h2>实验后问卷</h2><form id="post" class="grid">
  ${selectField('task_difficulty','总体而言，这项任务有多难？',['1 非常容易','2','3','4','5','6','7 非常困难'],true)}
  ${selectField('source_guess','您能否判断哪些视频来自人类、哪些来自自动驾驶系统？',['完全无法判断','偶尔能判断','大多能判断','几乎都能判断'],true)}
  <div class="field"><label>您认为本实验主要想研究什么？（可选）</label><textarea name="hypothesis_guess"></textarea></div>
  <div class="field"><label>使用过哪些驾驶辅助功能？（可选）</label><input name="adas_experience" placeholder="例如 ACC、LKA、自动泊车"></div>
  ${selectField('av_ride_experience','是否体验过自动驾驶车辆？',['从未','1–2次','多次'],false,'不回答')}
  ${selectField('study_familiarity','实验前是否了解 IPV、驾驶社会性或本研究？',['完全不了解','听说过','比较了解','直接参与过相关研究'],false,'不回答')}
  <div class="field"><label>其他反馈（可选）</label><textarea name="open_feedback"></textarea></div><div id="post-error" class="error-text"></div><div class="actions"><button class="btn" type="submit">提交并结束实验</button></div></form>`;
  document.getElementById('post').onsubmit=async e=>{e.preventDefault();const f=new FormData(e.currentTarget),p=Object.fromEntries(f.entries());p.task_difficulty=Number(String(p.task_difficulty).split(' ')[0]);try{await api(`/api/session/${encodeURIComponent(token)}/post`,{method:'POST',body:JSON.stringify(p)});await api(`/api/session/${encodeURIComponent(token)}/complete`,{method:'POST',body:'{}'});showComplete()}catch(err){document.getElementById('post-error').textContent=err.message}};
}
function showComplete(){setProgress();localStorage.removeItem('sociality_session_token');token=null;appEl.innerHTML=`<h1>实验已完成</h1><div class="notice success"><strong>感谢您的参与。</strong><p>您的匿名回答已经保存。</p></div>`}
async function logEvent(type,payload={}){if(!token)return;try{await api(`/api/session/${encodeURIComponent(token)}/event`,{method:'POST',body:JSON.stringify({trial_id:currentTrial?.id||null,event_type:type,payload})})}catch(_){}}
document.addEventListener('visibilitychange',()=>logEvent(document.hidden?'page_hidden':'page_visible'));
(async()=>{try{config=await api('/api/public/config');showWelcome()}catch(err){appEl.innerHTML=errorBox(err.message)}})();
