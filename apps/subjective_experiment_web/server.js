'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { URL } = require('node:url');

const { Store } = require('./lib/store');
const {
  StudyConfigurationError,
  buildAssignment,
  loadStudyBundle,
  publicStimulus,
  sha256,
} = require('./lib/study');
const { DEMO_MEDIA } = require('./lib/demo-media');

const ROOT = __dirname;
const CONFIG = {
  port: Number(process.env.PORT || 3000),
  host: process.env.HOST || '0.0.0.0',
  secret: process.env.EXPERIMENT_SECRET_KEY || 'dev-change-me',
  adminToken: process.env.EXPERIMENT_ADMIN_TOKEN || 'change-me',
  database: path.resolve(process.env.EXPERIMENT_DATABASE || path.join(ROOT, 'data', 'experiment.sqlite3')),
  studyConfig: path.resolve(process.env.EXPERIMENT_STUDY_CONFIG || path.join(ROOT, 'config', 'study.demo.json')),
  stimulusManifest: path.resolve(process.env.EXPERIMENT_STIMULUS_MANIFEST || path.join(ROOT, 'config', 'stimuli.demo.csv')),
  secureCookie: process.env.EXPERIMENT_SECURE_COOKIE === '1',
  nodeEnv: process.env.NODE_ENV || 'development',
};

if (CONFIG.nodeEnv === 'production') {
  if (CONFIG.secret === 'dev-change-me' || CONFIG.adminToken === 'change-me') {
    throw new Error('Production requires strong EXPERIMENT_SECRET_KEY and EXPERIMENT_ADMIN_TOKEN values.');
  }
}

let bundle;
try {
  bundle = loadStudyBundle(CONFIG.studyConfig, CONFIG.stimulusManifest);
} catch (error) {
  if (!(error instanceof StudyConfigurationError)) throw error;
  console.error(`Study configuration error: ${error.message}`);
  process.exitCode = 1;
  throw error;
}
const store = new Store(CONFIG.database);

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function hmac(value) {
  return crypto.createHmac('sha256', CONFIG.secret).update(value).digest('base64url');
}

function signedValue(value) {
  return `${value}.${hmac(value)}`;
}

function verifySignedValue(candidate) {
  if (!candidate || !candidate.includes('.')) return null;
  const split = candidate.lastIndexOf('.');
  const value = candidate.slice(0, split);
  const signature = candidate.slice(split + 1);
  const expected = hmac(value);
  if (signature.length !== expected.length) return null;
  return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected)) ? value : null;
}

function parseCookies(req) {
  const header = req.headers.cookie || '';
  return Object.fromEntries(header.split(';').map((part) => {
    const index = part.indexOf('=');
    if (index < 0) return ['', ''];
    return [part.slice(0, index).trim(), decodeURIComponent(part.slice(index + 1).trim())];
  }).filter(([key]) => key));
}

function cookieHeader(name, value, { maxAge = null, httpOnly = true } = {}) {
  const parts = [
    `${name}=${encodeURIComponent(value)}`,
    'Path=/',
    'SameSite=Lax',
  ];
  if (httpOnly) parts.push('HttpOnly');
  if (CONFIG.secureCookie) parts.push('Secure');
  if (maxAge !== null) parts.push(`Max-Age=${maxAge}`);
  return parts.join('; ');
}

function participantContext(req) {
  const cookies = parseCookies(req);
  const sessionId = verifySignedValue(cookies.exp_session);
  if (!sessionId) return null;
  const context = store.getContext(sessionId);
  return context || null;
}

function isAdmin(req) {
  return verifySignedValue(parseCookies(req).exp_admin) === 'admin';
}

function timingSafeTextEqual(a, b) {
  const left = Buffer.from(String(a));
  const right = Buffer.from(String(b));
  if (left.length !== right.length) return false;
  return crypto.timingSafeEqual(left, right);
}

function baseHeaders(contentType = 'text/html; charset=utf-8') {
  return {
    'Content-Type': contentType,
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    'Content-Security-Policy': [
      "default-src 'self'",
      "script-src 'self'",
      "style-src 'self'",
      "img-src 'self' data:",
      "media-src 'self'",
      "connect-src 'self'",
      "font-src 'self'",
      "form-action 'self'",
      "frame-ancestors 'none'",
      "base-uri 'none'",
    ].join('; '),
  };
}

function send(res, status, body, headers = {}) {
  const data = Buffer.isBuffer(body) ? body : Buffer.from(String(body));
  res.writeHead(status, {
    ...baseHeaders(headers['Content-Type']),
    'Content-Length': data.length,
    ...headers,
  });
  res.end(data);
}

function redirect(res, location, cookies = []) {
  const headers = { Location: location };
  if (cookies.length) headers['Set-Cookie'] = cookies;
  res.writeHead(303, { ...baseHeaders(), ...headers });
  res.end();
}

async function readBody(req, maxBytes = 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) {
      const error = new Error('Request body too large.');
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function parseForm(req) {
  const body = await readBody(req);
  return Object.fromEntries(new URLSearchParams(body));
}

function csrfInput(context) {
  return `<input type="hidden" name="csrf_token" value="${escapeHtml(context.csrf_token)}">`;
}

function validateCsrf(context, token) {
  if (!context || !token || !timingSafeTextEqual(context.csrf_token, token)) {
    const error = new Error('Invalid CSRF token.');
    error.statusCode = 403;
    throw error;
  }
}

function progress(context) {
  const total = context.pairwise_count + context.single_count;
  const done = Math.min(context.current_index, total);
  return {
    total,
    done,
    percent: total ? Math.round((done / total) * 100) : 0,
  };
}

function progressMarkup(context) {
  if (!context || !['pairwise', 'single'].includes(context.current_stage)) return '';
  const state = progress(context);
  return `
    <div class="progress-wrap" aria-label="实验进度">
      <div class="progress-copy"><span>正式任务进度</span><strong>${state.done} / ${state.total}</strong></div>
      <div class="progress-track"><span style="width:${state.percent}%"></span></div>
    </div>`;
}

function layout({ title, body, context = null, script = null, wide = false, admin = false }) {
  const studyTitle = escapeHtml(bundle.config.study_title);
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="csrf-token" content="${context ? escapeHtml(context.csrf_token) : ''}">
  <title>${escapeHtml(title)} · ${studyTitle}</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body data-page="${escapeHtml(script || '')}">
  <header class="site-header">
    <a class="brand" href="/">${studyTitle}</a>
    <div class="header-meta">${admin ? '研究管理端' : '匿名研究系统'}</div>
  </header>
  <main class="shell ${wide ? 'shell-wide' : ''}">
    ${progressMarkup(context)}
    ${body}
  </main>
  <footer class="site-footer">数据仅用于本研究；系统不收集姓名或驾照号码。</footer>
  <script src="/static/app.js" defer></script>
</body>
</html>`;
}

function card(title, content, extraClass = '') {
  return `<section class="card ${extraClass}"><h1>${title}</h1>${content}</section>`;
}

function flashMarkup(message, kind = 'error') {
  return message ? `<div class="notice notice-${kind}" role="alert">${escapeHtml(message)}</div>` : '';
}

function fieldSelect(name, label, options, { required = true, help = '' } = {}) {
  return `<label class="field"><span>${escapeHtml(label)}${required ? '<em>*</em>' : ''}</span>
    <select name="${escapeHtml(name)}" ${required ? 'required' : ''}>
      <option value="">请选择</option>
      ${options.map(([value, text]) => `<option value="${escapeHtml(value)}">${escapeHtml(text)}</option>`).join('')}
    </select>${help ? `<small>${escapeHtml(help)}</small>` : ''}</label>`;
}

function likert(name, prompt, low, high, { required = true } = {}) {
  const buttons = Array.from({ length: 7 }, (_, index) => {
    const value = index + 1;
    return `<label class="scale-option"><input type="radio" name="${escapeHtml(name)}" value="${value}" ${required ? 'required' : ''}><span>${value}</span></label>`;
  }).join('');
  return `<fieldset class="rating"><legend>${escapeHtml(prompt)}</legend>
    <div class="scale-labels"><span>${escapeHtml(low)}</span><span>${escapeHtml(high)}</span></div>
    <div class="scale-options">${buttons}</div>
  </fieldset>`;
}

function videoUrl(videoPath) {
  if (videoPath.startsWith('demo:')) {
    return `/media/demo/${encodeURIComponent(videoPath.slice(5))}.mp4`;
  }
  if (videoPath.startsWith('/stimuli/')) return videoPath;
  return `/stimuli/${encodeURIComponent(videoPath)}`;
}

function mediaPanel(stimulus, label, id) {
  return `<div class="media-panel">
    <div class="media-label">${escapeHtml(label)}</div>
    <video id="${escapeHtml(id)}" class="stimulus-video" preload="auto" playsinline disablepictureinpicture controlslist="nodownload noplaybackrate nofullscreen">
      <source src="${escapeHtml(videoUrl(stimulus.video_path))}" type="video/mp4">
    </video>
  </div>`;
}

function routeForStage(context) {
  const map = {
    consent: '/consent',
    profile: '/profile',
    instructions: '/instructions',
    practice: '/practice',
    pairwise: '/trial',
    single: '/trial',
    post_survey: '/post-survey',
    complete: '/complete',
    declined: '/declined',
    screened_out: '/screened-out',
  };
  return map[context.current_stage] || '/';
}

function requireParticipant(req, res) {
  const context = participantContext(req);
  if (!context) {
    redirect(res, '/');
    return null;
  }
  return context;
}

function currentTrial(context) {
  return store.getCurrentTrial(context.session_id, context.current_index);
}

function parseInteger(value, minimum, maximum, fieldName) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
    const error = new Error(`Invalid ${fieldName}.`);
    error.statusCode = 400;
    throw error;
  }
  return parsed;
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = typeof value === 'object' ? JSON.stringify(value) : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function rowsToCsv(rows) {
  if (!rows.length) return '';
  const headers = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  return [
    headers.map(csvEscape).join(','),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(',')),
  ].join('\n');
}

function renderLanding(req, res) {
  const context = participantContext(req);
  const resume = context ? `<a class="button button-secondary" href="/resume">继续上次实验</a>` : '';
  const body = card('驾驶互动行为评价实验', `
    <p class="lead">您将观看若干匿名驾驶互动片段，并从另一辆车驾驶员的角度作出判断。</p>
    <ul class="summary-list">
      <li>预计用时：${escapeHtml(bundle.config.estimated_minutes || '25–35')} 分钟</li>
      <li>请使用电脑和稳定网络，并保持页面可见</li>
      <li>系统不会告诉您片段来自哪类驾驶主体</li>
    </ul>
    <form method="post" action="/start" class="actions">
      <button class="button button-primary" type="submit">开始实验</button>
      ${resume}
    </form>
  `, 'hero-card');
  send(res, 200, layout({ title: '开始', body }));
}

function renderConsent(context, res, message = '') {
  const body = card('知情同意', `
    ${flashMarkup(message)}
    <div class="prose">
      <p>本研究关注驾驶互动行为的主观感受。您将观看匿名、统一呈现的短视频，并回答偏好与评分问题。</p>
      <p>参加完全自愿。您可以随时退出。系统不收集姓名、手机号、身份证号或驾照号码。</p>
      <p>提交后，匿名回答将用于科研分析和汇总发表。点击“同意并继续”表示您已阅读以上说明、年满18岁并自愿参加。</p>
    </div>
    <form method="post" action="/consent" class="actions stacked">
      ${csrfInput(context)}
      <button class="button button-primary" name="decision" value="accept" type="submit">同意并继续</button>
      <button class="button button-text" name="decision" value="decline" type="submit">不同意，退出实验</button>
    </form>
    <p class="fine-print">同意书版本：${escapeHtml(bundle.config.consent_version)}</p>
  `);
  send(res, 200, layout({ title: '知情同意', body, context }));
}

function renderProfile(context, res, message = '') {
  const body = card('基本信息', `
    ${flashMarkup(message)}
    <p class="lead small">仅收集分析所需的最少驾驶背景信息。</p>
    <form method="post" action="/profile" class="form-grid">
      ${csrfInput(context)}
      ${fieldSelect('age_band', '年龄范围', [['18-24','18–24'],['25-34','25–34'],['35-44','35–44'],['45-54','45–54'],['55-64','55–64'],['65+','65岁及以上']])}
      ${fieldSelect('gender', '性别（可不回答）', [['female','女'],['male','男'],['other','其他'],['prefer_not','不愿回答']], { required: false })}
      ${fieldSelect('valid_licence', '目前是否持有有效机动车驾驶证', [['yes','是'],['no','否']])}
      ${fieldSelect('years_licensed', '持有驾驶证年限', [['lt1','不足1年'],['1-3','1–3年'],['4-7','4–7年'],['8-15','8–15年'],['gt15','15年以上']])}
      ${fieldSelect('driving_frequency', '过去一年驾驶频率', [['daily','几乎每天'],['weekly_multi','每周数次'],['weekly','每周一次'],['monthly_multi','每月数次'],['rarely','很少']])}
      ${fieldSelect('annual_mileage_band', '过去一年大致驾驶里程（可不回答）', [['lt1000','少于1,000 km'],['1000-5000','1,000–5,000 km'],['5001-10000','5,001–10,000 km'],['10001-20000','10,001–20,000 km'],['gt20000','超过20,000 km'],['unsure','不确定']], { required: false })}
      ${fieldSelect('urban_driving_frequency', '城市道路驾驶频率', [['never','从不'],['rarely','很少'],['sometimes','有时'],['often','经常'],['very_often','非常经常']])}
      ${fieldSelect('professional_driver', '是否从事职业驾驶（可不回答）', [['yes','是'],['no','否']], { required: false })}
      <input type="hidden" name="device_type" value="desktop-web">
      <div class="form-actions"><button class="button button-primary" type="submit">继续</button></div>
    </form>
  `);
  send(res, 200, layout({ title: '基本信息', body, context }));
}

function renderInstructions(context, res) {
  const body = card('任务说明', `
    <div class="instruction-grid">
      <div><span class="step-number">1</span><h2>代入灰色车辆</h2><p>请始终想象自己是灰色车辆的驾驶员。</p></div>
      <div><span class="step-number">2</span><h2>评价蓝色目标车</h2><p>视频中的蓝色车辆始终是需要评价的对象。</p></div>
      <div><span class="step-number">3</span><h2>完整观看</h2><p>每段视频完整播放后，问题才会出现；每个 trial 最多完整重播一次。</p></div>
    </div>
    <div class="notice notice-info">研究只关心您对行为本身的直观感受。请不要猜测车辆由谁驾驶。</div>
    <form method="post" action="/instructions" class="actions">
      ${csrfInput(context)}
      <button class="button button-primary" type="submit">进入练习</button>
    </form>
  `);
  send(res, 200, layout({ title: '任务说明', body, context }));
}

function renderPractice(context, res, error = '') {
  const practiceId = (bundle.config.practice_stimulus_ids || [])[0];
  const stimulus = publicStimulus(bundle, practiceId);
  const body = card('练习：确认任务对象', `
    ${flashMarkup(error)}
    <p class="lead small">请观看视频。您应代入灰色车辆，并评价蓝色车辆。</p>
    <div class="single-media">${mediaPanel(stimulus, '练习片段', 'practice-video')}</div>
    <div class="sequence-status" data-sequence-status>点击下方按钮播放练习片段。</div>
    <div class="actions" data-start-area><button class="button button-primary" type="button" data-start-practice>开始播放</button></div>
    <form method="post" action="/practice" class="response-form hidden" data-practice-form>
      ${csrfInput(context)}
      <input type="hidden" name="playback_complete" value="0" data-playback-a>
      <fieldset class="choice-group"><legend>刚才需要评价的是哪辆车？</legend>
        <label class="choice-card"><input type="radio" name="target_vehicle" value="BLUE" required><span>蓝色车辆</span></label>
        <label class="choice-card"><input type="radio" name="target_vehicle" value="GREY" required><span>灰色车辆</span></label>
        <label class="choice-card"><input type="radio" name="target_vehicle" value="UNSURE" required><span>不确定</span></label>
      </fieldset>
      <div class="actions"><button class="button button-primary" type="submit">提交练习</button></div>
    </form>
  `);
  send(res, 200, layout({ title: '练习', body, context, script: 'practice', wide: true }));
}

function renderPairwise(context, trial, res) {
  const a = publicStimulus(bundle, trial.stimulus_a_id);
  const b = publicStimulus(bundle, trial.stimulus_b_id);
  const body = card('成对比较', `
    <div class="trial-context"><strong>请代入灰色车辆驾驶员。</strong> 两段视频来自同一类情境，请比较蓝色目标车辆的处理方式。</div>
    <div class="sequence-stage" data-sequence-stage>
      <div class="media-grid pair-media">
        ${mediaPanel(a, 'Vehicle A', 'video-a')}
        ${mediaPanel(b, 'Vehicle B', 'video-b')}
      </div>
      <div class="sequence-status" data-sequence-status>点击下方按钮开始，系统将依次播放 A 和 B。</div>
      <div class="actions" data-start-area><button class="button button-primary" type="button" data-start-sequence>开始播放</button></div>
    </div>
    <form method="post" action="/trial" class="response-form hidden" data-response-form data-max-replays="${Number(bundle.config.max_replays_per_trial || 1)}">
      ${csrfInput(context)}
      <input type="hidden" name="trial_id" value="${escapeHtml(trial.trial_id)}">
      <input type="hidden" name="playback_a_complete" value="0" data-playback-a>
      <input type="hidden" name="playback_b_complete" value="0" data-playback-b>
      <input type="hidden" name="replay_count" value="0" data-replay-count>
      <input type="hidden" name="response_time_ms" value="0" data-response-time>
      <fieldset class="choice-group"><legend>如果您是灰色车辆驾驶员，更愿意与哪辆车这样互动？</legend>
        <label class="choice-card"><input type="radio" name="preference" value="A" required><span>Vehicle A</span></label>
        <label class="choice-card"><input type="radio" name="preference" value="B" required><span>Vehicle B</span></label>
        <label class="choice-card"><input type="radio" name="preference" value="NO_PREFERENCE" required><span>无明显偏好</span></label>
      </fieldset>
      <fieldset class="rating compact"><legend>您对该选择有多大把握？</legend>
        <div class="scale-labels"><span>1 完全没把握</span><span>5 非常有把握</span></div>
        <div class="scale-options five">
          ${[1,2,3,4,5].map((value) => `<label class="scale-option"><input type="radio" name="choice_confidence" value="${value}" required><span>${value}</span></label>`).join('')}
        </div>
      </fieldset>
      <label class="field"><span>主要判断理由（可选）</span><textarea name="free_text_reason" maxlength="1000" rows="3" placeholder="例如：是否容易预判、是否让另一方频繁调整……"></textarea></label>
      <div class="actions split-actions">
        <button class="button button-secondary" type="button" data-replay>完整重播 A 和 B</button>
        <button class="button button-primary" type="submit">提交并进入下一题</button>
      </div>
    </form>
  `, 'trial-card');
  send(res, 200, layout({ title: '成对比较', body, context, script: 'pairwise', wide: true }));
}

function renderSingle(context, trial, res) {
  const stimulus = publicStimulus(bundle, trial.stimulus_a_id);
  const body = card('单片段评价', `
    <div class="trial-context"><strong>请代入灰色车辆驾驶员，评价蓝色目标车辆。</strong></div>
    <div class="single-media">${mediaPanel(stimulus, '驾驶互动片段', 'video-a')}</div>
    <div class="sequence-status" data-sequence-status>点击下方按钮开始播放。</div>
    <div class="actions" data-start-area><button class="button button-primary" type="button" data-start-single>开始播放</button></div>
    <form method="post" action="/trial" class="response-form hidden" data-response-form data-max-replays="${Number(bundle.config.max_replays_per_trial || 1)}">
      ${csrfInput(context)}
      <input type="hidden" name="trial_id" value="${escapeHtml(trial.trial_id)}">
      <input type="hidden" name="playback_a_complete" value="0" data-playback-a>
      <input type="hidden" name="playback_b_complete" value="1" data-playback-b>
      <input type="hidden" name="replay_count" value="0" data-replay-count>
      <input type="hidden" name="response_time_ms" value="0" data-response-time>
      ${likert('acceptability', '在当前路况下，这辆车的处理方式是可以接受的。', '1 完全不可接受', '7 完全可以接受')}
      ${likert('comfort', '如果我是灰色车辆驾驶员，与这辆车互动会让我感到舒适。', '1 非常不舒适', '7 非常舒适')}
      ${likert('predictability', '这辆车接下来的行为容易预判。', '1 完全无法预判', '7 非常容易预判')}
      ${likert('interaction_burden', '这辆车让灰色车辆作出了不必要的额外减速、等待或调整。', '1 完全没有', '7 非常明显')}
      ${likert('perceived_unsafe', '这辆车的行为让我感到不安全。', '1 完全没有', '7 非常明显')}
      ${likert('too_aggressive', '这辆车表现得过于激进或强势。', '1 完全没有', '7 非常明显')}
      ${likert('too_cautious', '这辆车表现得过于谨慎、迟疑或让行过度。', '1 完全没有', '7 非常明显')}
      <fieldset class="rating compact"><legend>您对以上判断有多大把握？（可选）</legend>
        <div class="scale-labels"><span>1 完全没把握</span><span>5 非常有把握</span></div>
        <div class="scale-options five">${[1,2,3,4,5].map((value) => `<label class="scale-option"><input type="radio" name="rating_confidence" value="${value}"><span>${value}</span></label>`).join('')}</div>
      </fieldset>
      <label class="field"><span>主要判断理由（可选）</span><textarea name="free_text_reason" maxlength="1000" rows="3"></textarea></label>
      <div class="actions split-actions">
        <button class="button button-secondary" type="button" data-replay>完整重播</button>
        <button class="button button-primary" type="submit">提交并进入下一题</button>
      </div>
    </form>
  `, 'trial-card');
  send(res, 200, layout({ title: '单片段评价', body, context, script: 'single', wide: true }));
}

function renderBreak(context, res) {
  const body = card('休息一下', `
    <p class="lead">成对比较已完成。接下来是单片段评分，建议休息 1–2 分钟再继续。</p>
    <form method="post" action="/break" class="actions">
      ${csrfInput(context)}
      <button class="button button-primary" type="submit">开始单片段评分</button>
    </form>
  `);
  send(res, 200, layout({ title: '休息', body, context }));
}

function renderPostSurvey(context, res) {
  const body = card('实验后问卷', `
    <form method="post" action="/post-survey" class="form-grid">
      ${csrfInput(context)}
      ${fieldSelect('adas_experience', '使用过哪些驾驶辅助功能（可不回答）', [['none','没有'],['acc','ACC'],['lka','车道保持'],['parking','自动泊车'],['navigation','领航辅助'],['multiple','多种功能']], { required: false })}
      ${fieldSelect('av_ride_experience', '是否乘坐或体验过自动驾驶车辆（可不回答）', [['never','从未'],['1-2','1–2次'],['multiple','多次']], { required: false })}
      ${fieldSelect('study_familiarity', '实验前对 IPV 或驾驶社会性研究的熟悉程度', [['none','完全不了解'],['heard','听说过'],['familiar','较了解'],['involved','直接参与过相关研究']])}
      ${fieldSelect('task_difficulty', '完成本实验的难度', [['1','1 非常容易'],['2','2'],['3','3'],['4','4'],['5','5'],['6','6'],['7','7 非常困难']])}
      <label class="field field-wide"><span>您是否认为某些片段来自人类或自动驾驶？请简要说明（可选）</span><textarea name="source_guess" maxlength="1000" rows="3"></textarea></label>
      <label class="field field-wide"><span>您认为本实验主要想研究什么？（可选）</span><textarea name="hypothesis_guess" maxlength="1000" rows="3"></textarea></label>
      <label class="field field-wide"><span>其他意见（可选）</span><textarea name="open_feedback" maxlength="2000" rows="4"></textarea></label>
      <div class="form-actions"><button class="button button-primary" type="submit">提交并完成实验</button></div>
    </form>
  `);
  send(res, 200, layout({ title: '实验后问卷', body, context }));
}

function renderAdminLogin(req, res, message = '') {
  const body = card('研究管理端', `
    ${flashMarkup(message)}
    <form method="post" action="/admin/login" class="form-narrow">
      <label class="field"><span>管理员口令</span><input type="password" name="admin_token" autocomplete="current-password" required></label>
      <div class="actions"><button class="button button-primary" type="submit">登录</button></div>
    </form>
  `);
  send(res, 200, layout({ title: '管理端登录', body, admin: true }));
}

function renderAdminDashboard(req, res) {
  const summary = store.dashboardSummary();
  const body = `
    <section class="admin-hero"><div><p class="eyebrow">研究管理端</p><h1>${escapeHtml(bundle.config.study_title)}</h1><p>Study version: <code>${escapeHtml(bundle.version)}</code></p></div><a class="button button-text" href="/admin/logout">退出</a></section>
    <section class="metric-grid">
      <div class="metric"><span>参与者</span><strong>${summary.participants}</strong></div>
      <div class="metric"><span>已完成</span><strong>${summary.completed}</strong></div>
      <div class="metric"><span>Pairwise回答</span><strong>${summary.pairwise}</strong></div>
      <div class="metric"><span>Single回答</span><strong>${summary.single}</strong></div>
    </section>
    <section class="card"><h2>数据导出</h2><div class="export-grid">
      ${['participants','sessions','trials','pairwise','single','events'].map((name) => `<a class="export-link" href="/admin/export/${name}.csv">${name}.csv</a>`).join('')}
      <a class="export-link" href="/admin/export/all.json">all-data.json</a>
    </div></section>
    <section class="card"><h2>当前阶段分布</h2><table class="data-table"><thead><tr><th>阶段</th><th>会话数</th></tr></thead><tbody>${summary.sessions.map((row) => `<tr><td>${escapeHtml(row.current_stage)}</td><td>${row.n}</td></tr>`).join('')}</tbody></table></section>
    <section class="card"><h2>当前冻结输入</h2><dl class="definition-list"><dt>Study config</dt><dd><code>${escapeHtml(bundle.configPath)}</code></dd><dt>Stimulus manifest</dt><dd><code>${escapeHtml(bundle.manifestPath)}</code></dd><dt>Database</dt><dd><code>${escapeHtml(CONFIG.database)}</code></dd></dl></section>
  `;
  send(res, 200, layout({ title: '管理端', body, wide: true, admin: true }));
}

async function handleParticipantRoutes(req, res, url) {
  if (req.method === 'GET' && url.pathname === '/') return renderLanding(req, res);

  if (req.method === 'POST' && url.pathname === '/start') {
    const participantId = `P_${crypto.randomUUID().replaceAll('-', '').slice(0, 16)}`;
    const sessionId = `S_${crypto.randomUUID().replaceAll('-', '')}`;
    const csrfToken = crypto.randomBytes(24).toString('base64url');
    const assignment = buildAssignment(bundle, participantId).map((trial) => ({
      ...trial,
      trial_id: `T_${crypto.randomUUID().replaceAll('-', '')}`,
    }));
    const assignmentHash = sha256(JSON.stringify(assignment));
    store.createParticipantAndSession({
      participantId,
      sessionId,
      csrfToken,
      studyId: bundle.config.study_id,
      studyVersion: bundle.version,
      assignment,
      assignmentHash,
    });
    store.logEvent({ sessionId, participantId, eventType: 'session_created', payload: { study_version: bundle.version } });
    return redirect(res, '/consent', [cookieHeader('exp_session', signedValue(sessionId))]);
  }

  if (req.method === 'GET' && url.pathname === '/resume') {
    const context = participantContext(req);
    return redirect(res, context ? routeForStage(context) : '/');
  }

  if (url.pathname === '/consent') {
    const context = requireParticipant(req, res); if (!context) return;
    if (req.method === 'GET') return renderConsent(context, res);
    if (req.method === 'POST') {
      const form = await parseForm(req); validateCsrf(context, form.csrf_token);
      const accepted = form.decision === 'accept';
      if (!accepted && form.decision !== 'decline') return renderConsent(context, res, '请选择是否同意。');
      store.recordConsent(context.session_id, bundle.config.consent_version, accepted);
      store.logEvent({ sessionId: context.session_id, participantId: context.participant_id, eventType: accepted ? 'consent_accepted' : 'consent_declined' });
      return redirect(res, accepted ? '/profile' : '/declined');
    }
  }

  if (url.pathname === '/profile') {
    const context = requireParticipant(req, res); if (!context) return;
    if (req.method === 'GET') return renderProfile(context, res);
    if (req.method === 'POST') {
      const form = await parseForm(req); validateCsrf(context, form.csrf_token);
      const required = ['age_band','valid_licence','years_licensed','driving_frequency','urban_driving_frequency'];
      if (required.some((name) => !form[name])) return renderProfile(context, res, '请填写所有必填信息。');
      const validLicence = form.valid_licence === 'yes';
      store.recordProfile(context.session_id, {
        age_band: form.age_band,
        gender: form.gender || '',
        valid_licence: validLicence,
        years_licensed: form.years_licensed,
        driving_frequency: form.driving_frequency,
        annual_mileage_band: form.annual_mileage_band || '',
        urban_driving_frequency: form.urban_driving_frequency,
        professional_driver: form.professional_driver || '',
        device_type: form.device_type || '',
      });
      store.logEvent({ sessionId: context.session_id, participantId: context.participant_id, eventType: validLicence ? 'screening_passed' : 'screening_failed' });
      return redirect(res, validLicence ? '/instructions' : '/screened-out');
    }
  }

  if (url.pathname === '/instructions') {
    const context = requireParticipant(req, res); if (!context) return;
    if (req.method === 'GET') return renderInstructions(context, res);
    if (req.method === 'POST') {
      const form = await parseForm(req); validateCsrf(context, form.csrf_token);
      store.updateStage(context.session_id, 'practice');
      return redirect(res, '/practice');
    }
  }

  if (url.pathname === '/practice') {
    const context = requireParticipant(req, res); if (!context) return;
    if (req.method === 'GET') return renderPractice(context, res, url.searchParams.get('error') ? '需要评价的是蓝色目标车辆，请再试一次。' : '');
    if (req.method === 'POST') {
      const form = await parseForm(req); validateCsrf(context, form.csrf_token);
      const passed = form.playback_complete === '1' && form.target_vehicle === 'BLUE';
      store.logEvent({ sessionId: context.session_id, participantId: context.participant_id, eventType: passed ? 'practice_passed' : 'practice_failed', payload: { answer: form.target_vehicle || '', playback_complete: form.playback_complete === '1' } });
      if (!passed) return redirect(res, '/practice?error=1');
      store.completePractice(context.session_id);
      return redirect(res, '/trial');
    }
  }

  if (url.pathname === '/trial') {
    let context = requireParticipant(req, res); if (!context) return;
    const total = context.pairwise_count + context.single_count;
    if (context.current_index >= total) {
      store.updateStage(context.session_id, 'post_survey');
      return redirect(res, '/post-survey');
    }
    if (context.current_index >= context.pairwise_count && !context.break_taken) {
      return redirect(res, '/break');
    }
    let trial = currentTrial(context);
    if (!trial) {
      store.updateStage(context.session_id, 'post_survey');
      return redirect(res, '/post-survey');
    }
    const expectedStage = trial.trial_type === 'pairwise' ? 'pairwise' : 'single';
    if (context.current_stage !== expectedStage) {
      store.updateStage(context.session_id, expectedStage);
      context = store.getContext(context.session_id);
    }
    trial = store.markTrialStarted(trial.trial_id);

    if (req.method === 'GET') {
      store.logEvent({ sessionId: context.session_id, participantId: context.participant_id, trialId: trial.trial_id, eventType: 'trial_viewed', payload: { trial_type: trial.trial_type } });
      return trial.trial_type === 'pairwise'
        ? renderPairwise(context, trial, res)
        : renderSingle(context, trial, res);
    }
    if (req.method === 'POST') {
      const form = await parseForm(req); validateCsrf(context, form.csrf_token);
      if (form.trial_id !== trial.trial_id) {
        const error = new Error('Submitted trial is not the current trial.'); error.statusCode = 409; throw error;
      }
      const playbackA = form.playback_a_complete === '1';
      const playbackB = form.playback_b_complete === '1';
      if (!playbackA || (trial.trial_type === 'pairwise' && !playbackB)) {
        const error = new Error('Videos must be watched completely before submission.'); error.statusCode = 400; throw error;
      }
      const replayCount = parseInteger(form.replay_count || '0', 0, Number(bundle.config.max_replays_per_trial || 1), 'replay count');
      const responseTimeMs = parseInteger(form.response_time_ms || '0', 0, 3600000, 'response time');

      if (trial.trial_type === 'pairwise') {
        if (!['A','B','NO_PREFERENCE'].includes(form.preference)) {
          const error = new Error('Invalid pairwise preference.'); error.statusCode = 400; throw error;
        }
        const confidence = parseInteger(form.choice_confidence, 1, 5, 'choice confidence');
        const preferredStimulusId = form.preference === 'A' ? trial.stimulus_a_id : form.preference === 'B' ? trial.stimulus_b_id : null;
        store.submitPairwise({
          sessionId: context.session_id,
          trialId: trial.trial_id,
          preferenceRaw: form.preference,
          preferredStimulusId,
          confidence,
          freeTextReason: String(form.free_text_reason || '').slice(0, 1000),
          playbackAComplete: playbackA,
          playbackBComplete: playbackB,
          replayCount,
          responseTimeMs,
        });
      } else {
        const ratings = {};
        ['acceptability','comfort','predictability','interaction_burden','perceived_unsafe','too_aggressive','too_cautious']
          .forEach((name) => { ratings[name] = parseInteger(form[name], 1, 7, name); });
        const confidence = form.rating_confidence ? parseInteger(form.rating_confidence, 1, 5, 'rating confidence') : null;
        store.submitSingle({
          sessionId: context.session_id,
          trialId: trial.trial_id,
          ratings,
          confidence,
          freeTextReason: String(form.free_text_reason || '').slice(0, 1000),
          playbackComplete: playbackA,
          replayCount,
          responseTimeMs,
        });
      }
      store.logEvent({ sessionId: context.session_id, participantId: context.participant_id, trialId: trial.trial_id, eventType: 'trial_submitted', payload: { trial_type: trial.trial_type, replay_count: replayCount, response_time_ms: responseTimeMs } });
      return redirect(res, '/trial');
    }
  }

  if (url.pathname === '/break') {
    const context = requireParticipant(req, res); if (!context) return;
    if (req.method === 'GET') return renderBreak(context, res);
    if (req.method === 'POST') {
      const form = await parseForm(req); validateCsrf(context, form.csrf_token);
      store.takeBreak(context.session_id);
      store.logEvent({ sessionId: context.session_id, participantId: context.participant_id, eventType: 'break_completed' });
      return redirect(res, '/trial');
    }
  }

  if (url.pathname === '/post-survey') {
    const context = requireParticipant(req, res); if (!context) return;
    if (req.method === 'GET') return renderPostSurvey(context, res);
    if (req.method === 'POST') {
      const form = await parseForm(req); validateCsrf(context, form.csrf_token);
      const taskDifficulty = parseInteger(form.task_difficulty, 1, 7, 'task difficulty');
      const postSurvey = {
        adas_experience: String(form.adas_experience || '').slice(0, 500),
        av_ride_experience: String(form.av_ride_experience || '').slice(0, 100),
        study_familiarity: String(form.study_familiarity || '').slice(0, 100),
        task_difficulty: taskDifficulty,
        source_guess: String(form.source_guess || '').slice(0, 1000),
        hypothesis_guess: String(form.hypothesis_guess || '').slice(0, 1000),
        open_feedback: String(form.open_feedback || '').slice(0, 2000),
      };
      store.finishSession(context.session_id, postSurvey);
      store.logEvent({ sessionId: context.session_id, participantId: context.participant_id, eventType: 'session_completed' });
      return redirect(res, '/complete');
    }
  }

  if (req.method === 'GET' && url.pathname === '/complete') {
    const context = requireParticipant(req, res); if (!context) return;
    const body = card('实验已完成', `<p class="lead">感谢您的参与。您的匿名编号为：</p><div class="completion-code">${escapeHtml(context.participant_id)}</div><p>请保存该编号，以便需要时联系研究团队。</p>`);
    return send(res, 200, layout({ title: '完成', body, context }));
  }
  if (req.method === 'GET' && url.pathname === '/declined') {
    return send(res, 200, layout({ title: '已退出', body: card('您已退出实验', '<p>没有进入正式实验任务。</p>') }));
  }
  if (req.method === 'GET' && url.pathname === '/screened-out') {
    return send(res, 200, layout({ title: '筛查结束', body: card('感谢您的关注', '<p>本轮研究招募对象为持有有效机动车驾驶证的成年人。</p>') }));
  }

  return false;
}

async function handleApi(req, res, url) {
  if (req.method !== 'POST' || url.pathname !== '/api/event') return false;
  const context = requireParticipant(req, res); if (!context) return true;
  const body = await readBody(req);
  let payload;
  try { payload = JSON.parse(body || '{}'); } catch { payload = {}; }
  validateCsrf(context, req.headers['x-csrf-token'] || payload.csrf_token);
  const allowed = new Set(['visibility_hidden','visibility_visible','video_play','video_pause','video_ended','video_error','replay_requested','form_shown','page_unload']);
  const eventType = String(payload.event_type || '').slice(0, 80);
  if (!allowed.has(eventType)) {
    const error = new Error('Unsupported event type.'); error.statusCode = 400; throw error;
  }
  if (eventType === 'visibility_hidden') store.incrementVisibilityLoss(context.session_id);
  store.logEvent({
    sessionId: context.session_id,
    participantId: context.participant_id,
    trialId: payload.trial_id ? String(payload.trial_id).slice(0, 100) : null,
    eventType,
    payload,
  });
  return send(res, 200, JSON.stringify({ ok: true }), { 'Content-Type': 'application/json; charset=utf-8' });
}

async function handleAdmin(req, res, url) {
  if (!url.pathname.startsWith('/admin')) return false;
  if (url.pathname === '/admin/login') {
    if (req.method === 'GET') return renderAdminLogin(req, res);
    if (req.method === 'POST') {
      const form = await parseForm(req);
      if (!timingSafeTextEqual(form.admin_token || '', CONFIG.adminToken)) {
        return renderAdminLogin(req, res, '管理员口令不正确。');
      }
      return redirect(res, '/admin', [cookieHeader('exp_admin', signedValue('admin'), { maxAge: 8 * 3600 })]);
    }
  }
  if (url.pathname === '/admin/logout') {
    return redirect(res, '/admin/login', [cookieHeader('exp_admin', '', { maxAge: 0 })]);
  }
  if (!isAdmin(req)) return redirect(res, '/admin/login');
  if (req.method === 'GET' && url.pathname === '/admin') return renderAdminDashboard(req, res);
  const csvMatch = url.pathname.match(/^\/admin\/export\/(participants|sessions|trials|pairwise|single|events)\.csv$/);
  if (req.method === 'GET' && csvMatch) {
    const name = csvMatch[1];
    const csv = `\uFEFF${rowsToCsv(store.exportRows(name))}`;
    return send(res, 200, csv, {
      'Content-Type': 'text/csv; charset=utf-8',
      'Content-Disposition': `attachment; filename="${name}.csv"`,
    });
  }
  if (req.method === 'GET' && url.pathname === '/admin/export/all.json') {
    const data = {
      exported_at: new Date().toISOString(),
      study: { id: bundle.config.study_id, version: bundle.version, config: bundle.config },
      stimuli: bundle.stimuli,
      participants: store.exportRows('participants'),
      sessions: store.exportRows('sessions'),
      trials: store.exportRows('trials'),
      pairwise: store.exportRows('pairwise'),
      single: store.exportRows('single'),
      events: store.exportRows('events'),
    };
    return send(res, 200, JSON.stringify(data, null, 2), {
      'Content-Type': 'application/json; charset=utf-8',
      'Content-Disposition': 'attachment; filename="all-data.json"',
    });
  }
  return false;
}

function handleStatic(req, res, url) {
  if (req.method !== 'GET') return false;
  if (url.pathname === '/static/app.css' || url.pathname === '/static/app.js') {
    const filename = url.pathname.endsWith('.css') ? 'app.css' : 'app.js';
    const filePath = path.join(ROOT, 'public', filename);
    const type = filename.endsWith('.css') ? 'text/css; charset=utf-8' : 'text/javascript; charset=utf-8';
    return send(res, 200, fs.readFileSync(filePath), { 'Content-Type': type, 'Cache-Control': 'public, max-age=300' });
  }
  const demoMatch = url.pathname.match(/^\/media\/demo\/(clip_[a-z0-9_]+)\.mp4$/);
  if (demoMatch) {
    const data = DEMO_MEDIA[demoMatch[1]];
    if (!data) return false;
    return send(res, 200, data, { 'Content-Type': 'video/mp4', 'Cache-Control': 'public, max-age=3600' });
  }
  if (url.pathname.startsWith('/stimuli/')) {
    const relative = decodeURIComponent(url.pathname.slice('/stimuli/'.length));
    const root = path.resolve(ROOT, 'public', 'stimuli');
    const filePath = path.resolve(root, relative);
    if (!filePath.startsWith(`${root}${path.sep}`) || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return false;
    const ext = path.extname(filePath).toLowerCase();
    const type = ext === '.webm' ? 'video/webm' : 'video/mp4';
    return send(res, 200, fs.readFileSync(filePath), { 'Content-Type': type, 'Cache-Control': 'private, max-age=3600' });
  }
  return false;
}

async function handler(req, res) {
  try {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    if (req.method === 'GET' && url.pathname === '/health') {
      return send(res, 200, JSON.stringify({
        status: 'ok',
        study_id: bundle.config.study_id,
        study_version: bundle.version,
        stimuli: bundle.stimuli.length,
      }), { 'Content-Type': 'application/json; charset=utf-8' });
    }
    if (handleStatic(req, res, url) !== false) return;
    if (await handleApi(req, res, url) !== false) return;
    if (await handleAdmin(req, res, url) !== false) return;
    if (await handleParticipantRoutes(req, res, url) !== false) return;
    return send(res, 404, layout({ title: '未找到页面', body: card('页面不存在', '<p><a href="/">返回实验首页</a></p>') }));
  } catch (error) {
    console.error(error);
    const status = Number(error.statusCode || 500);
    const message = status >= 500 ? '系统暂时无法处理该请求，请联系研究人员。' : error.message;
    return send(res, status, layout({ title: '发生错误', body: card('无法继续', `<div class="notice notice-error">${escapeHtml(message)}</div><p><a href="/resume">返回当前实验进度</a></p>`) }));
  }
}

function createServer() {
  return http.createServer(handler);
}

if (require.main === module) {
  const server = createServer();
  server.listen(CONFIG.port, CONFIG.host, () => {
    console.log(`Subjective experiment server: http://${CONFIG.host}:${CONFIG.port}`);
    console.log(`Study ${bundle.config.study_id} / ${bundle.version}; ${bundle.stimuli.length} stimuli.`);
  });
  const shutdown = () => server.close(() => { store.close(); process.exit(0); });
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

module.exports = {
  CONFIG,
  bundle,
  createServer,
  escapeHtml,
  handler,
  rowsToCsv,
  signedValue,
  store,
  verifySignedValue,
};
