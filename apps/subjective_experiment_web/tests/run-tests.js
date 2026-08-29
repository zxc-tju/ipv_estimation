'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

const {
  buildAssignment,
  deriveCondition,
  loadStudyBundle,
  parseCsv,
} = require('../lib/study');

function formEncode(values) {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => params.set(key, String(value)));
  return params.toString();
}

function csrf(html) {
  const match = html.match(/name="csrf_token" value="([^"]+)"/);
  assert.ok(match, 'CSRF field missing');
  return match[1];
}

function hiddenValue(html, name) {
  const match = html.match(new RegExp(`name="${name}" value="([^"]*)"`));
  assert.ok(match, `Hidden field ${name} missing`);
  return match[1];
}

function makeClient(base) {
  let cookie = '';
  return async function request(route, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (cookie) headers.Cookie = cookie;
    const response = await fetch(`${base}${route}`, {
      redirect: 'manual',
      ...options,
      headers,
    });
    const setCookie = response.headers.get('set-cookie');
    if (setCookie) cookie = setCookie.split(';')[0];
    const body = await response.text();
    return { response, body, cookie };
  };
}

async function waitForHealth(base, child) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (child.exitCode !== null) throw new Error(`Server exited with code ${child.exitCode}`);
    try {
      const response = await fetch(`${base}/health`);
      if (response.ok) return response.json();
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('Server did not become healthy.');
}

async function main() {
  const root = path.resolve(__dirname, '..');
  const bundle = loadStudyBundle(
    path.join(root, 'config', 'study.demo.json'),
    path.join(root, 'config', 'stimuli.demo.csv'),
  );
  assert.equal(bundle.stimuli.length, 19);
  assert.equal(deriveCondition({ stimulus_id:'x', actor_source:'HUMAN', verdict_class:'outside', deviation_side:'assertive' }), 'H_OUT_ASSERTIVE');
  assert.deepEqual(parseCsv('a,b\n"x,y",z\n'), [{ a: 'x,y', b: 'z', __row: 2 }]);
  const assignment1 = buildAssignment(bundle, 'P_FIXED');
  const assignment2 = buildAssignment(bundle, 'P_FIXED');
  assert.deepEqual(assignment1, assignment2, 'Assignment must be deterministic for one participant');
  assert.equal(assignment1.filter((x) => x.trial_type === 'pairwise').length, 4);
  assert.equal(assignment1.filter((x) => x.trial_type === 'single').length, 6);
  assert.ok(assignment1.slice(0, 4).every((x) => x.trial_type === 'pairwise'));
  assert.ok(assignment1.slice(4).every((x) => x.trial_type === 'single'));

  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'subjective-web-test-'));
  const database = path.join(temp, 'test.sqlite3');
  const port = 39000 + Math.floor(Math.random() * 1000);
  const base = `http://127.0.0.1:${port}`;
  const child = spawn(process.execPath, ['server.js'], {
    cwd: root,
    env: {
      ...process.env,
      PORT: String(port),
      HOST: '127.0.0.1',
      EXPERIMENT_DATABASE: database,
      EXPERIMENT_SECRET_KEY: 'test-secret-with-enough-entropy',
      EXPERIMENT_ADMIN_TOKEN: 'test-admin-token',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stderr = '';
  child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
  try {
    const health = await waitForHealth(base, child);
    assert.equal(health.status, 'ok');
    assert.equal(health.stimuli, 19);

    const client = makeClient(base);
    let result = await client('/', { method: 'GET' });
    assert.equal(result.response.status, 200);
    assert.match(result.body, /开始实验/);

    result = await client('/start', { method: 'POST' });
    assert.equal(result.response.status, 303);
    assert.equal(result.response.headers.get('location'), '/consent');
    assert.ok(result.cookie.startsWith('exp_session='));

    result = await client('/consent');
    const consentCsrf = csrf(result.body);
    const bad = await client('/consent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formEncode({ csrf_token: 'wrong', decision: 'accept' }),
    });
    assert.equal(bad.response.status, 403, 'CSRF rejection expected');

    result = await client('/consent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formEncode({ csrf_token: consentCsrf, decision: 'accept' }),
    });
    assert.equal(result.response.headers.get('location'), '/profile');

    result = await client('/profile');
    result = await client('/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formEncode({
        csrf_token: csrf(result.body),
        age_band: '25-34',
        gender: 'prefer_not',
        valid_licence: 'yes',
        years_licensed: '4-7',
        driving_frequency: 'weekly_multi',
        annual_mileage_band: '5001-10000',
        urban_driving_frequency: 'often',
        professional_driver: 'no',
        device_type: 'desktop-web',
      }),
    });
    assert.equal(result.response.headers.get('location'), '/instructions');

    result = await client('/instructions');
    result = await client('/instructions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formEncode({ csrf_token: csrf(result.body) }),
    });
    assert.equal(result.response.headers.get('location'), '/practice');

    result = await client('/practice');
    assert.doesNotMatch(result.body, /actor_source|verdict_class|H_OUT_ASSERTIVE/);
    result = await client('/practice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formEncode({ csrf_token: csrf(result.body), playback_complete: '1', target_vehicle: 'BLUE' }),
    });
    assert.equal(result.response.headers.get('location'), '/trial');

    let pairwiseSeen = 0;
    let singleSeen = 0;
    let completed = false;
    for (let guard = 0; guard < 30 && !completed; guard += 1) {
      result = await client('/trial');
      if (result.response.status === 303) {
        const location = result.response.headers.get('location');
        if (location === '/break') {
          const breakPage = await client('/break');
          result = await client('/break', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formEncode({ csrf_token: csrf(breakPage.body) }),
          });
          assert.equal(result.response.headers.get('location'), '/trial');
          continue;
        }
        if (location === '/post-survey') {
          completed = true;
          break;
        }
      }
      assert.equal(result.response.status, 200);
      assert.doesNotMatch(result.body, /actor_source|verdict_class|deviation_side|H_OUT_ASSERTIVE|AV_OUT_ASSERTIVE/);
      const trialId = hiddenValue(result.body, 'trial_id');
      const token = csrf(result.body);
      if (result.body.includes('data-start-sequence')) {
        pairwiseSeen += 1;
        result = await client('/trial', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formEncode({
            csrf_token: token,
            trial_id: trialId,
            playback_a_complete: '1',
            playback_b_complete: '1',
            replay_count: '0',
            response_time_ms: '850',
            preference: pairwiseSeen % 3 === 0 ? 'NO_PREFERENCE' : 'A',
            choice_confidence: '4',
            free_text_reason: 'test',
          }),
        });
      } else {
        singleSeen += 1;
        result = await client('/trial', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formEncode({
            csrf_token: token,
            trial_id: trialId,
            playback_a_complete: '1',
            playback_b_complete: '1',
            replay_count: '0',
            response_time_ms: '1250',
            acceptability: '5',
            comfort: '5',
            predictability: '6',
            interaction_burden: '3',
            perceived_unsafe: '2',
            too_aggressive: '2',
            too_cautious: '3',
            rating_confidence: '4',
            free_text_reason: 'test',
          }),
        });
      }
      assert.equal(result.response.status, 303);
      assert.equal(result.response.headers.get('location'), '/trial');
    }
    assert.equal(pairwiseSeen, 4);
    assert.equal(singleSeen, 6);

    result = await client('/post-survey');
    result = await client('/post-survey', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formEncode({
        csrf_token: csrf(result.body),
        adas_experience: 'acc',
        av_ride_experience: 'never',
        study_familiarity: 'none',
        task_difficulty: '3',
        source_guess: '',
        hypothesis_guess: '',
        open_feedback: 'ok',
      }),
    });
    assert.equal(result.response.headers.get('location'), '/complete');
    result = await client('/complete');
    assert.match(result.body, /实验已完成/);

    const video = await fetch(`${base}/media/demo/clip_01.mp4`);
    assert.equal(video.status, 200);
    const videoBuffer = Buffer.from(await video.arrayBuffer());
    assert.equal(videoBuffer.subarray(4, 8).toString('ascii'), 'ftyp');

    const admin = makeClient(base);
    result = await admin('/admin/login');
    assert.match(result.body, /研究管理端/);
    result = await admin('/admin/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formEncode({ admin_token: 'test-admin-token' }),
    });
    assert.equal(result.response.headers.get('location'), '/admin');
    result = await admin('/admin');
    assert.match(result.body, /Pairwise回答/);
    assert.match(result.body, />4</);
    result = await admin('/admin/export/pairwise.csv');
    assert.equal(result.response.status, 200);
    assert.match(result.body, /preference_raw/);
    result = await admin('/admin/export/single.csv');
    assert.match(result.body, /acceptability/);

    console.log('PASS: unit checks, complete participant flow, hidden-condition check, media and admin exports.');
  } finally {
    child.kill('SIGTERM');
    await new Promise((resolve) => child.once('exit', resolve));
    fs.rmSync(temp, { recursive: true, force: true });
    if (child.exitCode && child.exitCode !== 0) console.error(stderr);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
