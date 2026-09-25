'use strict';

(() => {
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';
  const page = document.body.dataset.page || '';
  const currentTrialId = document.querySelector('input[name="trial_id"]')?.value || null;

  function postEvent(eventType, detail = {}) {
    if (!csrf) return;
    const payload = JSON.stringify({ event_type: eventType, trial_id: currentTrialId, ...detail });
    fetch('/api/event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrf },
      body: payload,
      credentials: 'same-origin',
      keepalive: true,
    }).catch(() => {});
  }

  document.addEventListener('visibilitychange', () => {
    postEvent(document.hidden ? 'visibility_hidden' : 'visibility_visible', { hidden: document.hidden });
  });
  window.addEventListener('pagehide', () => postEvent('page_unload'));

  function attachVideoEvents(video, label) {
    if (!video) return;
    video.addEventListener('play', () => postEvent('video_play', { video: label, current_time: video.currentTime }));
    video.addEventListener('pause', () => {
      if (!video.ended) postEvent('video_pause', { video: label, current_time: video.currentTime });
    });
    video.addEventListener('ended', () => postEvent('video_ended', { video: label }));
    video.addEventListener('error', () => postEvent('video_error', { video: label, code: video.error?.code || null }));
    video.addEventListener('contextmenu', (event) => event.preventDefault());
  }

  function playFromStart(video) {
    video.pause();
    video.currentTime = 0;
    return video.play();
  }

  function revealForm(form, status, message) {
    form.classList.remove('hidden');
    if (status) status.textContent = message;
    const shownAt = performance.now();
    form.dataset.shownAt = String(shownAt);
    postEvent('form_shown');
    form.querySelector('input, button, textarea')?.focus({ preventScroll: true });
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function bindSubmissionTiming(form) {
    if (!form) return;
    form.addEventListener('submit', () => {
      const start = Number(form.dataset.shownAt || performance.now());
      const elapsed = Math.max(0, Math.round(performance.now() - start));
      const field = form.querySelector('[data-response-time]');
      if (field) field.value = String(elapsed);
      form.querySelectorAll('button').forEach((button) => { button.disabled = true; });
    });
  }

  function setupPairwise() {
    const videoA = document.getElementById('video-a');
    const videoB = document.getElementById('video-b');
    const startButton = document.querySelector('[data-start-sequence]');
    const startArea = document.querySelector('[data-start-area]');
    const form = document.querySelector('[data-response-form]');
    const status = document.querySelector('[data-sequence-status]');
    const replayButton = document.querySelector('[data-replay]');
    const replayField = form?.querySelector('[data-replay-count]');
    const playbackA = form?.querySelector('[data-playback-a]');
    const playbackB = form?.querySelector('[data-playback-b]');
    const maxReplays = Number(form?.dataset.maxReplays || 1);
    let running = false;
    let replayCount = 0;

    attachVideoEvents(videoA, 'A');
    attachVideoEvents(videoB, 'B');
    if (!videoA || !videoB || !startButton || !form) return;

    async function runSequence(isReplay = false) {
      if (running) return;
      running = true;
      form.classList.add('hidden');
      if (status) status.textContent = '正在播放 Vehicle A…';
      videoA.classList.add('active-video');
      videoB.classList.remove('active-video');
      try {
        await playFromStart(videoA);
        await new Promise((resolve) => videoA.addEventListener('ended', resolve, { once: true }));
        if (playbackA) playbackA.value = '1';
        if (status) status.textContent = '请稍候，接下来播放 Vehicle B…';
        await new Promise((resolve) => setTimeout(resolve, 600));
        videoA.classList.remove('active-video');
        videoB.classList.add('active-video');
        if (status) status.textContent = '正在播放 Vehicle B…';
        await playFromStart(videoB);
        await new Promise((resolve) => videoB.addEventListener('ended', resolve, { once: true }));
        if (playbackB) playbackB.value = '1';
        videoB.classList.remove('active-video');
        revealForm(form, status, '两段视频已播放完毕，请作答。');
        if (isReplay) postEvent('replay_requested', { replay_count: replayCount });
      } catch (error) {
        if (status) status.textContent = '视频播放失败，请刷新页面或联系研究人员。';
        postEvent('video_error', { message: String(error) });
      } finally {
        running = false;
      }
    }

    startButton.addEventListener('click', () => {
      startArea?.classList.add('hidden');
      runSequence(false);
    });
    replayButton?.addEventListener('click', () => {
      if (replayCount >= maxReplays || running) return;
      replayCount += 1;
      if (replayField) replayField.value = String(replayCount);
      if (replayCount >= maxReplays) replayButton.disabled = true;
      runSequence(true);
    });
    bindSubmissionTiming(form);
  }

  function setupSingle() {
    const video = document.getElementById('video-a');
    const startButton = document.querySelector('[data-start-single]');
    const startArea = document.querySelector('[data-start-area]');
    const form = document.querySelector('[data-response-form]');
    const status = document.querySelector('[data-sequence-status]');
    const replayButton = document.querySelector('[data-replay]');
    const replayField = form?.querySelector('[data-replay-count]');
    const playbackA = form?.querySelector('[data-playback-a]');
    const maxReplays = Number(form?.dataset.maxReplays || 1);
    let running = false;
    let replayCount = 0;

    attachVideoEvents(video, 'single');
    if (!video || !startButton || !form) return;

    async function runVideo(isReplay = false) {
      if (running) return;
      running = true;
      form.classList.add('hidden');
      if (status) status.textContent = '正在播放…';
      video.classList.add('active-video');
      try {
        await playFromStart(video);
        await new Promise((resolve) => video.addEventListener('ended', resolve, { once: true }));
        if (playbackA) playbackA.value = '1';
        video.classList.remove('active-video');
        revealForm(form, status, '视频已播放完毕，请作答。');
        if (isReplay) postEvent('replay_requested', { replay_count: replayCount });
      } catch (error) {
        if (status) status.textContent = '视频播放失败，请刷新页面或联系研究人员。';
        postEvent('video_error', { message: String(error) });
      } finally {
        running = false;
      }
    }

    startButton.addEventListener('click', () => {
      startArea?.classList.add('hidden');
      runVideo(false);
    });
    replayButton?.addEventListener('click', () => {
      if (replayCount >= maxReplays || running) return;
      replayCount += 1;
      if (replayField) replayField.value = String(replayCount);
      if (replayCount >= maxReplays) replayButton.disabled = true;
      runVideo(true);
    });
    bindSubmissionTiming(form);
  }

  function setupPractice() {
    const video = document.getElementById('practice-video');
    const startButton = document.querySelector('[data-start-practice]');
    const startArea = document.querySelector('[data-start-area]');
    const form = document.querySelector('[data-practice-form]');
    const status = document.querySelector('[data-sequence-status]');
    const playback = form?.querySelector('[data-playback-a]');
    attachVideoEvents(video, 'practice');
    if (!video || !startButton || !form) return;
    startButton.addEventListener('click', async () => {
      startArea?.classList.add('hidden');
      if (status) status.textContent = '正在播放练习片段…';
      try {
        await playFromStart(video);
        await new Promise((resolve) => video.addEventListener('ended', resolve, { once: true }));
        if (playback) playback.value = '1';
        revealForm(form, status, '练习片段已播放完毕，请回答。');
      } catch (error) {
        if (status) status.textContent = '视频播放失败，请刷新页面或联系研究人员。';
      }
    });
  }

  if (page === 'pairwise') setupPairwise();
  if (page === 'single') setupSingle();
  if (page === 'practice') setupPractice();
})();
