'use strict';

const fs = require('node:fs');
const crypto = require('node:crypto');
const path = require('node:path');

const REQUIRED_COLUMNS = [
  'stimulus_id',
  'matched_set_id',
  'scenario_id',
  'actor_source',
  'verdict_class',
  'deviation_side',
  'video_path',
  'is_practice',
  'estimability_pass',
  'human_support_pass',
  'stimulus_qc_pass',
];

const VALID_CONDITIONS = new Set([
  'H_IN',
  'H_OUT_ASSERTIVE',
  'H_OUT_ACCOMMODATING',
  'AV_IN',
  'AV_OUT_ASSERTIVE',
  'AV_OUT_ACCOMMODATING',
]);

class StudyConfigurationError extends Error {}

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ',') {
      row.push(field);
      field = '';
    } else if (char === '\n') {
      row.push(field.replace(/\r$/, ''));
      if (row.some((item) => item.trim() !== '')) rows.push(row);
      row = [];
      field = '';
    } else {
      field += char;
    }
  }
  if (quoted) throw new StudyConfigurationError('CSV contains an unclosed quoted field.');
  row.push(field.replace(/\r$/, ''));
  if (row.some((item) => item.trim() !== '')) rows.push(row);
  if (rows.length === 0) throw new StudyConfigurationError('Stimulus CSV is empty.');

  const headers = rows[0].map((item) => item.replace(/^\uFEFF/, '').trim());
  const duplicates = headers.filter((item, index) => headers.indexOf(item) !== index);
  if (duplicates.length) {
    throw new StudyConfigurationError(`Duplicate CSV columns: ${[...new Set(duplicates)].join(', ')}`);
  }
  return rows.slice(1).map((values, index) => {
    const result = {};
    headers.forEach((header, column) => {
      result[header] = (values[column] ?? '').trim();
    });
    result.__row = index + 2;
    return result;
  });
}

function asBool(value) {
  return ['1', 'true', 'yes', 'y', 'pass'].includes(String(value).trim().toLowerCase());
}

function deriveCondition(stimulus) {
  const source = String(stimulus.actor_source).trim().toUpperCase();
  const verdict = String(stimulus.verdict_class).trim().toLowerCase();
  const side = String(stimulus.deviation_side).trim().toLowerCase();
  if (!['HUMAN', 'AV'].includes(source)) {
    throw new StudyConfigurationError(`Invalid actor_source for ${stimulus.stimulus_id}: ${source}`);
  }
  const prefix = source === 'HUMAN' ? 'H' : 'AV';
  if (verdict === 'inside') return `${prefix}_IN`;
  if (verdict !== 'outside') {
    throw new StudyConfigurationError(`Invalid verdict_class for ${stimulus.stimulus_id}: ${verdict}`);
  }
  if (side === 'assertive') return `${prefix}_OUT_ASSERTIVE`;
  if (side === 'accommodating') return `${prefix}_OUT_ACCOMMODATING`;
  throw new StudyConfigurationError(`Invalid deviation_side for ${stimulus.stimulus_id}: ${side}`);
}

function loadStudyBundle(configPath, manifestPath) {
  const configText = fs.readFileSync(configPath, 'utf8');
  const manifestText = fs.readFileSync(manifestPath, 'utf8');
  const config = JSON.parse(configText);
  const rawRows = parseCsv(manifestText);
  const headers = rawRows.length ? Object.keys(rawRows[0]) : [];
  const missingColumns = REQUIRED_COLUMNS.filter((column) => !headers.includes(column));
  if (missingColumns.length) {
    throw new StudyConfigurationError(`Stimulus CSV is missing: ${missingColumns.join(', ')}`);
  }

  const stimuli = rawRows.map((row) => {
    if (!row.stimulus_id) throw new StudyConfigurationError(`Blank stimulus_id on row ${row.__row}.`);
    const stimulus = {
      ...row,
      is_practice: asBool(row.is_practice),
      estimability_pass: asBool(row.estimability_pass),
      human_support_pass: asBool(row.human_support_pass),
      stimulus_qc_pass: asBool(row.stimulus_qc_pass),
    };
    stimulus.condition = deriveCondition(stimulus);
    return stimulus;
  });

  const ids = stimuli.map((item) => item.stimulus_id);
  const duplicateIds = ids.filter((item, index) => ids.indexOf(item) !== index);
  if (duplicateIds.length) {
    throw new StudyConfigurationError(`Duplicate stimulus IDs: ${[...new Set(duplicateIds)].join(', ')}`);
  }

  const version = sha256(`${configText}\n${manifestText}`).slice(0, 16);
  const bundle = {
    config,
    stimuli,
    byId: new Map(stimuli.map((item) => [item.stimulus_id, item])),
    version,
    configPath: path.resolve(configPath),
    manifestPath: path.resolve(manifestPath),
  };
  validateBundle(bundle);
  return bundle;
}

function eligibleFormal(item) {
  return !item.is_practice
    && item.estimability_pass
    && item.human_support_pass
    && item.stimulus_qc_pass;
}

function validateBundle(bundle) {
  const config = bundle.config;
  ['study_id', 'study_title', 'consent_version', 'pairwise_comparisons', 'single_counts']
    .forEach((key) => {
      if (config[key] === undefined) throw new StudyConfigurationError(`Study config is missing ${key}.`);
    });

  if (!Array.isArray(config.pairwise_comparisons)) {
    throw new StudyConfigurationError('pairwise_comparisons must be an array.');
  }
  const formal = bundle.stimuli.filter(eligibleFormal);
  const availableConditions = new Set(formal.map((item) => item.condition));
  const requiredConditions = new Set(Object.keys(config.single_counts));
  config.pairwise_comparisons.forEach((comparison) => {
    requiredConditions.add(comparison.condition_a);
    requiredConditions.add(comparison.condition_b);
    if (!Number.isInteger(comparison.count) || comparison.count < 0) {
      throw new StudyConfigurationError(`Invalid count in comparison ${comparison.id || 'unnamed'}.`);
    }
  });
  [...requiredConditions].forEach((condition) => {
    if (!VALID_CONDITIONS.has(condition)) {
      throw new StudyConfigurationError(`Unknown condition in config: ${condition}`);
    }
    if (!availableConditions.has(condition)) {
      throw new StudyConfigurationError(`No eligible formal stimulus for ${condition}.`);
    }
  });

  const bySet = new Map();
  formal.forEach((item) => {
    if (!bySet.has(item.matched_set_id)) bySet.set(item.matched_set_id, new Set());
    bySet.get(item.matched_set_id).add(item.condition);
  });
  config.pairwise_comparisons.forEach((comparison) => {
    const support = [...bySet.values()].some(
      (conditions) => conditions.has(comparison.condition_a) && conditions.has(comparison.condition_b),
    );
    if (!support) {
      throw new StudyConfigurationError(
        `No matched set supports ${comparison.condition_a} vs ${comparison.condition_b}.`,
      );
    }
  });

  const practiceIds = config.practice_stimulus_ids || [];
  practiceIds.forEach((id) => {
    const item = bundle.byId.get(id);
    if (!item || !item.is_practice) {
      throw new StudyConfigurationError(`Practice stimulus ${id} is missing or not marked practice.`);
    }
  });
}

function createPrng(seedText) {
  let seed = Number.parseInt(sha256(seedText).slice(0, 8), 16) >>> 0;
  return function random() {
    seed += 0x6D2B79F5;
    let value = seed;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(items, random) {
  const output = [...items];
  for (let i = output.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [output[i], output[j]] = [output[j], output[i]];
  }
  return output;
}

function choose(items, random) {
  if (!items.length) throw new StudyConfigurationError('No eligible item available for assignment.');
  return items[Math.floor(random() * items.length)];
}

function buildAssignment(bundle, participantId) {
  const { config } = bundle;
  const random = createPrng(
    `${config.global_seed || 20260829}|${bundle.version}|${participantId}`,
  );
  const formal = bundle.stimuli.filter(eligibleFormal);
  const bySet = new Map();
  formal.forEach((item) => {
    if (!bySet.has(item.matched_set_id)) bySet.set(item.matched_set_id, []);
    bySet.get(item.matched_set_id).push(item);
  });

  const used = new Set();
  const pairwise = [];
  config.pairwise_comparisons.forEach((comparison) => {
    for (let repetition = 0; repetition < comparison.count; repetition += 1) {
      const supportedSets = [...bySet.entries()].filter(([, items]) => {
        const conditions = new Set(items.map((item) => item.condition));
        return conditions.has(comparison.condition_a) && conditions.has(comparison.condition_b);
      });
      const setEntry = choose(shuffle(supportedSets, random), random);
      const [matchedSetId, items] = setEntry;
      const candidatesA = items.filter((item) => item.condition === comparison.condition_a);
      const candidatesB = items.filter((item) => item.condition === comparison.condition_b);
      const freshA = candidatesA.filter((item) => !used.has(item.stimulus_id));
      const freshB = candidatesB.filter((item) => !used.has(item.stimulus_id));
      const stimulusA = choose(freshA.length ? freshA : candidatesA, random);
      const stimulusB = choose(freshB.length ? freshB : candidatesB, random);
      used.add(stimulusA.stimulus_id);
      used.add(stimulusB.stimulus_id);
      const swap = random() < 0.5;
      pairwise.push({
        trial_type: 'pairwise',
        block: 'pairwise',
        comparison_id: comparison.id || `${comparison.condition_a}_vs_${comparison.condition_b}`,
        matched_set_id: matchedSetId,
        stimulus_a_id: swap ? stimulusB.stimulus_id : stimulusA.stimulus_id,
        stimulus_b_id: swap ? stimulusA.stimulus_id : stimulusB.stimulus_id,
      });
    }
  });

  const single = [];
  Object.entries(config.single_counts).forEach(([condition, count]) => {
    const candidates = formal.filter((item) => item.condition === condition);
    for (let repetition = 0; repetition < count; repetition += 1) {
      const fresh = candidates.filter((item) => !used.has(item.stimulus_id));
      const stimulus = choose(fresh.length ? fresh : candidates, random);
      used.add(stimulus.stimulus_id);
      single.push({
        trial_type: 'single',
        block: 'single',
        stimulus_a_id: stimulus.stimulus_id,
        stimulus_b_id: null,
      });
    }
  });

  return [...shuffle(pairwise, random), ...shuffle(single, random)].map((trial, index) => ({
    ...trial,
    sequence_index: index,
  }));
}

function publicStimulus(bundle, stimulusId) {
  const item = bundle.byId.get(stimulusId);
  if (!item) throw new StudyConfigurationError(`Unknown stimulus ID: ${stimulusId}`);
  return {
    stimulus_id: item.stimulus_id,
    video_path: item.video_path,
    target_label: item.target_label || '蓝色车辆',
    context_text: item.context_text || '请代入灰色车辆驾驶员，评价蓝色目标车辆。',
  };
}

module.exports = {
  REQUIRED_COLUMNS,
  VALID_CONDITIONS,
  StudyConfigurationError,
  asBool,
  buildAssignment,
  deriveCondition,
  eligibleFormal,
  loadStudyBundle,
  parseCsv,
  publicStimulus,
  sha256,
  validateBundle,
};
