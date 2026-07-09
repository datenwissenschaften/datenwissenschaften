<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import FitnessHistogram from './FitnessHistogram.vue'

const snapshot = ref({ episodes: [], metadata: {} })
const connected = ref(false)
const error = ref('')
const stateFilter = ref('')
const expandedEpisode = ref(null)
const showResetDialog = ref(false)
const resetting = ref(false)
const resetError = ref('')
const resetStartedAt = ref(null)
let timer

const load = async () => {
  try {
    const response = await fetch('/api/snapshot', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    snapshot.value = payload
    if (resetting.value && resetStartedAt.value && payload.started_at !== resetStartedAt.value) {
      resetting.value = false
      resetStartedAt.value = null
    }
    connected.value = true
    error.value = ''
  } catch (reason) {
    connected.value = false
    error.value = reason.message
  }
}

onMounted(() => { load(); timer = window.setInterval(load, 1500) })
onBeforeUnmount(() => window.clearInterval(timer))

const episodes = computed(() => snapshot.value.episodes || [])
const states = computed(() => [...new Set(episodes.value.map(row => row.training_state).filter(Boolean))]
  .sort((left, right) => left.localeCompare(right)))
watch(states, availableStates => {
  if (!availableStates.includes(stateFilter.value)) {
    stateFilter.value = episodes.value.map(row => row.training_state).filter(Boolean).at(-1) || ''
  }
}, { immediate: true })
const stateHistory = computed(() => episodes.value.filter(row => row.training_state === stateFilter.value))
const filtered = computed(() => stateHistory.value)
const reversed = computed(() => [...filtered.value].reverse())
const latest = computed(() => filtered.value.at(-1))
const summary = computed(() => snapshot.value.summary || {})
const stateSummary = computed(() => summary.value.by_state?.[stateFilter.value] || null)
const activeSummary = computed(() => stateSummary.value || summary.value)
const summarizedEpisodes = computed(() => Number(activeSummary.value.episodes) || 0)
const best = computed(() => activeSummary.value.best_fitness ?? null)
const wins = computed(() => Number(activeSummary.value.wins) || 0)
const winRate = computed(() => summarizedEpisodes.value ? wins.value / summarizedEpisodes.value * 100 : 0)
const timedEpisodes = computed(() => filtered.value.filter(row => Number.isFinite(Number(row.duration_seconds))))
const avgDuration = computed(() => timedEpisodes.value.length
  ? timedEpisodes.value.reduce((sum, row) => sum + Number(row.duration_seconds), 0) / timedEpisodes.value.length
  : null)
const summarizedAvgDuration = computed(() => {
  const timed = Number(activeSummary.value.timed_episodes) || 0
  return timed ? Number(activeSummary.value.duration_seconds_total) / timed : avgDuration.value
})
const fitnessValues = computed(() => stateHistory.value
  .filter(row => row.fitness != null)
  .map(row => Number(row.fitness))
  .filter(Number.isFinite))
const fitnessBucketCount = computed(() => fitnessValues.value.length
  ? Math.ceil(Math.log2(fitnessValues.value.length) + 1)
  : 1)
const model = computed(() => snapshot.value.metadata?.model || {})
const ppo = computed(() => model.value.ppo || {})
const rnd = computed(() => model.value.rnd || {})
const environment = computed(() => snapshot.value.metadata?.environment || {})
const runtimeDetails = computed(() => {
  const { class: _environmentClass, ...details } = environment.value
  return { class: model.value.class || 'Unknown', ...details }
})
const run = computed(() => snapshot.value.metadata?.run || {})
const savestateProgress = computed(() => snapshot.value.metadata?.savestate_progress || {})
const savestateProgressRows = computed(() => Object.entries(savestateProgress.value)
  .map(([state, value]) => ({ state, ...(value || {}) }))
  .sort((left, right) => left.state.localeCompare(right.state)))
const selectedSavestateProgress = computed(() => savestateProgress.value[stateFilter.value] || null)
const selectedBeatenCount = computed(() => Number(selectedSavestateProgress.value?.beaten_count) || 0)
const selectedBeatenThreshold = computed(() => Number(selectedSavestateProgress.value?.beaten_threshold) || Number(run.value.savestate_beaten_threshold) || 0)
const server = computed(() => snapshot.value.server || {})
const versionLabel = computed(() => server.value.version === 'DEVELOPMENT'
  ? 'DEVELOPMENT'
  : server.value.version ? `v${server.value.version}` : '—')
const activeAlgorithm = computed(() => entries(ppo.value).length ? 'ppo' : null)
const control = computed(() => snapshot.value.control || {})

const resetModel = async () => {
  resetting.value = true
  resetStartedAt.value = snapshot.value.started_at
  resetError.value = ''
  try {
    const response = await fetch('/api/model/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': control.value.csrf_token },
      body: JSON.stringify({ game: run.value.game }),
    })
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`)
    showResetDialog.value = false
  } catch (reason) {
    resetting.value = false
    resetError.value = reason.message
  }
}

const fmt = (value, digits = 0) => value == null ? '—' : Intl.NumberFormat('en', { maximumFractionDigits: digits }).format(value)
const duration = value => {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  const seconds = Math.max(0, Math.round(Number(value)))
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}
const display = value => {
  if (value == null || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (Array.isArray(value)) return value.join('\n')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
const entries = object => Object.entries(object || {}).filter(([, value]) => value != null && value !== 'None')
const label = key => key.replaceAll('_', ' ')
const ramEntries = row => Object.entries(row.ram || {})
const toggleRam = row => { expandedEpisode.value = expandedEpisode.value === row.index ? null : row.index }
</script>

<template>
  <main>
    <header class="topbar">
      <div class="brand">
        <div class="mark"><span></span><span></span><span></span></div>
        <div><p class="eyebrow">DATENWISSENSCHAFTEN · {{ versionLabel }}</p><h1>Training Observatory</h1></div>
      </div>
      <div class="run-summary">
        <span>{{ run.game || 'Awaiting run' }}</span>
        <span class="separator">/</span>
        <span>{{ latest?.training_state || run.savestate || 'no state' }}</span>
        <span class="endpoint">{{ server.bind_address || '—' }}</span>
        <span :class="['connection', { offline: !connected }]"><i></i>{{ connected ? 'Live' : 'Disconnected' }}</span>
      </div>
    </header>

    <section class="controls panel">
      <div><p class="eyebrow">OBSERVATION WINDOW</p><strong>Episode telemetry</strong></div>
      <label>State<select v-model="stateFilter"><option v-for="state in states" :key="state">{{ state }}</option></select></label>
      <button class="reset-button" :disabled="!control.restart_supported || control.reset_pending || resetting" @click="showResetDialog = true">
        {{ control.reset_pending || resetting ? 'Restarting…' : 'Delete model' }}
      </button>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section class="kpis">
      <article class="panel metric"><p>Best fitness</p><strong class="mint">{{ fmt(best, 2) }}</strong><small>observed episodes</small></article>
      <article class="panel metric"><p>Win rate</p><strong>{{ fmt(winRate, 1) }}<em>%</em></strong><small>{{ wins }} successful / {{ summarizedEpisodes }} episodes</small></article>
      <article class="panel metric"><p>Avg training time</p><strong>{{ duration(summarizedAvgDuration) }}</strong><small>{{ duration(latest?.duration_seconds) }} latest retained</small></article>
      <article class="panel metric"><p>Savestate beaten</p><strong>{{ selectedSavestateProgress ? `${fmt(selectedBeatenCount)} / ${fmt(selectedBeatenThreshold)}` : '—' }}</strong><small>{{ selectedSavestateProgress?.beaten ? 'threshold reached' : selectedSavestateProgress?.has_savestate ? 'automatic savestate active' : 'no automatic savestate yet' }}</small></article>
    </section>

    <section class="charts">
      <article class="panel chart-card wide">
        <div class="card-heading"><div><p class="eyebrow">REWARD SIGNAL</p><h2>Fitness distribution</h2></div><div class="legend"><i style="--color:#8cf5c6"></i>{{ fitnessBucketCount }} {{ fitnessBucketCount === 1 ? 'bucket' : 'buckets' }} · {{ fitnessValues.length }} episodes</div></div>
        <FitnessHistogram :values="fitnessValues" :bucket-count="fitnessBucketCount" />
      </article>
    </section>

    <section :class="['details-grid', { 'two-column': !entries(rnd).length }]">
      <article class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">RUNTIME</p><h2>Environment</h2></div><span class="chip">{{ environment.num_envs || run.configured_envs || '—' }} envs</span></div>
        <dl><template v-for="([key, value]) in entries(runtimeDetails)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
      </article>
      <article v-if="activeAlgorithm === 'ppo'" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">POLICY OPTIMIZATION</p><h2>{{ entries(rnd).length ? 'Recurrent PPO' : 'PPO' }}</h2></div><span class="chip" :class="{ muted: !entries(ppo).length }">{{ entries(ppo).length ? 'Configured' : 'Not active' }}</span></div>
        <dl v-if="entries(ppo).length"><template v-for="([key, value]) in entries(ppo)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
        <p v-else class="placeholder">No PPO parameters on the active model.</p>
      </article>
      <article v-if="activeAlgorithm === 'ppo' && entries(rnd).length" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">INTRINSIC EXPLORATION</p><h2>Random Network Distillation</h2></div><span class="chip">Active</span></div>
        <dl><template v-for="([key, value]) in entries(rnd)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
      </article>
      <article v-if="!activeAlgorithm" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">MODEL</p><h2>Algorithm</h2></div><span class="chip muted">Waiting</span></div>
        <p class="placeholder">Algorithm details appear when PPO starts.</p>
      </article>
      <article v-if="savestateProgressRows.length" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">AUTOMATIC SAVESTATES</p><h2>Beaten counts</h2></div><span class="chip">{{ fmt(selectedBeatenThreshold) }} target</span></div>
        <dl><template v-for="row in savestateProgressRows" :key="row.state"><dt>{{ row.state }}</dt><dd>{{ fmt(row.beaten_count) }} / {{ fmt(row.beaten_threshold) }} · {{ row.beaten ? 'beaten' : row.has_savestate ? 'saved' : 'pending' }}</dd></template></dl>
      </article>
    </section>

    <section class="panel episodes-card">
      <div class="card-heading"><div><p class="eyebrow">DIAGNOSTICS</p><h2>Recent episodes</h2></div><span class="count">{{ episodes.length }} retained / {{ summary.episodes || episodes.length }} total</span></div>
      <div class="table-scroll"><table><thead><tr><th>#</th><th>Training state</th><th>Fitness</th><th>Won</th><th>Final state</th><th>Details</th></tr></thead>
        <tbody><template v-for="row in reversed.slice(0, 100)" :key="row.index">
          <tr :class="{ expanded: expandedEpisode === row.index }"><td class="dim">{{ row.index }}</td><td><span class="state">{{ row.training_state }}</span></td><td class="fitness">{{ fmt(row.fitness, 2) }}</td><td><span :class="['status', row.won === true ? 'success' : 'neutral']">{{ row.won == null ? '—' : row.won ? 'Won' : 'No' }}</span></td><td>{{ row.final_state || '—' }}</td><td><button v-if="ramEntries(row).length" type="button" class="ram-toggle" :aria-expanded="expandedEpisode === row.index" @click="toggleRam(row)">{{ expandedEpisode === row.index ? 'Hide RAM' : 'Show RAM' }}</button><span v-else class="dim">—</span></td></tr>
          <tr v-if="expandedEpisode === row.index" class="ram-detail"><td colspan="6"><dl><template v-for="([key, value]) in ramEntries(row)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl></td></tr>
        </template>
        <tr v-if="!reversed.length"><td colspan="6" class="empty-row">Waiting for the evaluator to complete an episode.</td></tr></tbody>
      </table></div>
    </section>
    <footer>Local telemetry · refreshes every 1.5 seconds · {{ filtered.length }} retained / {{ summarizedEpisodes }} observed episodes</footer>

    <div v-if="showResetDialog" class="modal-backdrop" @click.self="showResetDialog = false">
      <section class="reset-dialog panel" role="dialog" aria-modal="true" aria-labelledby="reset-title">
        <p class="eyebrow danger-text">DESTRUCTIVE ACTION</p>
        <h2 id="reset-title">Delete {{ run.game }} model?</h2>
        <p>All checkpoints and model history for this game, plus all automatic savestates in the configured savestate directory, will be deleted. The active training run will stop and restart from zero.</p>
        <p v-if="resetError" class="error">{{ resetError }}</p>
        <div class="dialog-actions">
          <button class="cancel-button" :disabled="resetting" @click="showResetDialog = false">Cancel</button>
          <button class="confirm-reset" :disabled="resetting" @click="resetModel">{{ resetting ? 'Requesting…' : 'Delete and restart' }}</button>
        </div>
      </section>
    </div>
  </main>
</template>
