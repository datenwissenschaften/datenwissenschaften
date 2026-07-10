<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const snapshot = ref({ episodes: [], metadata: {} })
const connected = ref(false)
const error = ref('')
const stateFilter = ref('')
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
const latest = computed(() => episodes.value.at(-1))
const summary = computed(() => snapshot.value.summary || {})
const summarizedEpisodes = computed(() => Number(summary.value.episodes) || 0)
const best = computed(() => summary.value.best_fitness ?? null)
const wins = computed(() => Number(summary.value.wins) || 0)
const winRate = computed(() => summarizedEpisodes.value ? wins.value / summarizedEpisodes.value * 100 : 0)
const summarizedAvgDuration = computed(() => {
  const timed = Number(summary.value.timed_episodes) || 0
  return timed ? Number(summary.value.duration_seconds_total) / timed : null
})
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
        <dl><template v-for="row in savestateProgressRows" :key="row.state"><dt>{{ row.state }}</dt><dd>{{ fmt(row.beaten_count) }} / {{ fmt(row.beaten_threshold) }} · {{ row.beaten ? 'beaten' : row.has_savestate ? 'saved' : 'pending' }} · reward {{ fmt(row.reward_baseline, 2) }}</dd></template></dl>
      </article>
    </section>

    <footer>Local telemetry · refreshes every 1.5 seconds · {{ episodes.length }} retained / {{ summarizedEpisodes }} observed episodes</footer>

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
