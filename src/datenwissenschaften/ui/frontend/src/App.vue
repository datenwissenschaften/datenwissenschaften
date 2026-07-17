<script setup>
import hljs from 'highlight.js/lib/core'
import dockerfile from 'highlight.js/lib/languages/dockerfile'
import ini from 'highlight.js/lib/languages/ini'
import plaintext from 'highlight.js/lib/languages/plaintext'
import python from 'highlight.js/lib/languages/python'
import yaml from 'highlight.js/lib/languages/yaml'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

hljs.registerLanguage('dockerfile', dockerfile)
hljs.registerLanguage('python', python)
hljs.registerLanguage('yaml', yaml)
hljs.registerLanguage('toml', ini)
hljs.registerLanguage('text', plaintext)

const snapshot = ref({ episodes: [], metadata: {} })
const connected = ref(false)
const error = ref('')
const showResetDialog = ref(false)
const resetting = ref(false)
const resetError = ref('')
const resetStartedAt = ref(null)
const sourceFiles = ref([])
const selectedSource = ref(null)
const sourceError = ref('')
const selectedSavestate = ref('')
const savestateSelectionInitialized = ref(false)
const learnedEnemies = ref([])
let timer
let enemyTimer

const highlightedSource = computed(() => {
  if (!selectedSource.value?.content) return ''
  const language = selectedSource.value.language
  if (language && hljs.getLanguage(language)) {
    return hljs.highlight(selectedSource.value.content, { language }).value
  }
  return hljs.highlightAuto(selectedSource.value.content).value
})

const load = async () => {
  try {
    const response = await fetch('/api/snapshot', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    snapshot.value = payload
    if (!savestateSelectionInitialized.value) {
      selectedSavestate.value = payload.metadata?.run?.savestate || ''
      savestateSelectionInitialized.value = true
    }
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

const loadSource = async path => {
  sourceError.value = ''
  try {
    const response = await fetch(`/api/source?path=${encodeURIComponent(path)}`, { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    selectedSource.value = await response.json()
  } catch (reason) {
    sourceError.value = reason.message
  }
}

const loadSources = async () => {
  try {
    const response = await fetch('/api/sources', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    sourceFiles.value = payload.files || []
    if (sourceFiles.value.length) await loadSource(sourceFiles.value[0].path)
  } catch (reason) {
    sourceError.value = reason.message
  }
}

const loadEnemies = async () => {
  try {
    const response = await fetch('/api/enemies', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    learnedEnemies.value = (await response.json()).enemies || []
  } catch {
    learnedEnemies.value = []
  }
}

onMounted(() => {
  load(); loadSources(); loadEnemies()
  timer = window.setInterval(load, 1500)
  enemyTimer = window.setInterval(loadEnemies, 5000)
})
onBeforeUnmount(() => { window.clearInterval(timer); window.clearInterval(enemyTimer) })

const summary = computed(() => snapshot.value.summary || {})
const stateSummaries = computed(() => summary.value.by_state || {})
const savestateSummaries = computed(() => summary.value.by_savestate || {})
const stateTraining = computed(() => snapshot.value.metadata?.state_training || {})
const savestateCurriculum = computed(() => snapshot.value.metadata?.savestate_curriculum || {})
const configuredStates = computed(() => Object.keys(stateTraining.value))
const states = computed(() => (configuredStates.value.length ? configuredStates.value : Object.keys(stateSummaries.value))
  .sort((left, right) => left.localeCompare(right)))
const availableSavestates = computed(() => [...new Set([
  ...(run.value.savestates || []),
  ...Object.keys(savestateSummaries.value),
])].filter(Boolean).sort((left, right) => left.localeCompare(right)))
const activeSummary = computed(() => selectedSavestate.value
  ? savestateSummaries.value[selectedSavestate.value] || {}
  : summary.value)
const activeSavestateLabel = computed(() => selectedSavestate.value || 'All savestates')
const visibleEnemies = computed(() => learnedEnemies.value.filter(enemy =>
  !selectedSavestate.value || enemy.savestate === selectedSavestate.value))
const stateRows = computed(() => states.value.map(state => ({
  state,
  ...(stateTraining.value[state] || {}),
  curriculum: savestateCurriculum.value[state] || {},
})))
const summarizedEpisodes = computed(() => Number(activeSummary.value.episodes) || 0)
const best = computed(() => activeSummary.value.best_fitness ?? null)
const wins = computed(() => Number(activeSummary.value.wins) || 0)
const winRate = computed(() => summarizedEpisodes.value ? wins.value / summarizedEpisodes.value * 100 : 0)
const summarizedAvgDuration = computed(() => {
  const timed = Number(activeSummary.value.timed_episodes) || 0
  return timed ? Number(activeSummary.value.duration_seconds_total) / timed : null
})
const latestTrainingState = computed(() => activeSummary.value.latest_training_state || summary.value.latest_training_state)
const latestDuration = computed(() => activeSummary.value.latest_duration_seconds ?? null)
const model = computed(() => snapshot.value.metadata?.model || {})
const ppo = computed(() => model.value.ppo || {})
const rnd = computed(() => model.value.rnd || {})
const environment = computed(() => snapshot.value.metadata?.environment || {})
const runtimeDetails = computed(() => {
  const { class: _environmentClass, ...details } = environment.value
  return { class: model.value.class || 'Unknown', ...details }
})
const run = computed(() => snapshot.value.metadata?.run || {})
const server = computed(() => snapshot.value.server || {})
const versionLabel = computed(() => server.value.version === 'DEVELOPMENT'
  ? 'DEVELOPMENT'
  : server.value.version ? `v${server.value.version}` : '—')
const activeAlgorithm = computed(() => entries(ppo.value).length ? 'ppo' : null)
const modelName = computed(() => model.value.display_name || (entries(rnd.value).length ? 'Adaptive Recurrent PPO + RND' : 'PPO'))
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
        <span>{{ latestTrainingState || run.savestate || 'no state' }}</span>
        <span class="endpoint">{{ server.bind_address || '—' }}</span>
        <span :class="['connection', { offline: !connected }]"><i></i>{{ connected ? 'Live' : 'Disconnected' }}</span>
      </div>
    </header>

    <section class="controls panel">
      <div><p class="eyebrow">OBSERVATION WINDOW</p><strong>Episode telemetry</strong></div>
      <label>Savestate
        <select v-model="selectedSavestate">
          <option value="">All savestates</option>
          <option v-for="savestate in availableSavestates" :key="savestate" :value="savestate">{{ savestate }}</option>
        </select>
      </label>
      <button class="reset-button" :disabled="!control.restart_supported || control.reset_pending || resetting" @click="showResetDialog = true">
        {{ control.reset_pending || resetting ? 'Restarting…' : 'Delete model' }}
      </button>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section class="kpis">
      <article class="panel metric"><p>Best fitness</p><strong class="mint">{{ fmt(best, 2) }}</strong><small>{{ activeSavestateLabel }} episode history</small></article>
      <article class="panel metric"><p>Win rate</p><strong>{{ fmt(winRate, 1) }}<em>%</em></strong><small>{{ wins }} successful / {{ summarizedEpisodes }} episodes</small></article>
      <article class="panel metric"><p>Avg training time</p><strong>{{ duration(summarizedAvgDuration) }}</strong><small>{{ duration(latestDuration) }} latest episode</small></article>
      <article class="panel metric"><p>Episodes</p><strong>{{ fmt(summarizedEpisodes) }}</strong><small>{{ activeSavestateLabel }} observed</small></article>
    </section>

    <section class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">STATE CURRICULUM</p><h2>Training state models</h2><p>One learned policy per game state, ordered by the generated workflow.</p></div>
        <span>{{ stateRows.length }} models</span>
      </div>
      <div class="details-grid state-model-grid">
      <article v-for="(row, index) in stateRows" :key="row.state" class="panel detail-card state-model-card">
        <div class="card-heading">
          <div class="state-title"><span class="state-index">{{ index + 1 }}</span><div><p class="eyebrow">LEARNED STATE POLICY</p><h2>{{ row.state }}</h2></div></div>
          <span class="chip" :class="{ muted: !row.active_environments }">{{ row.active_environments ? `${row.active_environments} active` : 'Waiting' }}</span>
        </div>
        <dl>
          <dt>Collected training steps</dt><dd>{{ fmt(row.collected_steps) }}</dd>
          <dt>Rollout buffer</dt><dd>{{ fmt(row.rollout_steps) }} / {{ fmt(row.rollout_capacity) }}</dd>
          <dt>Model updates</dt><dd>{{ fmt(row.model_updates) }}</dd>
          <dt>Completed segments</dt><dd>{{ fmt(row.completed_segments) }}</dd>
          <dt>Best state fitness</dt><dd>{{ fmt(row.best_fitness, 2) }}</dd>
          <dt>Curriculum checkpoint</dt><dd>{{ row.curriculum.has_checkpoint ? 'Saved' : '—' }}</dd>
          <dt>Consecutive successes</dt><dd>{{ fmt(row.curriculum.consecutive_successes) }} / {{ fmt(row.curriculum.success_threshold) }}</dd>
          <dt>Typical attempt</dt><dd>{{ fmt(row.curriculum.typical_episode_steps) }} steps</dd>
          <dt>Bad-checkpoint evidence</dt><dd>{{ fmt(row.curriculum.bad_checkpoint_evidence) }} / {{ fmt(row.curriculum.failure_threshold) }}</dd>
          <dt>Curriculum status</dt><dd>{{ row.curriculum.mastered ? 'Mastered' : row.curriculum.active ? 'Training now' : 'Waiting' }}</dd>
        </dl>
      </article>
      </div>
    </section>

    <section class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">TRAINING ENGINE</p><h2>Runtime and learning system</h2><p>Shared environment configuration and optimization details used by every state model.</p></div>
      </div>
      <div :class="['details-grid system-grid', { 'two-column': !entries(rnd).length }]">
      <article class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">SHARED RUNTIME</p><h2>Environment</h2><p class="card-description">Emulator, wrappers, observations, and action spaces.</p></div><span class="chip">{{ environment.num_envs || run.configured_envs || '—' }} envs</span></div>
        <dl><template v-for="([key, value]) in entries(runtimeDetails)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
      </article>
      <article v-if="activeAlgorithm === 'ppo'" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">SHARED OPTIMIZER</p><h2>{{ modelName }}</h2><p class="card-description">PPO settings used to update each learned state policy.</p><p v-if="model.description" class="placeholder">{{ model.description }}</p></div><span class="chip" :class="{ muted: !entries(ppo).length }">{{ entries(ppo).length ? 'Configured' : 'Not active' }}</span></div>
        <dl v-if="entries(ppo).length"><template v-for="([key, value]) in entries(ppo)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
        <p v-else class="placeholder">No PPO parameters on the active model.</p>
      </article>
      <article v-if="activeAlgorithm === 'ppo' && entries(rnd).length" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">SHARED EXPLORATION</p><h2>Self-tuned RND</h2><p class="card-description">Curiosity and exploration pressure shared across state training.</p><p class="placeholder">Uses score staleness and missing wins to tune curiosity, entropy, PPO step size, clip range, and RND update pressure.</p></div><span class="chip">Active</span></div>
        <dl><template v-for="([key, value]) in entries(rnd)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
      </article>
      <article v-if="!activeAlgorithm" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">MODEL</p><h2>Algorithm</h2></div><span class="chip muted">Waiting</span></div>
        <p class="placeholder">Algorithm details appear when PPO starts.</p>
      </article>
      </div>
    </section>

    <section class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">VISUAL ENEMY MEMORY</p><h2>Learned enemies</h2><p>Motion-isolated sprites captured near Explorer when the RAM-defined hit signal fires.</p></div>
        <span>{{ visibleEnemies.length }} images</span>
      </div>
      <div v-if="visibleEnemies.length" class="enemy-gallery">
        <article v-for="enemy in visibleEnemies" :key="enemy.path" class="panel enemy-card">
          <div class="enemy-image"><img :src="`/api/enemy?path=${encodeURIComponent(enemy.path)}`" :alt="`Learned enemy ${enemy.id}`" /></div>
          <div><strong>{{ enemy.state }}</strong><span>{{ enemy.savestate }}</span><small>{{ enemy.id }}</small></div>
        </article>
      </div>
      <div v-else class="panel enemy-empty">No enemy images learned for {{ activeSavestateLabel }} yet. Define Explorer’s <code>hit</code> outcome from RAM to supervise learning.</div>
    </section>

    <section class="panel source-browser">
      <div class="source-browser-heading">
        <div><p class="eyebrow">GENERATED PROJECT</p><h2>Files and source code</h2></div>
        <span class="chip" :class="{ muted: !sourceFiles.length }">{{ sourceFiles.length }} files</span>
      </div>
      <div v-if="sourceFiles.length" class="source-browser-body">
        <nav class="source-files" aria-label="Generated files">
          <button
            v-for="file in sourceFiles"
            :key="file.path"
            :class="{ active: selectedSource?.path === file.path }"
            @click="loadSource(file.path)"
          >
            <strong>{{ file.path }}</strong><small>{{ file.language }} · {{ fmt(file.size) }} B</small>
          </button>
        </nav>
        <article class="source-viewer">
          <header v-if="selectedSource"><strong>{{ selectedSource.path }}</strong><span>{{ selectedSource.language }}</span></header>
          <pre v-if="selectedSource"><code class="hljs" v-html="highlightedSource"></code></pre>
          <p v-else class="placeholder">Choose a generated file to inspect it.</p>
        </article>
      </div>
      <p v-else-if="sourceError" class="error">Generated files could not be loaded: {{ sourceError }}</p>
      <p v-else class="placeholder">No generated runner files were found.</p>
    </section>

    <footer>Local telemetry · refreshes every 1.5 seconds · {{ summarizedEpisodes }} complete episodes for {{ activeSavestateLabel }}</footer>

    <div v-if="showResetDialog" class="modal-backdrop" @click.self="showResetDialog = false">
      <section class="reset-dialog panel" role="dialog" aria-modal="true" aria-labelledby="reset-title">
        <p class="eyebrow danger-text">DESTRUCTIVE ACTION</p>
        <h2 id="reset-title">Delete {{ run.game }} model?</h2>
        <p>All models, recordings, cache data, and training history for this runner will be deleted. The active training run will restart from its configured savestate.</p>
        <p v-if="resetError" class="error">{{ resetError }}</p>
        <div class="dialog-actions">
          <button class="cancel-button" :disabled="resetting" @click="showResetDialog = false">Cancel</button>
          <button class="confirm-reset" :disabled="resetting" @click="resetModel">{{ resetting ? 'Requesting…' : 'Delete and restart' }}</button>
        </div>
      </section>
    </div>
  </main>
</template>
