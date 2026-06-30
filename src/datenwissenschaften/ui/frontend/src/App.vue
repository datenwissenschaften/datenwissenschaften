<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import MetricChart from './MetricChart.vue'

const snapshot = ref({ episodes: [], generations: [], metadata: {} })
const connected = ref(false)
const error = ref('')
const stateFilter = ref('all')
const windowSize = ref(200)
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
const states = computed(() => [...new Set(episodes.value.map(row => row.training_state).filter(Boolean))])
const stateHistory = computed(() => episodes.value.filter(row => stateFilter.value === 'all' || row.training_state === stateFilter.value))
const filtered = computed(() => stateHistory.value.slice(-windowSize.value))
const reversed = computed(() => [...filtered.value].reverse())
const latest = computed(() => filtered.value.at(-1))
const best = computed(() => filtered.value.length ? Math.max(...filtered.value.map(row => Number(row.fitness) || 0)) : null)
const wins = computed(() => filtered.value.filter(row => row.won === true).length)
const winRate = computed(() => filtered.value.length ? wins.value / filtered.value.length * 100 : 0)
const average = key => computed(() => filtered.value.length ? filtered.value.reduce((sum, row) => sum + (Number(row[key]) || 0), 0) / filtered.value.length : 0)
const avgFitness = average('fitness')
const fitnessHistory = computed(() => {
  let sum = 0
  const history = stateHistory.value.map((row, index) => {
    sum += Number(row.fitness) || 0
    return { ...row, mean_fitness: sum / (index + 1) }
  })
  return history.slice(-windowSize.value)
})
const model = computed(() => snapshot.value.metadata?.model || {})
const ppo = computed(() => model.value.ppo || {})
const neat = computed(() => snapshot.value.metadata?.neat || {})
const environment = computed(() => snapshot.value.metadata?.environment || {})
const runtimeDetails = computed(() => {
  const { class: _environmentClass, ...details } = environment.value
  return { class: model.value.class || 'Unknown', ...details }
})
const run = computed(() => snapshot.value.metadata?.run || {})
const server = computed(() => snapshot.value.server || {})
const generation = computed(() => snapshot.value.generations?.at(-1))
const currentGeneration = computed(() => neat.value.current_generation ?? generation.value?.generation)
const generationEpisodesCompleted = computed(() => Number(neat.value.generation_episodes_completed) || 0)
const generationEpisodesTotal = computed(() => Number(neat.value.generation_episodes_total) || 0)
const generationProgress = computed(() => generationEpisodesTotal.value
  ? Math.min(100, generationEpisodesCompleted.value / generationEpisodesTotal.value * 100)
  : 0)
const neatDetails = computed(() => {
  const {
    current_generation: _currentGeneration,
    generation_episodes_completed: _generationEpisodesCompleted,
    generation_episodes_total: _generationEpisodesTotal,
    ...details
  } = neat.value
  return details
})
const activeAlgorithm = computed(() => entries(neat.value).length ? 'neat' : entries(ppo.value).length ? 'ppo' : null)
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

const fitnessSeries = [{ key: 'mean_fitness', label: 'Mean fitness', color: '#8cf5c6' }]
const fmt = (value, digits = 0) => value == null ? '—' : Intl.NumberFormat('en', { maximumFractionDigits: digits }).format(value)
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
        <div><p class="eyebrow">DATENWISSENSCHAFTEN</p><h1>Training Observatory</h1></div>
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
      <label>State<select v-model="stateFilter"><option value="all">All states</option><option v-for="state in states" :key="state">{{ state }}</option></select></label>
      <label>Range<select v-model.number="windowSize"><option :value="50">Last 50</option><option :value="200">Last 200</option><option :value="500">Last 500</option><option :value="5000">All retained</option></select></label>
      <button class="reset-button" :disabled="!control.restart_supported || control.reset_pending || resetting" @click="showResetDialog = true">
        {{ control.reset_pending || resetting ? 'Restarting…' : 'Delete model' }}
      </button>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section class="kpis">
      <article class="panel metric"><p>Latest fitness</p><strong>{{ fmt(latest?.fitness, 2) }}</strong><small>mean {{ fmt(avgFitness, 2) }}</small></article>
      <article class="panel metric"><p>Best fitness</p><strong class="mint">{{ fmt(best, 2) }}</strong><small>within selection</small></article>
      <article class="panel metric"><p>Win rate</p><strong>{{ fmt(winRate, 1) }}<em>%</em></strong><small>{{ wins }} successful / {{ filtered.length }} episodes</small></article>
    </section>

    <section class="charts">
      <article class="panel chart-card wide">
        <div class="card-heading"><div><p class="eyebrow">REWARD SIGNAL</p><h2>Fitness over time</h2></div><div class="legend"><i style="--color:#8cf5c6"></i>Mean fitness</div></div>
        <MetricChart :rows="fitnessHistory" :series="fitnessSeries" :include-zero="true" />
      </article>
    </section>

    <section class="details-grid two-column">
      <article class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">RUNTIME</p><h2>Environment</h2></div><span class="chip">{{ environment.num_envs || run.configured_envs || '—' }} envs</span></div>
        <dl><template v-for="([key, value]) in entries(runtimeDetails)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
      </article>
      <article v-if="activeAlgorithm === 'ppo'" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">POLICY OPTIMIZATION</p><h2>PPO</h2></div><span class="chip" :class="{ muted: !entries(ppo).length }">{{ entries(ppo).length ? 'Configured' : 'Not active' }}</span></div>
        <dl v-if="entries(ppo).length"><template v-for="([key, value]) in entries(ppo)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
        <p v-else class="placeholder">No PPO parameters on the active model.</p>
      </article>
      <article v-else-if="activeAlgorithm === 'neat'" class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">EVOLUTION</p><h2>NEAT</h2></div><span class="chip">gen {{ currentGeneration ?? '—' }}</span></div>
        <div v-if="generationEpisodesTotal" class="generation-progress">
          <div><span>Current generation</span><strong>{{ fmt(generationEpisodesCompleted) }} / {{ fmt(generationEpisodesTotal) }} episodes</strong></div>
          <div class="progress-track"><i :style="{ width: `${generationProgress}%` }"></i></div>
          <small>{{ fmt(generationProgress, 1) }}% complete</small>
        </div>
        <dl v-if="entries(neatDetails).length"><template v-for="([key, value]) in entries(neatDetails)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
        <p v-else class="placeholder">NEAT details appear when evolution starts.</p>
      </article>
      <article v-else class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">MODEL</p><h2>Algorithm</h2></div><span class="chip muted">Waiting</span></div>
        <p class="placeholder">Algorithm details appear when PPO or NEAT starts.</p>
      </article>
    </section>

    <section class="panel episodes-card">
      <div class="card-heading"><div><p class="eyebrow">DIAGNOSTICS</p><h2>Recent episodes</h2></div><span class="count">{{ episodes.length }} retained</span></div>
      <div class="table-scroll"><table><thead><tr><th>#</th><th>Env</th><th>Training state</th><th>Fitness</th><th>Won</th><th>Final state</th></tr></thead>
        <tbody><tr v-for="row in reversed.slice(0, 100)" :key="row.index"><td class="dim">{{ row.index }}</td><td>{{ row.env }}</td><td><span class="state">{{ row.training_state }}</span></td><td class="fitness">{{ fmt(row.fitness, 2) }}</td><td><span :class="['status', row.won === true ? 'success' : 'neutral']">{{ row.won == null ? '—' : row.won ? 'Won' : 'No' }}</span></td><td>{{ row.final_state || '—' }}</td></tr>
        <tr v-if="!reversed.length"><td colspan="6" class="empty-row">Waiting for the evaluator to complete an episode.</td></tr></tbody>
      </table></div>
    </section>
    <footer>Local telemetry · refreshes every 1.5 seconds · {{ filtered.length }} episodes in view</footer>

    <div v-if="showResetDialog" class="modal-backdrop" @click.self="showResetDialog = false">
      <section class="reset-dialog panel" role="dialog" aria-modal="true" aria-labelledby="reset-title">
        <p class="eyebrow danger-text">DESTRUCTIVE ACTION</p>
        <h2 id="reset-title">Delete {{ run.game }} model?</h2>
        <p>All checkpoints and model history for this game will be deleted. The current generation will stop and training will restart from generation zero.</p>
        <p v-if="resetError" class="error">{{ resetError }}</p>
        <div class="dialog-actions">
          <button class="cancel-button" :disabled="resetting" @click="showResetDialog = false">Cancel</button>
          <button class="confirm-reset" :disabled="resetting" @click="resetModel">{{ resetting ? 'Requesting…' : 'Delete and restart' }}</button>
        </div>
      </section>
    </div>
  </main>
</template>
