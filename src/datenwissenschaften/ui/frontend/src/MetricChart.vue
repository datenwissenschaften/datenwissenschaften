<script setup>
import { computed } from 'vue'

const props = defineProps({
  rows: { type: Array, default: () => [] },
  series: { type: Array, required: true },
  height: { type: Number, default: 240 },
  includeZero: { type: Boolean, default: false },
})

const width = 1000
const pad = { top: 18, right: 18, bottom: 30, left: 62 }
const plotWidth = width - pad.left - pad.right
const plotHeight = computed(() => props.height - pad.top - pad.bottom)
const values = computed(() => props.rows.flatMap(row => props.series.map(item => Number(row[item.key])).filter(Number.isFinite)))
const domain = computed(() => {
  if (!values.value.length) return [0, 1]
  let min = Math.min(...values.value)
  let max = Math.max(...values.value)
  if (props.includeZero) {
    min = Math.min(min, 0)
    max = Math.max(max, 0)
  }
  if (min === max) {
    if (props.includeZero && min === 0) return [0, 1]
    min -= Math.max(Math.abs(min) * .1, 1)
    max += Math.max(Math.abs(max) * .1, 1)
  }
  const breathing = (max - min) * .08
  return [props.includeZero && min === 0 ? 0 : min - breathing, props.includeZero && max === 0 ? 0 : max + breathing]
})
const x = index => pad.left + (props.rows.length <= 1 ? plotWidth / 2 : index * plotWidth / (props.rows.length - 1))
const y = value => pad.top + (domain.value[1] - value) * plotHeight.value / (domain.value[1] - domain.value[0])
const path = key => props.rows.map((row, index) => {
  const value = Number(row[key])
  return Number.isFinite(value) ? `${index ? 'L' : 'M'}${x(index).toFixed(1)},${y(value).toFixed(1)}` : ''
}).join(' ')
const ticks = computed(() => Array.from({ length: 5 }, (_, i) => domain.value[1] - i * (domain.value[1] - domain.value[0]) / 4))
const format = value => Math.abs(value) >= 1000 ? Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) : value.toFixed(Math.abs(value) < 10 ? 1 : 0)
</script>

<template>
  <div class="chart-wrap">
    <svg :viewBox="`0 0 ${width} ${height}`" role="img" aria-label="Metric history chart">
      <defs>
        <linearGradient v-for="item in series" :id="`glow-${item.key}`" :key="item.key" x1="0" x2="1">
          <stop offset="0" :stop-color="item.color" stop-opacity=".25" />
          <stop offset=".5" :stop-color="item.color" />
          <stop offset="1" :stop-color="item.color" stop-opacity=".55" />
        </linearGradient>
      </defs>
      <g v-for="tick in ticks" :key="tick">
        <line :x1="pad.left" :x2="width - pad.right" :y1="y(tick)" :y2="y(tick)" class="grid-line" />
        <text :x="pad.left - 12" :y="y(tick) + 4" text-anchor="end" class="axis-label">{{ format(tick) }}</text>
      </g>
      <text :x="pad.left" :y="height - 7" class="axis-label">older</text>
      <text :x="width - pad.right" :y="height - 7" text-anchor="end" class="axis-label">latest · {{ rows.length }} episodes</text>
      <path v-for="item in series" :key="item.key" :d="path(item.key)" fill="none" :stroke="`url(#glow-${item.key})`" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
    <div v-if="!rows.length" class="empty-chart">Waiting for completed episodes…</div>
  </div>
</template>
