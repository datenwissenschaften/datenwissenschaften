<script setup>
import { computed } from 'vue'

const props = defineProps({
  values: { type: Array, default: () => [] },
  bucketCount: { type: Number, required: true },
  height: { type: Number, default: 240 },
})

const width = 1000
const pad = { top: 18, right: 18, bottom: 30, left: 62 }
const plotWidth = width - pad.left - pad.right
const plotHeight = computed(() => props.height - pad.top - pad.bottom)
const finiteValues = computed(() => props.values.map(Number).filter(Number.isFinite))
const format = value => Math.abs(value) >= 1000
  ? Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
  : Number(value).toFixed(Math.abs(value) < 10 ? 1 : 0)

const histogram = computed(() => {
  if (!finiteValues.value.length) return { buckets: [], min: 0, max: 1, maxCount: 1 }

  let min = Math.min(...finiteValues.value)
  let max = Math.max(...finiteValues.value)
  if (min === max) {
    const padding = Math.max(Math.abs(min) * 0.05, 0.5)
    min -= padding
    max += padding
  }

  const bucketWidth = (max - min) / props.bucketCount
  const buckets = Array.from({ length: props.bucketCount }, (_, index) => ({
    start: min + index * bucketWidth,
    end: min + (index + 1) * bucketWidth,
    count: 0,
  }))
  for (const value of finiteValues.value) {
    const index = Math.min(Math.floor((value - min) / bucketWidth), props.bucketCount - 1)
    buckets[index].count += 1
  }

  return { buckets, min, max, maxCount: Math.max(...buckets.map(bucket => bucket.count), 1) }
})

const barSlotWidth = computed(() => plotWidth / props.bucketCount)
const barWidth = computed(() => Math.max(barSlotWidth.value - 5, 1))
const x = index => pad.left + index * barSlotWidth.value + 2.5
const y = count => pad.top + (histogram.value.maxCount - count) * plotHeight.value / histogram.value.maxCount
const ticks = computed(() => {
  const max = histogram.value.maxCount
  return [...new Set(Array.from({ length: 5 }, (_, index) => Math.round(max * (4 - index) / 4)))]
})
</script>

<template>
  <div class="chart-wrap">
    <svg :viewBox="`0 0 ${width} ${height}`" role="img" :aria-label="`Fitness histogram with ${bucketCount} buckets`">
      <g v-for="tick in ticks" :key="tick">
        <line :x1="pad.left" :x2="width - pad.right" :y1="y(tick)" :y2="y(tick)" class="grid-line" />
        <text :x="pad.left - 12" :y="y(tick) + 4" text-anchor="end" class="axis-label">{{ tick }}</text>
      </g>
      <g v-for="(bucket, index) in histogram.buckets" :key="index">
        <rect
          :x="x(index)"
          :y="y(bucket.count)"
          :width="barWidth"
          :height="Math.max(pad.top + plotHeight - y(bucket.count), 0)"
          rx="2"
          class="histogram-bar"
        >
          <title>{{ format(bucket.start) }} to {{ format(bucket.end) }}: {{ bucket.count }} episodes</title>
        </rect>
      </g>
      <text :x="pad.left" :y="height - 7" class="axis-label">{{ format(histogram.min) }} fitness</text>
      <text :x="width - pad.right" :y="height - 7" text-anchor="end" class="axis-label">{{ format(histogram.max) }} fitness</text>
    </svg>
    <div v-if="!finiteValues.length" class="empty-chart">Waiting for completed episodes…</div>
  </div>
</template>
