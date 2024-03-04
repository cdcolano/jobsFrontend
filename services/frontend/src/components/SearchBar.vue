<script setup>
import { ref } from 'vue'
import router from "@/router/index.js";
import { useRoute } from 'vue-router';

const route = useRoute()
const query = ref({})

query.value = route.params.query ?? ''

function search() {
  if (query.value != null && query.value !== ""){
    console.log(query.value)
    router.push({
      name: 'search',
      params: { query: query.value }
    })
  }
}
</script>

<template>
    <div class="grow flex flex-col items-center justify-center text-left
                pb-14">
        <div class="h-12 w-10/12 relative mt-10">
            <p class="absolute -top-3/4 text-left text-lg">Search Jobs:</p>
            <input type="text"
                   class="h-full w-full focus:outline-slate-600 rounded-lg border
                    border-slate-400 p-3"
                   v-model="query"
                   @keydown.enter="search">
        </div>
    </div>
</template>

<style scoped>

</style>