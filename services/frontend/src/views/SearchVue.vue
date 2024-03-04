<script setup>
import {ref} from 'vue';
import {useRoute} from 'vue-router';

import SearchBar from "@/components/SearchBar.vue";
import SearchResult from "@/components/SearchResult.vue";

const route = useRoute();
const results = ref(null);

fetch('http://localhost:5001/search/' + route.params.query)
    .then(response => response.json())
    .then(data => results.value = data);
</script>

<template>
    <div class="flex flex-col gap-3 w-full">
        <div class="min-h-32 pt-12 flex border-b border-slate-500">
            <search-bar></search-bar>
        </div>
        <div v-if="results" class="flex grow justify-center mb-12">
            <ul class="flex flex-col w-10/12 px-3 divide-y-2 divide-slate-200
                        dark:divide-slate-600">
                <li v-for="result in results.results" :key="result.id">
                    <search-result
                        :title="result.title"
                        :description="result.description">
                    </search-result>
                </li>
            </ul>
        </div>
        <div v-if="!results" class="text-center">
            <div class="spinner-border spinner-border-sm"></div>
        </div>
    </div>
</template>

<style scoped>

</style>