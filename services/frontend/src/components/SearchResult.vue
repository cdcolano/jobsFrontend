<script setup>
defineProps(['title', 'description', 'url'])
</script>

<template>
    <a :ref="url"
        class="flex flex-col p-6 gap-3 group my-4
                hover:bg-slate-50 hover:cursor-pointer
                dark:hover:bg-slate-600">
        <p class="text-xl font-extrabold group-hover:underline">
            {{ title }}
        </p>
        <p class="text-slate-600 dark:text-white">
            {{ description }}
        </p>
    </a>
</template>

<style scoped>

</style>