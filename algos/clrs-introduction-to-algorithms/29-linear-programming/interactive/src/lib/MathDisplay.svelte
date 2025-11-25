<script lang="ts">
    import katex from "katex";

    interface Props {
        latex: string;
    }

    let { latex }: Props = $props();
    let container: HTMLDivElement;

    $effect(() => {
        if (container && latex) {
            katex.render(latex, container, {
                displayMode: true,
                throwOnError: false,
            });
        }
    });
</script>

<div bind:this={container} class="overflow-x-auto py-4"></div>
