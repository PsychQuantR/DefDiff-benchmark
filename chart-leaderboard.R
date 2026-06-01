#!/usr/bin/env Rscript
## chart-leaderboard.R
## Render the community benchmark CSV as bar charts — see DefDiff-benchmark#1.
##
## Two charts by timing stage (the CSV separates startup from steady state via the
## `stage` column; we plot two of them):
##   leaderboard-eval.png   — steady-state `eval` median (how fast once warm)
##   leaderboard-total.png  — cold-start: per-system sum of all non-`total` stages
##                            (DefDiff = build+first_eval+eval, PyTorch =
##                            import+trace+eval, numDeriv = eval). The CSV's own
##                            harness-written `total` rows are NOT used — they omit
##                            PyTorch and under-count DefDiff; recomputing here makes
##                            cold-start same-basis comparable across systems.
##
## Each chart: x = problem_id, fill = system, log10 y, facet_grid(threads ~ operation).
## Bars are the MEAN of per-run medians across run_id replicates, with a CI error bar
## (normal-approx mean ± qt * SE; falls back to no error bar when a cell has <2 runs).
## Replication — not eyeballing a single run — is what lets the chart honestly show
## whether a factor (e.g. threads) matters (Casella 2008, statistical design).
##
## Usage: Rscript chart-leaderboard.R [csv] [out_dir]
##   csv     default community-benchmark.csv
##   out_dir default . (writes leaderboard-eval.png + leaderboard-total.png)

args    <- commandArgs(trailingOnly = TRUE)
csv     <- if (length(args) >= 1L) args[[1L]] else "community-benchmark.csv"
out_dir <- if (length(args) >= 2L) args[[2L]] else "."

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("chart-leaderboard.R requires 'ggplot2'. Install with install.packages(\"ggplot2\").",
       call. = FALSE)
}
if (!requireNamespace("scales", quietly = TRUE)) {
  stop("chart-leaderboard.R requires 'scales'. Install with install.packages(\"scales\").",
       call. = FALSE)
}
library(ggplot2)

CI_LEVEL    <- 0.95
SYSTEM_ORDER <- c("DefDiff", "numDeriv", "PyTorch", "JAX")

## Pick the best raster device available (ragg if present for crisp text).
.save_png <- function(path, plot, width = 10, height = 6.5) {
  if (requireNamespace("ragg", quietly = TRUE)) {
    ggsave(path, plot, width = width, height = height, dpi = 150,
           device = ragg::agg_png)
  } else {
    ggsave(path, plot, width = width, height = height, dpi = 150)
  }
}

## Graceful degrade: empty / missing CSV → placeholder chart so `make leaderboard`
## never breaks on a fresh checkout.
.placeholder <- function(path, title) {
  p <- ggplot() +
    annotate("text", x = 0, y = 0,
             label = "No community submissions yet.\nRun the harness and open a PR adding your run-log.",
             size = 5, lineheight = 1.1) +
    theme_void() +
    labs(title = title)
  .save_png(path, p, width = 8, height = 4)
  message(sprintf("→ wrote placeholder %s (no data)", path))
}

eval_png  <- file.path(out_dir, "leaderboard-eval.png")
total_png <- file.path(out_dir, "leaderboard-total.png")

if (!file.exists(csv) || file.info(csv)$size == 0) {
  .placeholder(eval_png,  "Benchmark — steady-state (eval)")
  .placeholder(total_png, "Benchmark — cold-start (total)")
  quit(save = "no", status = 0)
}

d <- utils::read.csv(csv, stringsAsFactors = FALSE)
d <- d[!is.na(d$median_ms) & d$median_ms > 0, , drop = FALSE]
if (nrow(d) == 0L) {
  .placeholder(eval_png,  "Benchmark — steady-state (eval)")
  .placeholder(total_png, "Benchmark — cold-start (total)")
  quit(save = "no", status = 0)
}

## --- Build the two per-stage long tables ------------------------------------
## Grouping key for a "cell" = (chip, system, operation, problem_id, threads).
## Within a cell, each run_id contributes one value; we aggregate across run_ids.

.cell_keys <- c("chip", "system", "operation", "problem_id", "threads")

## eval: one value per (cell, run_id) = the eval-stage median of that run.
eval_per_run <- d[d$stage == "eval",
                  c(.cell_keys, "run_id", "n", "median_ms"), drop = FALSE]

## derived total: per (cell, run_id), sum every non-`total` stage's median.
nt <- d[d$stage != "total", , drop = FALSE]
total_per_run <- stats::aggregate(
  median_ms ~ chip + system + operation + problem_id + threads + run_id + n,
  data = nt, FUN = sum)

## Aggregate per-run values across run_id replicates → mean, SE, CI half-width.
.aggregate_ci <- function(per_run) {
  if (nrow(per_run) == 0L) return(per_run[0, , drop = FALSE])
  g <- split(per_run, interaction(per_run[.cell_keys], drop = TRUE))
  rows <- lapply(g, function(blk) {
    v  <- blk$median_ms
    nrep <- length(v)
    m  <- mean(v)
    if (nrep >= 2L) {
      se <- stats::sd(v) / sqrt(nrep)
      half <- stats::qt(1 - (1 - CI_LEVEL) / 2, df = nrep - 1L) * se
    } else {
      half <- NA_real_   # single replicate → no honest CI
    }
    data.frame(blk[1L, .cell_keys, drop = FALSE],
               n = blk$n[1L], mean_ms = m, ci_half = half,
               nrep = nrep, stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

eval_agg  <- .aggregate_ci(eval_per_run)
total_agg <- .aggregate_ci(total_per_run)

## --- Plot helper ------------------------------------------------------------
.fmt_threads <- function(x) ifelse(x == "1", "1 thread", paste0(x, " threads"))

.bar_chart <- function(agg, title, subtitle) {
  agg$system <- factor(agg$system, levels = intersect(SYSTEM_ORDER, unique(agg$system)))
  agg$threads_lab <- .fmt_threads(as.character(agg$threads))
  # On a log axis, geom_col is misleading: a bar's length encodes area-from-zero,
  # but log(0) = -inf, so bars appear to grow from an arbitrary baseline (a sub-ms
  # value reads as "hanging down from 1"). Use pointrange instead — the point's
  # POSITION reads the value directly, independent of any baseline, and the range
  # shows the CI. This is also the natural geometry for mean ± CI.
  agg$ymin <- ifelse(is.na(agg$ci_half), agg$mean_ms, agg$mean_ms - agg$ci_half)
  agg$ymax <- ifelse(is.na(agg$ci_half), agg$mean_ms, agg$mean_ms + agg$ci_half)
  agg$ymin <- pmax(agg$ymin, agg$mean_ms / 1e3)  # keep positive for log scale

  ggplot(agg, aes(x = problem_id, y = mean_ms, colour = system)) +
    geom_pointrange(aes(ymin = ymin, ymax = ymax),
                    position = position_dodge(width = 0.6),
                    size = 0.45, linewidth = 0.5, na.rm = TRUE) +
    scale_y_log10(labels = scales::label_number(drop0trailing = TRUE)) +
    facet_grid(threads_lab ~ operation, scales = "free_x", space = "free_x") +
    labs(title = title, subtitle = subtitle,
         x = "problem", y = "median ms (log scale, lower = faster)", colour = "system") +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          legend.position = "top",
          plot.subtitle = element_text(size = 9, colour = "grey35"))
}

## Caption notes: chips, replicate count, and the hessian/grad n + PyTorch caveat.
.subtitle <- function(agg, stage_desc) {
  chips <- paste(unique(agg$chip), collapse = ", ")
  reps  <- range(agg$nrep)
  rep_s <- if (reps[1L] == reps[2L]) sprintf("%d", reps[1L]) else sprintf("%d-%d", reps[1L], reps[2L])
  ns    <- tapply(agg$n, agg$operation, function(x) paste(unique(x), collapse = "/"))
  n_s   <- paste(sprintf("%s n=%s", names(ns), ns), collapse = ", ")
  paste0(stage_desc, " | ", chips, " | ", rep_s, " replicate run(s); error bars = ",
         sprintf("%.0f%%", CI_LEVEL * 100), " CI",
         " | ", n_s,
         if (!"PyTorch" %in% agg$system[agg$operation == "hessian"]) " | hessian: PyTorch not measured" else "")
}

if (nrow(eval_agg) > 0L) {
  .save_png(eval_png,
            .bar_chart(eval_agg, "Differentiation speed — steady-state (eval)",
                       .subtitle(eval_agg, "lower = faster, warm")))
  message(sprintf("→ wrote %s", eval_png))
} else .placeholder(eval_png, "Benchmark — steady-state (eval)")

if (nrow(total_agg) > 0L) {
  .save_png(total_png,
            .bar_chart(total_agg, "Differentiation speed — cold-start (total = sum of stages)",
                       .subtitle(total_agg, "lower = faster, cold start to first derivative")))
  message(sprintf("→ wrote %s", total_png))
} else .placeholder(total_png, "Benchmark — cold-start (total)")

message("Done.")
