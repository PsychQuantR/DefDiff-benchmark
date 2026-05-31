#!/usr/bin/env Rscript
## analyze-community-benchmark.R
## Mixed-effects analysis of the community benchmark CSV (see the
## `community-benchmark` spec). Fits
##
##   log10(median_ms) ~ system * operation * stage * threads + (1 | machine)
##
## via lme4::lmer, treating the machine (chip identity) as a random intercept
## and repeated runs as within-machine replicates. When fewer than five
## distinct machines are present the random intercept is not estimable, so it
## falls back to a fixed-effects lm on the same fixed structure and says so.
##
## Usage: Rscript inst/benchmarks/analyze-community-benchmark.R [csv_path]

MIN_MACHINES <- 5L

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1L) args[[1L]] else "community-benchmark.csv"

if (!requireNamespace("lme4", quietly = TRUE)) {
  stop("This analysis requires the 'lme4' package. ",
       "Install it with install.packages(\"lme4\").", call. = FALSE)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("This analysis requires the 'jsonlite' package (for log re-parsing). ",
       "Install it with install.packages(\"jsonlite\").", call. = FALSE)
}
if (!file.exists(csv_path)) {
  stop(sprintf("CSV not found: %s. Run the harness and regenerate the CSV first.",
               csv_path), call. = FALSE)
}

df <- utils::read.csv(csv_path, stringsAsFactors = FALSE)
if (nrow(df) == 0L) {
  stop("The benchmark CSV has no rows yet. Collect at least one run-log first.",
       call. = FALSE)
}

# Model positive, finite timings; log10 stabilizes variance across orders of
# magnitude (matching the reference VLM-OCR analysis).
df <- df[!is.na(df$median_ms) & df$median_ms > 0, , drop = FALSE]
df$log10_ms <- log10(df$median_ms)
df$machine  <- df$chip
for (v in c("system", "operation", "stage", "threads", "machine")) {
  df[[v]] <- factor(df[[v]])
}

n_machines <- nlevels(df$machine)
cat(sprintf("Community benchmark analysis\n"))
cat(sprintf("- rows            : %d\n", nrow(df)))
cat(sprintf("- distinct machines: %d (threshold for mixed model: %d)\n\n",
            n_machines, MIN_MACHINES))

fixed <- log10_ms ~ system * operation * stage * threads

if (n_machines >= MIN_MACHINES) {
  cat("Fitting mixed-effects model with a (1 | machine) random intercept.\n\n")
  fit <- lme4::lmer(update(fixed, . ~ . + (1 | machine)), data = df, REML = TRUE)
  print(summary(fit))
} else {
  cat(sprintf(paste0("Only %d machine(s) present, below the %d needed to ",
                     "estimate a random intercept.\nFalling back to a ",
                     "fixed-effects lm on the same fixed structure.\n\n"),
              n_machines, MIN_MACHINES))
  fit <- stats::lm(fixed, data = df)
  print(summary(fit))
}

invisible(fit)
