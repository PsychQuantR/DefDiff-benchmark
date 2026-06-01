#!/usr/bin/env Rscript
## run-community-benchmark.R
## Community differentiation benchmark harness (DefDiff-benchmark repo).
## Re-runnable: each invocation is ONE within-machine replicate. It emits
## exactly one self-contained JSON run-log (the source of truth) carrying full
## provenance plus every measurement, with timing decomposed by stage (build /
## first_eval / eval / total, plus import / trace / jit_compile for the optional
## Python systems) and swept across single- and multi-thread settings.
##
## DefDiff and numDeriv are required; PyTorch and JAX are optional and recorded
## as absent when undetected (intentionally unbalanced data).
##
## Usage:
##   Rscript run-community-benchmark.R [--append] [--reps N]
##           [--contributor NAME] [--quick] [--out DIR]
##
##   --append        write the log into ./community-logs/ (the tracked archive)
##                   instead of the current directory
##   --reps N        steady-state iterations per eval stage (default 50)
##   --contributor   override the contributor name (default: OS user)
##   --quick         tiny n grid for a fast smoke run
##   --out DIR       explicit output directory for the log file

# --- Require the DefDiff package ---------------------------------------------

if (!requireNamespace("DefDiff", quietly = TRUE)) {
  stop("This benchmark requires the DefDiff package. Install it with ",
       "remotes::install_github(\"PsychQuantR/DefDiff\").", call. = FALSE)
}
suppressPackageStartupMessages(library(DefDiff))

# --- Harness helpers (self-contained: provenance + timing + run id) ----------
# This repo owns the harness; the DefDiff package owns only the log-parsing and
# leaderboard-rendering API. The run-log schema (schema_version 1) is the
# contract DefDiff::parse_benchmark_logs consumes.

HARNESS_VERSION <- "0.1.0"
SCHEMA_VERSION  <- 1L

.or <- function(x, default) if (is.null(x) || length(x) == 0L) default else x

.bench_detect_chip <- function() {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    v <- tryCatch(system2("sysctl", c("-n", "machdep.cpu.brand_string"),
                          stdout = TRUE, stderr = NULL),
                  error = function(e) character())
    if (length(v) && nzchar(v[[1L]])) return(v[[1L]])
  }
  unname(Sys.info()[["machine"]])
}
.bench_detect_cores <- function() {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    v <- tryCatch(suppressWarnings(as.integer(
           system2("sysctl", c("-n", "hw.physicalcpu"), stdout = TRUE, stderr = NULL))),
         error = function(e) NA_integer_)
    if (length(v) && !is.na(v[[1L]])) return(v[[1L]])
  }
  as.integer(tryCatch(parallel::detectCores(logical = FALSE),
                      error = function(e) NA_integer_))
}
.bench_detect_ram_gb <- function() {
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    b <- tryCatch(suppressWarnings(as.numeric(
           system2("sysctl", c("-n", "hw.memsize"), stdout = TRUE, stderr = NULL))),
         error = function(e) NA_real_)
    if (length(b) && !is.na(b[[1L]])) return(round(b[[1L]] / 1024^3))
  }
  NA_real_
}
.bench_detect_blas <- function() {
  lib <- tryCatch(La_library(), error = function(e) "")
  if (!nzchar(lib)) lib <- tryCatch(unname(extSoftVersion()[["BLAS"]]),
                                    error = function(e) "")
  if (grepl("Accelerate", lib, ignore.case = TRUE)) return("Accelerate")
  if (nzchar(lib)) basename(lib) else NA_character_
}
.bench_detect_os <- function() {
  r <- tryCatch(utils::sessionInfo()$running, error = function(e) NULL)
  if (!is.null(r) && nzchar(r)) return(r)
  paste(Sys.info()[["sysname"]], Sys.info()[["release"]])
}

capture_prov <- function(contributor = NULL, date = NULL, systems = list()) {
  list(
    date        = .or(date, format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
    contributor = .or(contributor, unname(Sys.info()[["user"]])),
    hardware    = list(chip   = .bench_detect_chip(),
                       cores  = .bench_detect_cores(),
                       ram_gb = .bench_detect_ram_gb()),
    env         = list(os_version = .bench_detect_os(),
                       r_version  = as.character(getRversion()),
                       blas       = .bench_detect_blas()),
    systems     = systems,
    sessionInfo = paste(utils::capture.output(utils::sessionInfo()), collapse = "\n")
  )
}

new_run_id <- function(now = Sys.time(), suffix = NULL) {
  ts <- format(now, "%Y%m%dT%H%M%OS3%z")
  if (is.null(suffix)) {
    suffix <- paste(sample(c(0:9, letters), 4L, replace = TRUE), collapse = "")
  }
  paste0(ts, "-", suffix)
}

time_repeated <- function(thunk, reps = 50L) {
  if (requireNamespace("bench", quietly = TRUE)) {
    res <- tryCatch(
      bench::mark(thunk(), iterations = reps, check = FALSE, filter_gc = FALSE),
      error = function(e) NULL)
    if (!is.null(res)) {
      times_ms <- as.numeric(res$time[[1L]]) * 1000
      q <- stats::quantile(times_ms, c(0.25, 0.5, 0.75), na.rm = TRUE)
      return(list(median_ms = unname(q[[2L]]),
                  iqr_ms    = unname(q[[3L]] - q[[1L]]),
                  cv_pct    = 100 * stats::sd(times_ms, na.rm = TRUE) /
                                    mean(times_ms, na.rm = TRUE),
                  reps      = length(times_ms)))
    }
  }
  ts <- replicate(reps, {
    t0 <- proc.time()[["elapsed"]]; thunk(); (proc.time()[["elapsed"]] - t0) * 1000
  })
  list(median_ms = stats::median(ts), iqr_ms = NA_real_, cv_pct = NA_real_,
       reps = length(ts))
}

time_once <- function(thunk) {
  t0 <- proc.time()[["elapsed"]]
  thunk()
  list(median_ms = (proc.time()[["elapsed"]] - t0) * 1000,
       iqr_ms = NA_real_, cv_pct = NA_real_, reps = 1L)
}

# --- Arguments --------------------------------------------------------------

argv        <- commandArgs(trailingOnly = TRUE)
opt_append  <- "--append" %in% argv
opt_quick   <- "--quick"  %in% argv
opt_reps    <- { i <- match("--reps", argv);        if (!is.na(i)) as.integer(argv[i + 1L]) else 50L }
opt_contrib <- { i <- match("--contributor", argv); if (!is.na(i)) argv[i + 1L] else NULL }
opt_out     <- { i <- match("--out", argv);         if (!is.na(i)) argv[i + 1L] else NULL }

# --- System detection -------------------------------------------------------

have_numDeriv <- requireNamespace("numDeriv", quietly = TRUE)
# numDeriv's grad/hessian are S3 generics that DefDiff shadows for function
# input; call numDeriv's own methods directly to avoid mis-dispatch.
nd_grad <- if (have_numDeriv) getFromNamespace("grad.default", "numDeriv") else function(...) NA_real_
nd_hess <- if (have_numDeriv) getFromNamespace("hessian.default", "numDeriv") else function(...) NA_real_

py_version <- function(mod) {
  out <- tryCatch(system2("python3",
           c("-c", shQuote(sprintf("import %s; print(%s.__version__)", mod, mod))),
           stdout = TRUE, stderr = NULL),
         error = function(e) character(), warning = function(w) character())
  if (length(out) && nzchar(out[[1L]])) out[[1L]] else NA_character_
}
torch_ver <- py_version("torch")
jax_ver   <- py_version("jax")
have_torch <- !is.na(torch_ver)
have_jax   <- !is.na(jax_ver)

defdiff_ver <- as.character(utils::packageVersion("DefDiff"))
nd_ver      <- if (have_numDeriv) as.character(utils::packageVersion("numDeriv")) else NA_character_

systems_map <- list(
  DefDiff  = defdiff_ver,
  numDeriv = if (have_numDeriv) nd_ver else NULL,
  PyTorch  = if (have_torch) torch_ver else NULL,
  JAX      = if (have_jax) jax_ver else NULL)

# --- Fixed problem set (stable problem_id) ----------------------------------

problems_grad <- list(
  sum_v2    = function(v) sum(v^2),
  sum_v3    = function(v) sum(v^3),
  sum_sin_v = function(v) sum(sin(v)))
problems_hess <- list(
  sum_v2 = function(v) sum(v^2))

grad_ns <- if (opt_quick) c(1e3) else c(1e4, 1e6)
hess_ns <- if (opt_quick) c(20)  else c(50, 100)
ND_MAX_N <- 1e4   # numDeriv finite differences are O(n) evals — cap n.

thread_levels <- list(`1` = 1L, max = NA_integer_)

set_threads <- function(k) {
  kk <- if (is.na(k)) parallel::detectCores(logical = FALSE) else k
  Sys.setenv(VECLIB_MAXIMUM_THREADS = as.character(kk),
             OMP_NUM_THREADS        = as.character(kk),
             OPENBLAS_NUM_THREADS   = as.character(kk))
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    try(RhpcBLASctl::blas_set_num_threads(kk), silent = TRUE)
  }
  invisible(kk)
}

# --- Measurement assembly ---------------------------------------------------

measurements <- list()
add_m <- function(system, system_version, operation, problem_id, n, threads,
                  parallel_capable, stage, timing) {
  measurements[[length(measurements) + 1L]] <<- list(
    system = system, system_version = system_version, operation = operation,
    problem_id = problem_id, n = as.integer(n), precision = "float64",
    threads = threads, parallel_capable = parallel_capable, stage = stage,
    median_ms = timing$median_ms, iqr_ms = timing$iqr_ms,
    cv_pct = timing$cv_pct, reps = timing$reps)
}

bench_defdiff <- function(op, problems, ns_grid, threads_label, threads_k) {
  builder <- if (op == "grad") grad else hessian
  for (pid in names(problems)) {
    f <- problems[[pid]]
    for (n in ns_grid) {
      set.seed(20260525L)
      v  <- runif(as.integer(n))
      gf <- NULL
      b  <- time_once(function() gf <<- builder(f))
      fe <- time_once(function() invisible(gf(v)))
      ev <- time_repeated(function() invisible(gf(v)), reps = opt_reps)
      total <- list(median_ms = b$median_ms + fe$median_ms,
                    iqr_ms = NA_real_, cv_pct = NA_real_, reps = 1L)
      add_m("DefDiff", defdiff_ver, op, pid, n, threads_label, TRUE, "build", b)
      add_m("DefDiff", defdiff_ver, op, pid, n, threads_label, TRUE, "first_eval", fe)
      add_m("DefDiff", defdiff_ver, op, pid, n, threads_label, TRUE, "eval", ev)
      add_m("DefDiff", defdiff_ver, op, pid, n, threads_label, TRUE, "total", total)
      invisible(gc(verbose = FALSE))
    }
  }
}

bench_numDeriv <- function(op, problems, ns_grid, threads_label) {
  if (!have_numDeriv) return(invisible())
  fn <- if (op == "grad") nd_grad else nd_hess
  reps <- min(opt_reps, 10L)
  for (pid in names(problems)) {
    f <- problems[[pid]]
    for (n in ns_grid) {
      if (n > ND_MAX_N) next
      set.seed(20260525L)
      v  <- runif(as.integer(n))
      ev <- time_repeated(function() invisible(fn(f, v)), reps = reps)
      tot <- time_once(function() invisible(fn(f, v)))
      add_m("numDeriv", nd_ver, op, pid, n, threads_label, FALSE, "eval", ev)
      add_m("numDeriv", nd_ver, op, pid, n, threads_label, FALSE, "total", tot)
      invisible(gc(verbose = FALSE))
    }
  }
}

# Optional Python systems via the community sidecar (graceful skip).
sidecar <- "community-sidecar.py"
bench_python <- function(system, backend, version, threads_label, threads_k) {
  if (!file.exists(sidecar)) return(invisible())
  if (!requireNamespace("jsonlite", quietly = TRUE)) return(invisible())
  kk <- if (is.na(threads_k)) parallel::detectCores(logical = FALSE) else threads_k
  for (pid in names(problems_grad)) {
    for (n in grad_ns) {
      out <- tryCatch(system2("python3",
               c(sidecar, "--backend", backend, "--problem", pid,
                 "--n", format(as.integer(n), scientific = FALSE),
                 "--threads", as.character(kk)),
               stdout = TRUE, stderr = NULL),
             error = function(e) character())
      if (!length(out)) next
      parsed <- tryCatch(jsonlite::fromJSON(out[[length(out)]], simplifyVector = FALSE),
                         error = function(e) NULL)
      if (is.null(parsed) || is.null(parsed$stages)) next
      for (s in parsed$stages) {
        timing <- list(median_ms = s$median_ms,
                       iqr_ms = if (is.null(s$iqr_ms)) NA_real_ else s$iqr_ms,
                       cv_pct = if (is.null(s$cv_pct)) NA_real_ else s$cv_pct,
                       reps   = if (is.null(s$reps)) 1L else s$reps)
        add_m(system, version, "grad", pid, n, threads_label, TRUE, s$stage, timing)
      }
    }
  }
}

# --- Machine-load sampler (issue #2) ----------------------------------------
# Record CPU/memory/thermal state while the grid runs, so a run taken under
# background load can be detected after the fact (replication cancels random
# noise, not systematic contamination). Background shell process — R has no
# native threads; sampler.sh is decoupled and killed by PID. Record-only.

sampler_script <- "sampler.sh"
load_tmp <- tempfile(fileext = ".jsonl")
sampler_pid <- NA_integer_
if (file.exists(sampler_script)) {
  sampler_pid <- tryCatch(
    as.integer(system2("bash", c(sampler_script, shQuote(load_tmp), "2"),
                       stdout = FALSE, stderr = FALSE, wait = FALSE)),
    error = function(e) NA_integer_)
  # Kill the sampler + clean the tmpfile even if the grid errors out.
  on.exit({
    if (!is.na(sampler_pid)) tryCatch(tools::pskill(sampler_pid), error = function(e) NULL)
    unlink(load_tmp)
  }, add = TRUE)
}

## Parse the sampler JSONL into meta$load (samples + summary). Returns NULL when
## no sampler ran or it produced nothing (graceful — load is optional provenance).
collect_load <- function(path, cadence_s = 2L) {
  if (!file.exists(path)) return(NULL)
  lines <- tryCatch(readLines(path, warn = FALSE), error = function(e) character())
  lines <- lines[nzchar(lines)]
  if (!length(lines)) return(NULL)
  if (!requireNamespace("jsonlite", quietly = TRUE)) return(NULL)
  samples <- lapply(lines, function(ln) tryCatch(jsonlite::fromJSON(ln, simplifyVector = TRUE),
                                                 error = function(e) NULL))
  samples <- Filter(Negate(is.null), samples)
  if (!length(samples)) return(NULL)
  # JSON null → R NULL, which write_json re-serializes as {} (empty object).
  # Normalize the optional cpu_speed_limit field to NA so it round-trips as null.
  samples <- lapply(samples, function(s) {
    if (is.null(s$cpu_speed_limit)) s$cpu_speed_limit <- NA_real_
    s
  })
  l1   <- vapply(samples, function(s) as.numeric(.or(s$loadavg_1m, NA_real_)), numeric(1))
  free <- vapply(samples, function(s) as.numeric(.or(s$free_mb,   NA_real_)), numeric(1))
  spd  <- vapply(samples, function(s) if (is.null(s$cpu_speed_limit)) NA_real_ else as.numeric(s$cpu_speed_limit), numeric(1))
  list(
    samples = samples,
    summary = list(
      loadavg_1m_median = stats::median(l1, na.rm = TRUE),
      free_mb_min       = suppressWarnings(min(free, na.rm = TRUE)),
      throttled_ever    = any(!is.na(spd) & spd < 100),
      n_samples         = length(samples),
      cadence_s         = cadence_s
    )
  )
}

# --- Run the grid -----------------------------------------------------------

# Generate the run id BEFORE the seeded timing loops below: the loops call
# set.seed() for reproducible inputs, which would otherwise make the random
# run-id suffix identical across runs and risk same-second collisions.
run_id <- new_run_id()

for (tl in names(thread_levels)) {
  k <- thread_levels[[tl]]
  set_threads(k)
  bench_defdiff("grad", problems_grad, grad_ns, tl, k)
  bench_defdiff("hessian", problems_hess, hess_ns, tl, k)
  bench_numDeriv("grad", problems_grad, grad_ns, tl)
  bench_numDeriv("hessian", problems_hess, hess_ns, tl)
  if (have_torch) bench_python("PyTorch", "torch", torch_ver, tl, k)
  if (have_jax)   bench_python("JAX", "jax", jax_ver, tl, k)
}

# --- Stop the load sampler + collect ----------------------------------------

if (!is.na(sampler_pid)) {
  tryCatch(tools::pskill(sampler_pid), error = function(e) NULL)
  Sys.sleep(0.2)  # let the final buffered line flush
}
load_info <- collect_load(load_tmp, cadence_s = 2L)

# --- Build and write the run-log --------------------------------------------

meta <- capture_prov(contributor = opt_contrib, systems = systems_map)
if (!is.null(load_info)) meta$load <- load_info
log <- list(schema_version = SCHEMA_VERSION, run_id = run_id,
            harness_version = HARNESS_VERSION, meta = meta,
            measurements = measurements)

out_dir <- if (!is.null(opt_out)) {
  opt_out
} else if (opt_append) {
  "community-logs"
} else {
  getwd()
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_path <- file.path(out_dir, paste0(run_id, ".json"))

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Writing the run-log requires the 'jsonlite' package. ",
       "Install it with install.packages(\"jsonlite\").", call. = FALSE)
}
jsonlite::write_json(log, out_path, auto_unbox = TRUE, pretty = TRUE,
                     na = "null", null = "null")

# --- Summary ----------------------------------------------------------------

cat("\n# Community differentiation benchmark\n\n")
cat(sprintf("- run_id      : %s\n", run_id))
cat(sprintf("- chip        : %s (%s cores, %s GB)\n",
            meta$hardware$chip, meta$hardware$cores, meta$hardware$ram_gb))
cat(sprintf("- systems     : DefDiff %s | numDeriv %s | PyTorch %s | JAX %s\n",
            defdiff_ver,
            if (have_numDeriv) nd_ver else "absent",
            if (have_torch) torch_ver else "absent",
            if (have_jax) jax_ver else "absent"))
cat(sprintf("- measurements: %d rows across %d thread setting(s)\n",
            length(measurements), length(thread_levels)))
cat(sprintf("- log written : %s\n", normalizePath(out_path)))
if (!opt_append) {
  cat("\nThis log is in the current directory for inspection. To submit it,\n")
  cat("re-run with --append (writes into ./community-logs/) and open a PR\n")
  cat("adding that one file.\n")
}

# Compact eval-stage table for a quick human read.
cat("\n## eval-stage medians (ms)\n\n")
cat("| System | Operation | Problem | n | Threads | median_ms |\n")
cat("|---|---|---|---|---|---|\n")
for (m in measurements) {
  if (identical(m$stage, "eval")) {
    cat(sprintf("| %s | %s | %s | %.0e | %s | %s |\n",
                m$system, m$operation, m$problem_id, m$n, m$threads,
                if (is.null(m$median_ms) || is.na(m$median_ms)) "NA"
                else formatC(m$median_ms, format = "f", digits = 3)))
  }
}
cat("\nDone.\n")
