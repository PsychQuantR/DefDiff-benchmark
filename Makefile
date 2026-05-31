# DefDiff-benchmark — community differentiation benchmark
#   make leaderboard   # re-parse community-logs/ -> CSV, then regenerate README table
#
# Requires the DefDiff package (remotes::install_github("PsychQuantR/DefDiff")).

.PHONY: leaderboard

leaderboard:      ## re-parse community-logs/ into the CSV and regenerate the README leaderboard
	Rscript -e 'library(DefDiff); DefDiff::parse_benchmark_logs("community-logs", out_csv = "community-benchmark.csv"); DefDiff::bench_render_leaderboard("community-benchmark.csv", "README.md")'
