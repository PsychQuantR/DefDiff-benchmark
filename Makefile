# DefDiff-benchmark — community differentiation benchmark
#   make leaderboard   # re-parse community-logs/ -> CSV, regenerate README table + charts
#   make chart         # regenerate the bar charts only (from the existing CSV)
#
# Requires the DefDiff package (remotes::install_github("PsychQuantR/DefDiff"))
# plus ggplot2 + scales for the charts.

.PHONY: leaderboard chart

leaderboard:      ## re-parse community-logs/ into the CSV, regenerate the README table + charts
	Rscript -e 'library(DefDiff); DefDiff::parse_benchmark_logs("community-logs", out_csv = "community-benchmark.csv"); DefDiff::bench_render_leaderboard("community-benchmark.csv", "README.md")'
	$(MAKE) chart

chart:            ## regenerate leaderboard-eval.png + leaderboard-total.png from the CSV
	Rscript chart-leaderboard.R community-benchmark.csv .
