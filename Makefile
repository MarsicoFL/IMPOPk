.PHONY: build test clippy install clean download-data download-data-dry verify-data paper docker docker-test help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all binaries in release mode
	cargo build --release

test: ## Run all workspace tests
	cargo test --workspace

clippy: ## Run clippy lints
	cargo clippy --workspace -- -D warnings

install: ## Install binaries to ~/.cargo/bin
	cargo install --path src/ibs-cli
	cargo install --path src/ibd-cli
	cargo install --path src/ancestry-cli
	cargo install --path src/jacquard-cli

clean: ## Clean build artifacts (Rust + LaTeX)
	cargo clean
	[ -d paper ] && cd paper && rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.synctex.gz *.fdb_latexmk 2>/dev/null || true

download-data: ## Download all required external data
	bash scripts/download_all.sh

download-data-dry: ## Show what would be downloaded
	bash scripts/download_all.sh --dry-run

verify-data: ## Verify downloaded data checksums
	bash scripts/verify_checksums.sh

docker: ## Build Docker image
	docker build -t impopk .

docker-test: ## Test Docker image (all 9 binaries)
	docker run --rm impopk ibs --help
	docker run --rm impopk ibs-from-paf --help
	docker run --rm impopk ibs-from-tpa --help
	docker run --rm impopk tpa-spatial-index --help
	docker run --rm impopk tpa-validate --help
	docker run --rm impopk ibd --help
	docker run --rm impopk ibd-validate --help
	docker run --rm impopk ancestry --help
	docker run --rm impopk jacquard --help
	@echo "All 9 binaries OK"
