# Makefile for agent-skills
# Automates validation, building, and packaging of skills

.PHONY: all validate build zip clean install help

# Default target
all: validate build zip

# Install dependencies
install:
	@echo "Installing dependencies..."
	cd packages/code-security-build && pnpm install

# Validate all rule files
validate:
	@echo "Validating rule files..."
	cd packages/code-security-build && pnpm validate

# Build the skill (runs build-agents and extract-tests)
build:
	@echo "Building skills..."
	cd packages/code-security-build && pnpm build

# Create zip files for all skills
zip:
	@echo "Creating zip packages for all skills..."
	@for skill_dir in skills/*/; do \
		skill_name=$$(basename "$$skill_dir"); \
		if [ -f "$$skill_dir/SKILL.md" ]; then \
			echo "  Packaging $$skill_name..."; \
			cd skills && rm -f "$$skill_name.zip" && zip -rq "$$skill_name.zip" "$$skill_name/" && cd ..; \
			echo "  Created skills/$$skill_name.zip"; \
		else \
			echo "  Skipping $$skill_name (no SKILL.md)"; \
		fi \
	done
	@echo "Done!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f skills/*.zip
	@echo "Done!"

# Development workflow: validate and build
dev: validate build

# Full release workflow: validate, build, and package
release: validate build zip
	@echo ""
	@echo "Release complete! Zip files created:"
	@ls -la skills/*.zip 2>/dev/null || echo "  No zip files found"

# Show help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Validate, build, and create zip packages (default)"
	@echo "  install   - Install pnpm dependencies"
	@echo "  validate  - Validate all rule files"
	@echo "  build     - Build the skill files"
	@echo "  zip       - Create zip packages for all skills"
	@echo "  clean     - Remove generated zip files"
	@echo "  dev       - Validate and build (no zip)"
	@echo "  release   - Full release: validate, build, and zip"
	@echo "  help      - Show this help message"
