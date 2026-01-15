# Makefile for agent-skills
# Automates validation, building, and packaging of skills

.PHONY: all validate build zip clean install help

# Default target
all: validate build zip

# Install dependencies
install:
	@echo "Installing dependencies..."
	cd packages/skill-build && pnpm install

# Validate all skills with rules directories
validate:
	@echo "Validating all skills..."
	@for skill_dir in skills/*/; do \
		skill_name=$$(basename "$$skill_dir"); \
		if [ -d "$$skill_dir/rules" ]; then \
			echo ""; \
			echo "Validating $$skill_name..."; \
			cd packages/skill-build && pnpm validate "$$skill_name" && cd ../..; \
		fi \
	done
	@echo ""
	@echo "Done validating all skills!"

# Build all skills with rules directories
build:
	@echo "Building all skills..."
	@for skill_dir in skills/*/; do \
		skill_name=$$(basename "$$skill_dir"); \
		if [ -d "$$skill_dir/rules" ]; then \
			echo ""; \
			echo "Building $$skill_name..."; \
			cd packages/skill-build && pnpm build-agents "$$skill_name" && pnpm extract-tests "$$skill_name" && cd ../..; \
		fi \
	done
	@echo ""
	@echo "Done building all skills!"

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
	rm -f packages/skill-build/test-cases-*.json
	@echo "Done!"

# Development workflow: validate and build
dev: validate build

# Full release workflow: validate, build, and package
release: validate build zip
	@echo ""
	@echo "Release complete! Zip files created:"
	@ls -la skills/*.zip 2>/dev/null || echo "  No zip files found"

# Validate a single skill: make validate-skill SKILL=code-security
validate-skill:
ifndef SKILL
	$(error SKILL is required. Usage: make validate-skill SKILL=code-security)
endif
	@echo "Validating $(SKILL)..."
	cd packages/skill-build && pnpm validate "$(SKILL)"

# Build a single skill: make build-skill SKILL=code-security
build-skill:
ifndef SKILL
	$(error SKILL is required. Usage: make build-skill SKILL=code-security)
endif
	@echo "Building $(SKILL)..."
	cd packages/skill-build && pnpm build-agents "$(SKILL)" && pnpm extract-tests "$(SKILL)"

# Show help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Validate, build, and create zip packages (default)"
	@echo "  install        - Install pnpm dependencies"
	@echo "  validate       - Validate all skills with rules directories"
	@echo "  build          - Build AGENTS.md for all skills with rules"
	@echo "  zip            - Create zip packages for all skills"
	@echo "  clean          - Remove generated files"
	@echo "  dev            - Validate and build (no zip)"
	@echo "  release        - Full release: validate, build, and zip"
	@echo ""
	@echo "Single skill targets:"
	@echo "  validate-skill - Validate one skill: make validate-skill SKILL=name"
	@echo "  build-skill    - Build one skill: make build-skill SKILL=name"
	@echo ""
	@echo "  help           - Show this help message"
