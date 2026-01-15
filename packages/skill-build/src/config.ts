/**
 * Configuration for the build tooling
 *
 * Supports building any skill by accepting the skill name as CLI argument.
 * Usage: tsx src/build.ts [skill-name]
 *
 * If no skill name is provided, defaults to the SKILL_NAME env var or 'code-security'.
 */

import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { existsSync, readdirSync } from 'fs'

const __dirname = dirname(fileURLToPath(import.meta.url))

// Get skill name from CLI args, env var, or default
function getSkillName(): string {
  // Check CLI args first (skip node and script path)
  const args = process.argv.slice(2)
  if (args.length > 0 && !args[0].startsWith('-')) {
    return args[0]
  }

  // Check environment variable
  if (process.env.SKILL_NAME) {
    return process.env.SKILL_NAME
  }

  // Default
  return 'code-security'
}

// Current skill being processed
export const SKILL_NAME = getSkillName()

// Base paths
export const BUILD_DIR = join(__dirname, '..')
export const SKILLS_ROOT = join(__dirname, '../../..', 'skills')

// Skill-specific paths (computed from SKILL_NAME)
export const SKILL_DIR = join(SKILLS_ROOT, SKILL_NAME)
export const RULES_DIR = join(SKILL_DIR, 'rules')
export const METADATA_FILE = join(SKILL_DIR, 'metadata.json')
export const OUTPUT_FILE = join(SKILL_DIR, 'AGENTS.md')

// Test cases output goes to build directory, namespaced by skill
export const TEST_CASES_FILE = join(BUILD_DIR, `test-cases-${SKILL_NAME}.json`)

/**
 * Get all skills with rules directories
 */
export function getAllSkills(): string[] {
  const skills: string[] = []

  try {
    const entries = readdirSync(SKILLS_ROOT, { withFileTypes: true })
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const rulesDir = join(SKILLS_ROOT, entry.name, 'rules')
        if (existsSync(rulesDir)) {
          skills.push(entry.name)
        }
      }
    }
  } catch {
    // Return empty if skills directory doesn't exist
  }

  return skills
}

/**
 * Get paths for a specific skill (for use in batch operations)
 */
export function getSkillPaths(skillName: string) {
  const skillDir = join(SKILLS_ROOT, skillName)
  return {
    skillDir,
    rulesDir: join(skillDir, 'rules'),
    metadataFile: join(skillDir, 'metadata.json'),
    outputFile: join(skillDir, 'AGENTS.md'),
    testCasesFile: join(BUILD_DIR, `test-cases-${skillName}.json`),
  }
}

/**
 * Validate that the skill exists and has required structure
 */
export function validateSkillExists(): void {
  if (!existsSync(SKILL_DIR)) {
    console.error(`Error: Skill directory not found: ${SKILL_DIR}`)
    console.error(`Available skills: ${getAllSkills().join(', ') || 'none'}`)
    process.exit(1)
  }

  if (!existsSync(RULES_DIR)) {
    console.error(`Error: Rules directory not found: ${RULES_DIR}`)
    console.error('Skills must have a rules/ subdirectory')
    process.exit(1)
  }
}
