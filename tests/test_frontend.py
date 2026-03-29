"""
Frontend validation tests.

Tests:
  - package.json exists and is valid JSON
  - Required dependencies are declared
  - TypeScript compiles (tsc --noEmit) if node_modules is present
  - Next.js build succeeds (marked slow — takes ~1-2 minutes)

These tests do NOT require the backend to be running.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

import pytest

log = logging.getLogger("tests.frontend")

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
NODE_MODULES = FRONTEND_DIR / "node_modules"
PACKAGE_JSON = FRONTEND_DIR / "package.json"
TSCONFIG = FRONTEND_DIR / "tsconfig.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Package.json validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPackageJson:
    def test_package_json_exists(self):
        assert PACKAGE_JSON.exists(), f"package.json not found at {PACKAGE_JSON}"

    def test_package_json_is_valid_json(self):
        try:
            data = json.loads(PACKAGE_JSON.read_text())
        except json.JSONDecodeError as e:
            pytest.fail(f"package.json is not valid JSON: {e}")
        assert isinstance(data, dict)

    def test_package_json_has_name(self):
        data = json.loads(PACKAGE_JSON.read_text())
        assert "name" in data
        assert data["name"] == "gaussiansplat-studio"

    def test_package_json_has_scripts(self):
        data = json.loads(PACKAGE_JSON.read_text())
        assert "scripts" in data
        scripts = data["scripts"]
        assert "dev" in scripts
        assert "build" in scripts

    def test_required_dependencies_declared(self):
        data = json.loads(PACKAGE_JSON.read_text())
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        required = ["next", "react", "react-dom", "typescript"]
        missing = [d for d in required if d not in deps]
        assert not missing, f"Missing dependencies: {missing}"

    def test_next_version_declared(self):
        data = json.loads(PACKAGE_JSON.read_text())
        deps = data.get("dependencies", {})
        assert "next" in deps
        log.info("next version: %s", deps["next"])

    def test_three_js_declared(self):
        data = json.loads(PACKAGE_JSON.read_text())
        deps = data.get("dependencies", {})
        assert "three" in deps, "three.js not in dependencies"

    def test_zustand_declared(self):
        data = json.loads(PACKAGE_JSON.read_text())
        deps = data.get("dependencies", {})
        assert "zustand" in deps, "zustand not in dependencies"


# ═══════════════════════════════════════════════════════════════════════════════
# Source file structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrontendStructure:
    def test_src_directory_exists(self):
        assert (FRONTEND_DIR / "src").exists()

    def test_app_directory_exists(self):
        assert (FRONTEND_DIR / "src" / "app").exists()

    def test_tsconfig_exists(self):
        assert TSCONFIG.exists(), "tsconfig.json not found"

    def test_tsconfig_is_valid_json(self):
        try:
            data = json.loads(TSCONFIG.read_text())
        except json.JSONDecodeError as e:
            pytest.fail(f"tsconfig.json is not valid JSON: {e}")
        assert isinstance(data, dict)

    def test_has_layout_file(self):
        """Next.js app router requires a root layout."""
        layouts = list((FRONTEND_DIR / "src" / "app").glob("layout.*"))
        assert len(layouts) > 0, "No layout file found in src/app/"
        log.info("Layout file: %s", layouts[0].name)

    def test_has_page_file(self):
        """Root page should exist."""
        pages = list((FRONTEND_DIR / "src" / "app").glob("page.*"))
        assert len(pages) > 0, "No page file found in src/app/"

    def test_components_directory_exists(self):
        assert (FRONTEND_DIR / "src" / "components").exists()

    def test_typescript_files_exist(self):
        ts_files = list((FRONTEND_DIR / "src").rglob("*.tsx")) + \
                   list((FRONTEND_DIR / "src").rglob("*.ts"))
        assert len(ts_files) > 0, "No TypeScript files found in src/"
        log.info("TypeScript files: %d", len(ts_files))


# ═══════════════════════════════════════════════════════════════════════════════
# TypeScript type-check (requires node_modules)
# ═══════════════════════════════════════════════════════════════════════════════

def _node_modules_present():
    return NODE_MODULES.exists() and (NODE_MODULES / ".package-lock.json").exists() or \
           (NODE_MODULES / "next").exists()


@pytest.mark.skipif(not NODE_MODULES.exists(), reason="node_modules not installed")
class TestTypeScriptCompilation:
    def test_tsc_nonemit(self):
        """TypeScript compiler should report no type errors."""
        tsc = NODE_MODULES / ".bin" / "tsc"
        if not tsc.exists():
            pytest.skip("tsc binary not found in node_modules/.bin/")

        log.info("Running: tsc --noEmit in %s", FRONTEND_DIR)
        t0 = time.time()
        result = subprocess.run(
            [str(tsc), "--noEmit"],
            cwd=str(FRONTEND_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.time() - t0
        log.info("tsc --noEmit: exit=%d (%.1fs)", result.returncode, elapsed)

        if result.returncode != 0:
            log.error("TypeScript errors:\n%s", result.stdout + result.stderr)

        assert result.returncode == 0, (
            f"TypeScript compilation failed:\n{result.stdout}\n{result.stderr}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Next.js build (slow — ~1-2 minutes)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.skipif(not NODE_MODULES.exists(), reason="node_modules not installed")
class TestNextJsBuild:
    def test_nextjs_build_succeeds(self):
        """
        Run `npm run build` in the frontend directory.
        This verifies the entire Next.js compilation pipeline succeeds.
        Marked slow — takes 1-2 minutes.
        """
        npm = "npm"
        log.info("Running: npm run build in %s", FRONTEND_DIR)
        t0 = time.time()
        result = subprocess.run(
            [npm, "run", "build"],
            cwd=str(FRONTEND_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **__import__("os").environ,
                # Suppress Next.js telemetry during testing
                "NEXT_TELEMETRY_DISABLED": "1",
                # Skip linting during build for speed
                "NEXT_LINT_IGNORE_ALL": "1",
            },
        )
        elapsed = time.time() - t0
        log.info("next build: exit=%d (%.1fs)", result.returncode, elapsed)

        if result.returncode != 0:
            # Log the last 50 lines of output for diagnostics
            output = result.stdout + result.stderr
            lines = output.splitlines()
            log.error("Build failed. Last 50 lines:\n%s", "\n".join(lines[-50:]))

        assert result.returncode == 0, (
            f"next build failed (exit {result.returncode}).\n"
            f"Last output:\n{chr(10).join((result.stdout + result.stderr).splitlines()[-30:])}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Node.js dependency check
# ═══════════════════════════════════════════════════════════════════════════════

class TestNodeEnvironment:
    def test_node_installed(self):
        """node must be available in PATH."""
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, "node not found in PATH"
        version = result.stdout.strip()
        log.info("node version: %s", version)
        # Require node >= 18 for Next.js 15
        major = int(version.lstrip("v").split(".")[0])
        assert major >= 18, f"node {version} is too old; need >= 18"

    def test_npm_installed(self):
        """npm must be available."""
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, "npm not found in PATH"
        log.info("npm version: %s", result.stdout.strip())
